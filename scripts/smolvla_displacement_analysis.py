#!/usr/bin/env python3
"""
SmolVLA Cross-Task Displacement Analysis

Determines whether injecting task A's activations into task B causes the robot
to perform task A instead of task B. Adapted from xvla_displacement_analysis.py
for SmolVLA's MetaWorld cross-task injection data format.

Uses:
- Cosine similarity of injected action trajectory to source vs destination baselines
- TCP (end-effector) trajectory similarity
- Object displacement analysis (single-object MetaWorld scenes)
- Per-group breakdown (expert vs VLM, early/mid/late layers)

Outputs:
- SMOLVLA_DISPLACEMENT_ANALYSIS.md: Markdown with tables
- smolvla_displacement_analysis.json: Machine-readable results

Usage:
    # Analyze a single difficulty
    python scripts/smolvla_displacement_analysis.py --data-dir rollouts/smolvla/metaworld_cross_task/easy

    # Analyze all difficulties
    python scripts/smolvla_displacement_analysis.py --data-dir rollouts/smolvla/metaworld_cross_task

    # With baseline data for richer analysis
    python scripts/smolvla_displacement_analysis.py --data-dir rollouts/smolvla/metaworld_cross_task \
        --baseline-dir rollouts/smolvla/metaworld_baseline

    # Custom output directory
    python scripts/smolvla_displacement_analysis.py --data-dir rollouts/smolvla/metaworld_cross_task \
        --output-dir results/smolvla_displacement
"""

import json
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ============================================================
# UTILITIES
# ============================================================

def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return p_hat * 100, max(0, center - margin) * 100, min(1, center + margin) * 100


def format_rate(successes, total):
    """Format a rate with Wilson CI."""
    rate, lo, hi = wilson_ci(successes, total)
    return f"{rate:.1f}% ({successes}/{total}) [CI: {lo:.1f}-{hi:.1f}%]"


def load_json(path):
    """Load JSON, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def cosine_similarity(a, b):
    """Cosine similarity between two action sequences (as flat vectors)."""
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    # Truncate to same length
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    a, b = a[:min_len], b[:min_len]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def trajectory_distance(traj_a, traj_b):
    """Mean Euclidean distance between two TCP trajectories (3D positions)."""
    a = np.array(traj_a)
    b = np.array(traj_b)
    min_len = min(len(a), len(b))
    if min_len == 0:
        return float('inf')
    a, b = a[:min_len, :3], b[:min_len, :3]  # use xyz only
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def classify_behavior(cos_to_src, cos_to_dst, threshold=0.05):
    """Classify whether robot performed source task, destination task, or neither.

    Returns: 'source', 'destination', or 'ambiguous'
    """
    if cos_to_src is None or cos_to_dst is None:
        return 'ambiguous'
    diff = cos_to_src - cos_to_dst
    if diff > threshold:
        return 'source'
    elif diff < -threshold:
        return 'destination'
    else:
        return 'ambiguous'


def extract_tcp_trajectory(scene_states):
    """Extract TCP positions from scene_states list."""
    positions = []
    for s in scene_states:
        if isinstance(s, dict) and 'tcp_pos' in s:
            positions.append(s['tcp_pos'])
    return positions


def compute_object_displacement(scene_states):
    """Compute total object displacement from scene_states."""
    if not scene_states or len(scene_states) < 2:
        return 0.0, None, None
    first = scene_states[0]
    last = scene_states[-1]
    if not isinstance(first, dict) or not isinstance(last, dict):
        return 0.0, None, None
    obj_start = first.get('obj_pos')
    obj_end = last.get('obj_pos')
    if obj_start is None or obj_end is None:
        return 0.0, obj_start, obj_end
    disp = np.linalg.norm(np.array(obj_end) - np.array(obj_start))
    return float(disp), obj_start, obj_end


def compute_goal_distance(scene_states):
    """Compute final distance from object to goal."""
    if not scene_states:
        return float('inf')
    last = scene_states[-1]
    if not isinstance(last, dict):
        return float('inf')
    obj = last.get('obj_pos')
    goal = last.get('goal_pos')
    if obj is None or goal is None:
        return float('inf')
    return float(np.linalg.norm(np.array(obj) - np.array(goal)))


# ============================================================
# PAIR ANALYSIS
# ============================================================

INJECTION_GROUPS = [
    'expert_all', 'expert_early', 'expert_mid', 'expert_late',
    'vlm_all', 'vlm_early', 'vlm_mid', 'vlm_late',
]


def analyze_pair(pair_data, traj_dir=None):
    """Analyze a single SmolVLA cross-task pair for displacement behavior.

    Args:
        pair_data: Dict from cross_task_*.json (summary file)
        traj_dir: Optional path to trajectories/cross_task_*/ directory
                  for richer analysis with scene_states

    Returns:
        List of result dicts, one per injection condition + direction
    """
    results = []

    task_a = pair_data.get('task_a', '')
    task_b = pair_data.get('task_b', '')
    desc_a = pair_data.get('task_a_desc', task_a)
    desc_b = pair_data.get('task_b_desc', task_b)

    # Get baseline action sequences
    baseline_A = pair_data.get('baseline_A', {})
    baseline_B = pair_data.get('baseline_B', {})
    actions_A = baseline_A.get('actions', [])
    actions_B = baseline_B.get('actions', [])

    # Load detailed trajectory data if available
    traj_baseline_A = None
    traj_baseline_B = None
    if traj_dir and traj_dir.exists():
        traj_baseline_A = load_json(traj_dir / 'baseline_A.json')
        traj_baseline_B = load_json(traj_dir / 'baseline_B.json')

    # Compute baseline scene metrics
    baseline_A_obj_disp = 0.0
    baseline_B_obj_disp = 0.0
    baseline_A_goal_dist = float('inf')
    baseline_B_goal_dist = float('inf')
    baseline_A_tcp = None
    baseline_B_tcp = None

    if traj_baseline_A and 'scene_states' in traj_baseline_A:
        baseline_A_obj_disp, _, _ = compute_object_displacement(traj_baseline_A['scene_states'])
        baseline_A_goal_dist = compute_goal_distance(traj_baseline_A['scene_states'])
        baseline_A_tcp = extract_tcp_trajectory(traj_baseline_A['scene_states'])

    if traj_baseline_B and 'scene_states' in traj_baseline_B:
        baseline_B_obj_disp, _, _ = compute_object_displacement(traj_baseline_B['scene_states'])
        baseline_B_goal_dist = compute_goal_distance(traj_baseline_B['scene_states'])
        baseline_B_tcp = extract_tcp_trajectory(traj_baseline_B['scene_states'])

    # Process each injection direction and group
    for group in INJECTION_GROUPS:
        for direction in ['A_into_B', 'B_into_A']:
            key = f'inject_{direction}_{group}'
            inject_data = pair_data.get(key, {})
            if not inject_data:
                continue

            inject_actions = inject_data.get('actions', [])
            inject_success = inject_data.get('success', False)
            inject_n_steps = inject_data.get('n_steps', 0)
            total_injections = inject_data.get('total_injections', 0)

            # Determine source and destination
            if direction == 'A_into_B':
                src_task, dst_task = task_a, task_b
                src_desc, dst_desc = desc_a, desc_b
                src_actions, dst_actions = actions_A, actions_B
                src_baseline_success = baseline_A.get('success', False)
                dst_baseline_success = baseline_B.get('success', False)
                src_obj_disp = baseline_A_obj_disp
                dst_obj_disp = baseline_B_obj_disp
                src_tcp = baseline_A_tcp
                dst_tcp = baseline_B_tcp
            else:
                src_task, dst_task = task_b, task_a
                src_desc, dst_desc = desc_b, desc_a
                src_actions, dst_actions = actions_B, actions_A
                src_baseline_success = baseline_B.get('success', False)
                dst_baseline_success = baseline_A.get('success', False)
                src_obj_disp = baseline_B_obj_disp
                dst_obj_disp = baseline_A_obj_disp
                src_tcp = baseline_B_tcp
                dst_tcp = baseline_A_tcp

            # Compute cosine similarities to both baselines
            cos_to_src = cosine_similarity(inject_actions, src_actions) if src_actions and inject_actions else None
            cos_to_dst = cosine_similarity(inject_actions, dst_actions) if dst_actions and inject_actions else None

            # Pre-computed cosine to target (destination) baseline from the experiment
            precomputed_cos = inject_data.get('cosine_sim_with_target_baseline')

            behavior = classify_behavior(cos_to_src, cos_to_dst)

            result = {
                'src_task': src_task,
                'dst_task': dst_task,
                'src_desc': src_desc,
                'dst_desc': dst_desc,
                'direction': direction,
                'group': group,
                'condition': group,  # for compatibility with aggregate_by_condition
                'success': inject_success,
                'n_steps': inject_n_steps,
                'total_injections': total_injections,
                'cos_to_src': cos_to_src,
                'cos_to_dst': cos_to_dst,
                'cos_to_dst_precomputed': precomputed_cos,
                'behavior': behavior,
                'src_baseline_success': src_baseline_success,
                'dst_baseline_success': dst_baseline_success,
            }

            # Add scene-level analysis from trajectory data if available
            inject_traj_file = f'inject_{direction}_{group}.json'
            inject_traj = None
            if traj_dir and traj_dir.exists():
                inject_traj = load_json(traj_dir / inject_traj_file)

            if inject_traj and 'scene_states' in inject_traj:
                ss = inject_traj['scene_states']
                obj_disp, obj_start, obj_end = compute_object_displacement(ss)
                goal_dist = compute_goal_distance(ss)
                tcp_traj = extract_tcp_trajectory(ss)

                result['obj_displacement'] = obj_disp
                result['final_goal_distance'] = goal_dist
                result['src_baseline_obj_disp'] = src_obj_disp
                result['dst_baseline_obj_disp'] = dst_obj_disp

                # TCP trajectory similarity to source vs destination baselines
                if src_tcp and tcp_traj:
                    result['tcp_dist_to_src'] = trajectory_distance(tcp_traj, src_tcp)
                if dst_tcp and tcp_traj:
                    result['tcp_dist_to_dst'] = trajectory_distance(tcp_traj, dst_tcp)

            results.append(result)

    return results


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_by_condition(all_results):
    """Aggregate displacement results by injection group."""
    by_condition = defaultdict(lambda: {
        'total': 0, 'source': 0, 'destination': 0, 'ambiguous': 0,
        'successes': 0, 'cos_src_vals': [], 'cos_dst_vals': [],
        'obj_disp_vals': [], 'goal_dist_vals': [],
        'tcp_dist_src_vals': [], 'tcp_dist_dst_vals': [],
    })

    for r in all_results:
        cond = r['condition']
        agg = by_condition[cond]
        agg['total'] += 1
        agg[r['behavior']] += 1
        if r['success']:
            agg['successes'] += 1
        if r.get('cos_to_src') is not None:
            agg['cos_src_vals'].append(r['cos_to_src'])
        if r.get('cos_to_dst') is not None:
            agg['cos_dst_vals'].append(r['cos_to_dst'])
        if r.get('obj_displacement') is not None:
            agg['obj_disp_vals'].append(r['obj_displacement'])
        if r.get('final_goal_distance') is not None and r['final_goal_distance'] != float('inf'):
            agg['goal_dist_vals'].append(r['final_goal_distance'])
        if r.get('tcp_dist_to_src') is not None:
            agg['tcp_dist_src_vals'].append(r['tcp_dist_to_src'])
        if r.get('tcp_dist_to_dst') is not None:
            agg['tcp_dist_dst_vals'].append(r['tcp_dist_to_dst'])

    return by_condition


def aggregate_by_component(all_results):
    """Aggregate by expert vs VLM component."""
    by_component = defaultdict(lambda: {
        'total': 0, 'source': 0, 'destination': 0, 'ambiguous': 0,
        'successes': 0, 'cos_src_vals': [], 'cos_dst_vals': [],
    })

    for r in all_results:
        group = r['group']
        component = 'expert' if group.startswith('expert') else 'vlm'
        agg = by_component[component]
        agg['total'] += 1
        agg[r['behavior']] += 1
        if r['success']:
            agg['successes'] += 1
        if r.get('cos_to_src') is not None:
            agg['cos_src_vals'].append(r['cos_to_src'])
        if r.get('cos_to_dst') is not None:
            agg['cos_dst_vals'].append(r['cos_to_dst'])

    return by_component


def aggregate_by_layer_group(all_results):
    """Aggregate by layer group (all/early/mid/late)."""
    by_layer = defaultdict(lambda: {
        'total': 0, 'source': 0, 'destination': 0, 'ambiguous': 0,
        'successes': 0, 'cos_src_vals': [], 'cos_dst_vals': [],
    })

    for r in all_results:
        group = r['group']
        # Extract layer position: all, early, mid, late
        parts = group.split('_')
        layer_pos = parts[-1] if len(parts) >= 2 else 'unknown'
        agg = by_layer[layer_pos]
        agg['total'] += 1
        agg[r['behavior']] += 1
        if r['success']:
            agg['successes'] += 1
        if r.get('cos_to_src') is not None:
            agg['cos_src_vals'].append(r['cos_to_src'])
        if r.get('cos_to_dst') is not None:
            agg['cos_dst_vals'].append(r['cos_to_dst'])

    return by_layer


# ============================================================
# BASELINE ANALYSIS
# ============================================================

def load_baseline_stats(baseline_dir):
    """Load baseline statistics from metaworld_baseline directory."""
    stats = {}
    if not baseline_dir or not baseline_dir.exists():
        return stats

    traj_dir = baseline_dir / 'trajectories'
    if not traj_dir.exists():
        return stats

    for task_dir in sorted(traj_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        episodes = []
        for ep_file in sorted(task_dir.glob('ep*.json')):
            ep_data = load_json(ep_file)
            if ep_data:
                episodes.append({
                    'success': ep_data.get('success', False),
                    'n_steps': ep_data.get('n_steps', 0),
                    'obj_displacement': np.linalg.norm(ep_data['obj_displacement']).item()
                        if 'obj_displacement' in ep_data else 0.0,
                })
        if episodes:
            n_succ = sum(1 for e in episodes if e['success'])
            stats[task_name] = {
                'n_episodes': len(episodes),
                'success_rate': n_succ / len(episodes),
                'mean_steps': np.mean([e['n_steps'] for e in episodes]),
                'mean_obj_disp': np.mean([e['obj_displacement'] for e in episodes]),
            }

    return stats


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(all_difficulty_results, baseline_stats, output_dir):
    """Generate markdown report and JSON output."""

    lines = []
    lines.append("# SmolVLA Cross-Task Displacement Analysis")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append(f"\n*Model: SmolVLA MetaWorld (`jadechoghari/smolvla_metaworld`, 450M params)*")
    lines.append("\n**Question:** When we inject task A's MLP activations into task B's environment,")
    lines.append("does the robot attempt to perform task A (source) instead of task B (destination)?")
    lines.append("\n**Architecture:** 32 expert layers (hidden=480) + 32 VLM layers (hidden=960), interleaved.")
    lines.append("Injection groups: `expert_all/early/mid/late` and `vlm_all/early/mid/late` (8 groups).")
    lines.append("\n**Classification method:** Compare cosine similarity of the injected episode's")
    lines.append("action trajectory to the source baseline vs destination baseline.")
    lines.append("- **Source behavior**: cos(src) - cos(dst) > 0.05 (robot follows source task)")
    lines.append("- **Destination behavior**: cos(dst) - cos(src) > 0.05 (robot follows own task)")
    lines.append("- **Ambiguous**: |cos(src) - cos(dst)| <= 0.05\n")

    json_output = {'model': 'jadechoghari/smolvla_metaworld', 'difficulties': {}}

    # ─── Grand Summary ───
    lines.append("## Grand Summary\n")
    lines.append("| Difficulty | Pairs | Total Injections | Source Behavior | Dest Behavior | Ambiguous | Dest Success |")
    lines.append("|------------|-------|-----------------|-----------------|---------------|-----------|--------------|")

    all_results_flat = []
    for diff_name in sorted(all_difficulty_results.keys()):
        results = all_difficulty_results[diff_name]
        all_results_flat.extend(results)

        total = len(results)
        n_pairs = len(set((r['src_task'], r['dst_task']) for r in results)) if results else 0
        source = sum(1 for r in results if r['behavior'] == 'source')
        dest = sum(1 for r in results if r['behavior'] == 'destination')
        ambig = sum(1 for r in results if r['behavior'] == 'ambiguous')
        succ = sum(1 for r in results if r['success'])

        lines.append(f"| {diff_name} | {n_pairs} | {total} | {format_rate(source, total)} | {format_rate(dest, total)} | {format_rate(ambig, total)} | {format_rate(succ, total)} |")

    # Overall row
    if len(all_difficulty_results) > 1 and all_results_flat:
        total = len(all_results_flat)
        n_pairs = len(set((r['src_task'], r['dst_task']) for r in all_results_flat))
        source = sum(1 for r in all_results_flat if r['behavior'] == 'source')
        dest = sum(1 for r in all_results_flat if r['behavior'] == 'destination')
        ambig = sum(1 for r in all_results_flat if r['behavior'] == 'ambiguous')
        succ = sum(1 for r in all_results_flat if r['success'])
        lines.append(f"| **ALL** | {n_pairs} | {total} | {format_rate(source, total)} | {format_rate(dest, total)} | {format_rate(ambig, total)} | {format_rate(succ, total)} |")

    # ─── Per-Difficulty Details ───
    for diff_name in sorted(all_difficulty_results.keys()):
        results = all_difficulty_results[diff_name]
        if not results:
            continue

        lines.append(f"\n---\n\n## {diff_name}\n")

        # Per-group breakdown
        by_cond = aggregate_by_condition(results)

        lines.append("### Per-Group Displacement\n")
        lines.append("| Group | N | Source% | Dest% | Ambig% | Succ% | Mean cos->src | Mean cos->dst | cos->src > cos->dst? |")
        lines.append("|-------|---|---------|-------|--------|-------|---------------|---------------|---------------------|")

        for group in INJECTION_GROUPS:
            if group not in by_cond:
                continue
            agg = by_cond[group]
            n = agg['total']
            if n == 0:
                continue
            src_pct = agg['source'] / n * 100
            dst_pct = agg['destination'] / n * 100
            amb_pct = agg['ambiguous'] / n * 100
            succ_pct = agg['successes'] / n * 100
            mean_cos_src = np.mean(agg['cos_src_vals']) if agg['cos_src_vals'] else 0
            mean_cos_dst = np.mean(agg['cos_dst_vals']) if agg['cos_dst_vals'] else 0
            dominant = "YES" if mean_cos_src > mean_cos_dst else "no"

            lines.append(f"| {group} | {n} | {src_pct:.1f}% | {dst_pct:.1f}% | {amb_pct:.1f}% | {succ_pct:.1f}% | {mean_cos_src:.4f} | {mean_cos_dst:.4f} | {dominant} |")

        # Expert vs VLM comparison
        by_comp = aggregate_by_component(results)
        if len(by_comp) == 2:
            lines.append("\n### Expert vs VLM Component\n")
            lines.append("| Component | N | Source% | Dest% | Ambig% | Succ% | Mean cos->src | Mean cos->dst |")
            lines.append("|-----------|---|---------|-------|--------|-------|---------------|---------------|")

            for comp in ['expert', 'vlm']:
                if comp not in by_comp:
                    continue
                agg = by_comp[comp]
                n = agg['total']
                if n == 0:
                    continue
                src_pct = agg['source'] / n * 100
                dst_pct = agg['destination'] / n * 100
                amb_pct = agg['ambiguous'] / n * 100
                succ_pct = agg['successes'] / n * 100
                mean_src = np.mean(agg['cos_src_vals']) if agg['cos_src_vals'] else 0
                mean_dst = np.mean(agg['cos_dst_vals']) if agg['cos_dst_vals'] else 0
                lines.append(f"| {comp} | {n} | {src_pct:.1f}% | {dst_pct:.1f}% | {amb_pct:.1f}% | {succ_pct:.1f}% | {mean_src:.4f} | {mean_dst:.4f} |")

        # Layer position comparison (all/early/mid/late)
        by_layer = aggregate_by_layer_group(results)
        lines.append("\n### Layer Position (all/early/mid/late)\n")
        lines.append("| Position | N | Source% | Dest% | Ambig% | Mean cos->src | Mean cos->dst |")
        lines.append("|----------|---|---------|-------|--------|---------------|---------------|")

        for pos in ['all', 'early', 'mid', 'late']:
            if pos not in by_layer:
                continue
            agg = by_layer[pos]
            n = agg['total']
            if n == 0:
                continue
            src_pct = agg['source'] / n * 100
            dst_pct = agg['destination'] / n * 100
            amb_pct = agg['ambiguous'] / n * 100
            mean_src = np.mean(agg['cos_src_vals']) if agg['cos_src_vals'] else 0
            mean_dst = np.mean(agg['cos_dst_vals']) if agg['cos_dst_vals'] else 0
            lines.append(f"| {pos} | {n} | {src_pct:.1f}% | {dst_pct:.1f}% | {amb_pct:.1f}% | {mean_src:.4f} | {mean_dst:.4f} |")

        # Scene-level analysis (if trajectory data available)
        results_with_scene = [r for r in results if r.get('obj_displacement') is not None]
        if results_with_scene:
            lines.append("\n### Scene-Level Analysis\n")
            lines.append("Object displacement and TCP trajectory analysis from scene_states.\n")

            # Per-group scene metrics
            lines.append("| Group | Mean Obj Disp | Mean Goal Dist | Mean TCP->src | Mean TCP->dst |")
            lines.append("|-------|--------------|----------------|--------------|--------------|")

            for group in INJECTION_GROUPS:
                grp_results = [r for r in results_with_scene if r['group'] == group]
                if not grp_results:
                    continue
                mean_obj = np.mean([r['obj_displacement'] for r in grp_results])
                goal_dists = [r['final_goal_distance'] for r in grp_results
                              if r.get('final_goal_distance') is not None and r['final_goal_distance'] != float('inf')]
                mean_goal = np.mean(goal_dists) if goal_dists else float('inf')
                tcp_src = [r['tcp_dist_to_src'] for r in grp_results if r.get('tcp_dist_to_src') is not None]
                tcp_dst = [r['tcp_dist_to_dst'] for r in grp_results if r.get('tcp_dist_to_dst') is not None]
                mean_tcp_src = np.mean(tcp_src) if tcp_src else float('inf')
                mean_tcp_dst = np.mean(tcp_dst) if tcp_dst else float('inf')

                goal_str = f"{mean_goal:.4f}" if mean_goal != float('inf') else "-"
                tcp_src_str = f"{mean_tcp_src:.4f}" if mean_tcp_src != float('inf') else "-"
                tcp_dst_str = f"{mean_tcp_dst:.4f}" if mean_tcp_dst != float('inf') else "-"
                lines.append(f"| {group} | {mean_obj:.4f} | {goal_str} | {tcp_src_str} | {tcp_dst_str} |")

            # Behavior vs object displacement correlation
            source_results = [r for r in results_with_scene if r['behavior'] == 'source']
            dest_results = [r for r in results_with_scene if r['behavior'] == 'destination']

            if source_results and dest_results:
                lines.append("\n**Object displacement by behavior classification:**\n")
                lines.append("| Behavior | N | Mean Obj Disp | Mean Goal Dist |")
                lines.append("|----------|---|--------------|----------------|")

                for label, subset in [('source', source_results), ('destination', dest_results),
                                       ('ambiguous', [r for r in results_with_scene if r['behavior'] == 'ambiguous'])]:
                    if not subset:
                        continue
                    mean_obj = np.mean([r['obj_displacement'] for r in subset])
                    goal_dists = [r['final_goal_distance'] for r in subset
                                  if r.get('final_goal_distance') is not None and r['final_goal_distance'] != float('inf')]
                    mean_goal = np.mean(goal_dists) if goal_dists else float('inf')
                    goal_str = f"{mean_goal:.4f}" if mean_goal != float('inf') else "-"
                    lines.append(f"| {label} | {len(subset)} | {mean_obj:.4f} | {goal_str} |")

        # Top displaced pairs (most source-like behavior)
        pair_behaviors = defaultdict(lambda: {'source': 0, 'dest': 0, 'ambig': 0, 'total': 0})
        for r in results:
            pk = f"{r['src_task']} -> {r['dst_task']}"
            pb = pair_behaviors[pk]
            pb['total'] += 1
            if r['behavior'] == 'source':
                pb['source'] += 1
            elif r['behavior'] == 'destination':
                pb['dest'] += 1
            else:
                pb['ambig'] += 1

        # Sort by source rate
        top_displaced = sorted(pair_behaviors.items(),
                                key=lambda x: x[1]['source'] / max(x[1]['total'], 1), reverse=True)

        if top_displaced:
            lines.append("\n### Top 10 Most Displaced Pairs\n")
            lines.append("Pairs where injected activations most strongly caused source-task behavior.\n")
            lines.append("| Pair (src -> dst) | N | Source% | Dest% |")
            lines.append("|-------------------|---|---------|-------|")

            for pair_key, pb in top_displaced[:10]:
                n = pb['total']
                src_pct = pb['source'] / n * 100
                dst_pct = pb['dest'] / n * 100
                lines.append(f"| {pair_key} | {n} | {src_pct:.0f}% | {dst_pct:.0f}% |")

        # Build JSON for this difficulty
        json_diff = {
            'total_episodes': len(results),
            'n_pairs': len(set((r['src_task'], r['dst_task']) for r in results)),
            'per_group': {},
            'per_component': {},
            'per_layer_position': {},
        }

        for group, agg in by_cond.items():
            n = agg['total']
            json_diff['per_group'][group] = {
                'total': n,
                'source_behavior': agg['source'],
                'destination_behavior': agg['destination'],
                'ambiguous': agg['ambiguous'],
                'successes': agg['successes'],
                'mean_cos_to_src': float(np.mean(agg['cos_src_vals'])) if agg['cos_src_vals'] else None,
                'mean_cos_to_dst': float(np.mean(agg['cos_dst_vals'])) if agg['cos_dst_vals'] else None,
            }

        for comp, agg in by_comp.items():
            n = agg['total']
            json_diff['per_component'][comp] = {
                'total': n,
                'source_behavior': agg['source'],
                'destination_behavior': agg['destination'],
                'ambiguous': agg['ambiguous'],
                'successes': agg['successes'],
                'mean_cos_to_src': float(np.mean(agg['cos_src_vals'])) if agg['cos_src_vals'] else None,
                'mean_cos_to_dst': float(np.mean(agg['cos_dst_vals'])) if agg['cos_dst_vals'] else None,
            }

        for pos, agg in by_layer.items():
            n = agg['total']
            json_diff['per_layer_position'][pos] = {
                'total': n,
                'source_behavior': agg['source'],
                'destination_behavior': agg['destination'],
                'ambiguous': agg['ambiguous'],
            }

        json_output['difficulties'][diff_name] = json_diff

    # ─── Key Findings ───
    lines.append("\n---\n\n## Key Findings\n")

    if all_results_flat:
        total = len(all_results_flat)
        source = sum(1 for r in all_results_flat if r['behavior'] == 'source')
        dest = sum(1 for r in all_results_flat if r['behavior'] == 'destination')
        ambig = sum(1 for r in all_results_flat if r['behavior'] == 'ambiguous')

        lines.append(f"1. **Overall:** Across {total} injection episodes, the robot follows the source task's")
        lines.append(f"   trajectory {format_rate(source, total)}, destination task {format_rate(dest, total)}, ambiguous {format_rate(ambig, total)}.")

        # Cosine dominance
        src_dominant = sum(1 for r in all_results_flat
                           if r.get('cos_to_src') is not None and r.get('cos_to_dst') is not None
                           and r['cos_to_src'] > r['cos_to_dst'])
        with_both = sum(1 for r in all_results_flat
                        if r.get('cos_to_src') is not None and r.get('cos_to_dst') is not None)
        if with_both > 0:
            lines.append(f"\n2. **Cosine dominance:** cos->src > cos->dst in {format_rate(src_dominant, with_both)} of episodes.")

        # Expert vs VLM comparison
        by_comp_all = aggregate_by_component(all_results_flat)
        if 'expert' in by_comp_all and 'vlm' in by_comp_all:
            exp_n = by_comp_all['expert']['total']
            exp_src = by_comp_all['expert']['source']
            vlm_n = by_comp_all['vlm']['total']
            vlm_src = by_comp_all['vlm']['source']
            exp_rate = exp_src / exp_n * 100 if exp_n else 0
            vlm_rate = vlm_src / vlm_n * 100 if vlm_n else 0
            stronger = "Expert" if exp_rate > vlm_rate else "VLM"
            lines.append(f"\n3. **Expert vs VLM:** Expert source rate = {exp_rate:.1f}%, VLM source rate = {vlm_rate:.1f}%.")
            lines.append(f"   {stronger} layers cause stronger displacement when injected.")

        # Layer position analysis
        by_layer_all = aggregate_by_layer_group(all_results_flat)
        if by_layer_all:
            lines.append(f"\n4. **Layer position effects:**")
            for pos in ['all', 'early', 'mid', 'late']:
                if pos in by_layer_all:
                    agg = by_layer_all[pos]
                    n = agg['total']
                    if n > 0:
                        src_pct = agg['source'] / n * 100
                        lines.append(f"   - {pos}: {src_pct:.1f}% source behavior ({agg['source']}/{n})")

    # ─── Baseline Reference ───
    if baseline_stats:
        lines.append("\n---\n\n## Baseline Reference\n")
        lines.append("Task performance without any injection (from metaworld_baseline).\n")
        lines.append("| Task | Success Rate | Mean Steps | Mean Obj Displacement |")
        lines.append("|------|-------------|------------|----------------------|")
        for task in sorted(baseline_stats.keys()):
            bs = baseline_stats[task]
            lines.append(f"| {task} | {bs['success_rate']*100:.0f}% ({int(bs['success_rate']*bs['n_episodes'])}/{bs['n_episodes']}) | {bs['mean_steps']:.0f} | {bs['mean_obj_disp']:.4f} |")

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "SMOLVLA_DISPLACEMENT_ANALYSIS.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Wrote: {md_path}")

    json_path = output_dir / "smolvla_displacement_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"Wrote: {json_path}")

    return md_path, json_path


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SmolVLA Cross-Task Displacement Analysis")
    parser.add_argument("--data-dir", type=str,
                        default="rollouts/smolvla/metaworld_cross_task",
                        help="Path to cross-task data. Can be a single difficulty dir or parent with easy/medium/hard/very_hard subdirs.")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Path to baseline data (rollouts/smolvla/metaworld_baseline)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for reports. Defaults to data-dir.")
    parser.add_argument("--difficulties", type=str, nargs="+", default=None,
                        help="Specific difficulties to analyze (e.g., easy medium)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    print(f"SmolVLA Cross-Task Displacement Analysis")
    print(f"Data: {data_dir}")
    print("=" * 60)

    # Detect structure: single difficulty dir or parent with subdirs
    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    if args.difficulties:
        difficulties = args.difficulties

    all_difficulty_results = {}

    # Check if data_dir itself contains cross_task_*.json files (single difficulty)
    direct_jsons = list(data_dir.glob("cross_task_*.json"))
    if direct_jsons:
        # Single difficulty directory
        diff_name = data_dir.name
        print(f"\n[{diff_name}] Processing {len(direct_jsons)} pair files...")
        results = process_difficulty(data_dir)
        if results:
            all_difficulty_results[diff_name] = results
            print(f"  {len(results)} injection episodes analyzed")
    else:
        # Parent directory with difficulty subdirs
        for diff in difficulties:
            diff_dir = data_dir / diff
            if not diff_dir.exists():
                continue
            pair_files = list(diff_dir.glob("cross_task_*.json"))
            if not pair_files:
                continue
            print(f"\n[{diff}] Processing {len(pair_files)} pair files...")
            results = process_difficulty(diff_dir)
            if results:
                all_difficulty_results[diff] = results
                print(f"  {len(results)} injection episodes analyzed")

    if not all_difficulty_results:
        print("\nNo cross-task data found!")
        return

    # Load baseline stats
    baseline_stats = {}
    if baseline_dir:
        print(f"\nLoading baseline stats from {baseline_dir}...")
        baseline_stats = load_baseline_stats(baseline_dir)
        print(f"  {len(baseline_stats)} tasks loaded")

    # Auto-detect baseline dir if not specified
    if not baseline_stats:
        auto_baseline = data_dir.parent / 'metaworld_baseline'
        if auto_baseline.exists():
            print(f"\nAuto-detected baseline dir: {auto_baseline}")
            baseline_stats = load_baseline_stats(auto_baseline)
            print(f"  {len(baseline_stats)} tasks loaded")

    # Generate report
    print(f"\n{'=' * 60}")
    print("Generating displacement analysis report...")
    generate_report(all_difficulty_results, baseline_stats, output_dir)


def process_difficulty(diff_dir):
    """Process all pair files in a single difficulty directory."""
    results = []
    traj_base = diff_dir / 'trajectories'

    for pair_file in sorted(diff_dir.glob("cross_task_*.json")):
        pair_data = load_json(pair_file)
        if not pair_data:
            continue

        # Find matching trajectory directory
        pair_stem = pair_file.stem  # e.g., cross_task_basketball-v3_bin-picking-v3
        traj_dir = traj_base / pair_stem if traj_base.exists() else None

        pair_results = analyze_pair(pair_data, traj_dir)
        results.extend(pair_results)

    return results


if __name__ == "__main__":
    main()
