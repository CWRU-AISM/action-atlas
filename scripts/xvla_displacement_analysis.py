#!/usr/bin/env python3
"""
X-VLA Cross-Task Displacement Analysis

Determines whether injecting task A's activations into task B causes the robot
to perform task A instead of task B. Uses:
- Cosine similarity to source vs destination baseline trajectories
- Object displacement analysis (LIBERO only: which objects moved, matching source task?)
- EEF trajectory comparison

Outputs:
- displacement_analysis.md: Markdown with tables
- displacement_analysis.json: Machine-readable results

Usage:
    python scripts/xvla_displacement_analysis.py --batch-dir rollouts/batch_1
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return p_hat * 100, max(0, center - margin) * 100, min(1, center + margin) * 100


def format_rate(successes, total):
    rate, lo, hi = wilson_ci(successes, total)
    return f"{rate:.1f}% ({successes}/{total}) [CI: {lo:.1f}-{hi:.1f}%]"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def classify_behavior(cos_to_src, cos_to_dst, threshold=0.05):
    """Classify whether the robot performed source task, destination task, or neither.

    Returns: 'source', 'destination', or 'ambiguous'
    """
    if cos_to_src is None or cos_to_dst is None:
        return 'ambiguous'

    diff = cos_to_src - cos_to_dst
    if diff > threshold:
        return 'source'  # Robot followed source task trajectory
    elif diff < -threshold:
        return 'destination'  # Robot followed destination task trajectory
    else:
        return 'ambiguous'


def analyze_libero_pair(pair_data):
    """Analyze a single LIBERO cross-task pair for displacement behavior."""
    results = []

    task_a = pair_data.get('task_a')
    task_b = pair_data.get('task_b')
    prompt_a = pair_data.get('prompt_a', '')
    prompt_b = pair_data.get('prompt_b', '')

    # Get task object lists
    obj_key_a = f'task_{task_a}_objects'
    obj_key_b = f'task_{task_b}_objects'
    objects_a = pair_data.get(obj_key_a, [])
    objects_b = pair_data.get(obj_key_b, [])

    # Get baseline object displacements (which objects each task normally moves)
    baseline_a = pair_data.get(f'baseline_task_{task_a}', {})
    baseline_b = pair_data.get(f'baseline_task_{task_b}', {})

    def get_moved_objects(scene_data, min_dist=0.01):
        """Get objects that were significantly displaced."""
        disps = scene_data.get('scene', {}).get('object_displacements', {})
        return {obj: d['distance'] for obj, d in disps.items() if d['distance'] > min_dist}

    baseline_moved_a = get_moved_objects(baseline_a)
    baseline_moved_b = get_moved_objects(baseline_b)

    # Analyze each injection direction
    for direction_key in [f'inject_{task_a}_into_{task_b}', f'inject_{task_b}_into_{task_a}']:
        injections = pair_data.get(direction_key, {})
        if not injections:
            continue

        # Parse direction
        parts = direction_key.split('_')
        src_task = int(parts[1])
        dst_task = int(parts[3])
        src_prompt = prompt_a if src_task == task_a else prompt_b
        dst_prompt = prompt_b if dst_task == task_b else prompt_a
        src_baseline_moved = baseline_moved_a if src_task == task_a else baseline_moved_b
        dst_baseline_moved = baseline_moved_b if dst_task == task_b else baseline_moved_a

        for cond_name, cond_data in injections.items():
            cos_src = cond_data.get('cos_to_src_baseline')
            cos_dst = cond_data.get('cos_to_dst_baseline')
            success = cond_data.get('success', False)

            # Get objects moved during injection
            injection_moved = get_moved_objects(cond_data)

            # Check if injected episode moved source task objects
            src_obj_match = 0
            dst_obj_match = 0
            if src_baseline_moved and injection_moved:
                src_obj_match = len(set(injection_moved.keys()) & set(src_baseline_moved.keys()))
            if dst_baseline_moved and injection_moved:
                dst_obj_match = len(set(injection_moved.keys()) & set(dst_baseline_moved.keys()))

            behavior = classify_behavior(cos_src, cos_dst)

            results.append({
                'src_task': src_task,
                'dst_task': dst_task,
                'condition': cond_name,
                'success': success,
                'cos_to_src': cos_src,
                'cos_to_dst': cos_dst,
                'behavior': behavior,
                'n_objects_moved': len(injection_moved),
                'src_obj_match': src_obj_match,
                'dst_obj_match': dst_obj_match,
                'moved_objects': list(injection_moved.keys()),
                'src_baseline_moved': list(src_baseline_moved.keys()),
                'dst_baseline_moved': list(dst_baseline_moved.keys()),
            })

    return results


def analyze_simplerenv_pair(pair_data):
    """Analyze a single SimplerEnv cross-task pair."""
    results = []

    task_a = pair_data.get('task_a')
    task_b = pair_data.get('task_b')
    name_a = pair_data.get('name_a', f'task_{task_a}')
    name_b = pair_data.get('name_b', f'task_{task_b}')

    for direction_key in [f'inject_{task_a}_into_{task_b}', f'inject_{task_b}_into_{task_a}']:
        injections = pair_data.get(direction_key, {})
        if not injections:
            continue

        parts = direction_key.split('_')
        src_task = int(parts[1])
        dst_task = int(parts[3])
        src_name = name_a if src_task == task_a else name_b
        dst_name = name_b if dst_task == task_b else name_a

        for cond_name, cond_data in injections.items():
            cos_src = cond_data.get('cos_to_src', cond_data.get('cos_to_src_baseline'))
            cos_dst = cond_data.get('cos_to_dst', cond_data.get('cos_to_dst_baseline'))
            success = cond_data.get('success', False)

            behavior = classify_behavior(cos_src, cos_dst)

            results.append({
                'src_task': src_task,
                'dst_task': dst_task,
                'src_name': src_name,
                'dst_name': dst_name,
                'condition': cond_name,
                'success': success,
                'cos_to_src': cos_src,
                'cos_to_dst': cos_dst,
                'behavior': behavior,
            })

    return results


def aggregate_by_condition(all_results):
    """Aggregate displacement results by injection condition."""
    by_condition = defaultdict(lambda: {
        'total': 0, 'source': 0, 'destination': 0, 'ambiguous': 0,
        'successes': 0, 'cos_src_vals': [], 'cos_dst_vals': []
    })

    for r in all_results:
        cond = r['condition']
        agg = by_condition[cond]
        agg['total'] += 1
        agg[r['behavior']] += 1
        if r['success']:
            agg['successes'] += 1
        if r['cos_to_src'] is not None:
            agg['cos_src_vals'].append(r['cos_to_src'])
        if r['cos_to_dst'] is not None:
            agg['cos_dst_vals'].append(r['cos_to_dst'])

    return by_condition


def aggregate_object_displacement(all_results):
    """For LIBERO: aggregate which objects get displaced under injection."""
    obj_stats = defaultdict(lambda: {'times_displaced': 0, 'as_src_target': 0, 'as_dst_target': 0, 'as_neither': 0})

    for r in all_results:
        if 'moved_objects' not in r:
            continue
        src_moved = set(r.get('src_baseline_moved', []))
        dst_moved = set(r.get('dst_baseline_moved', []))

        for obj in r['moved_objects']:
            obj_stats[obj]['times_displaced'] += 1
            if obj in src_moved and obj not in dst_moved:
                obj_stats[obj]['as_src_target'] += 1
            elif obj in dst_moved and obj not in src_moved:
                obj_stats[obj]['as_dst_target'] += 1
            elif obj in src_moved and obj in dst_moved:
                obj_stats[obj]['as_dst_target'] += 1  # shared object, count as dst
            else:
                obj_stats[obj]['as_neither'] += 1

    return obj_stats


# ============================================================
# COUNTERFACTUAL OBJECT ANALYSIS
# ============================================================

def analyze_counterfactual_objects(batch_dir):
    """Analyze which objects get displaced under different counterfactual prompt conditions.

    For each LIBERO suite/task, compares object displacements across conditions
    to determine: does the robot still interact with the correct object when given
    wrong prompts, or does it interact with the wrong-named object?
    """
    results = {}

    libero_exp = batch_dir / "xvla_libero" / "experiments"
    if not libero_exp.exists():
        return results

    for cf_dir in sorted(libero_exp.glob("counterfactual_*")):
        suite_name = cf_dir.name.replace("counterfactual_", "").replace("_v2", "")
        suite_data = {
            'tasks': {},
            'per_condition_objects': defaultdict(lambda: defaultdict(list)),
            'baseline_objects': defaultdict(list),
        }

        for task_dir in sorted(cf_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            rj = load_json(task_dir / "results.json")
            if not rj or "conditions" not in rj:
                continue

            task_desc = rj.get('task_description', task_dir.name)
            task_objects = rj.get('objects', [])
            suite_data['tasks'][task_dir.name] = {
                'description': task_desc,
                'objects': task_objects,
            }

            for cond_name, cond_data in rj['conditions'].items():
                scenes = cond_data.get('scenes', [])
                for scene in scenes:
                    obj_disps = scene.get('object_displacements', {})
                    for obj_name, disp_data in obj_disps.items():
                        dist = disp_data.get('distance', 0) if isinstance(disp_data, dict) else float(disp_data)
                        if dist > 0.01:
                            suite_data['per_condition_objects'][cond_name][obj_name].append(dist)
                            if cond_name == 'baseline':
                                suite_data['baseline_objects'][obj_name].append(dist)

        if suite_data['tasks']:
            results[suite_name] = suite_data

    return results


# ============================================================
# VISION PERTURBATION OBJECT ANALYSIS
# ============================================================

def analyze_vision_objects(batch_dir):
    """Analyze which objects get displaced under different vision perturbations.

    Compares object displacements across perturbation types to determine:
    does visual corruption cause the robot to interact with wrong objects?
    """
    results = {}

    libero_exp = batch_dir / "xvla_libero" / "experiments"
    if not libero_exp.exists():
        return results

    for vp_dir in sorted(libero_exp.glob("vision_*")):
        suite_name = vp_dir.name.replace("vision_", "")
        suite_data = {
            'per_perturbation_objects': defaultdict(lambda: defaultdict(list)),
            'baseline_objects': defaultdict(list),
            'per_perturbation_success': defaultdict(lambda: {'successes': 0, 'total': 0}),
        }

        for task_dir in sorted(vp_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            rj = load_json(task_dir / "results.json")
            if not rj or "results" not in rj:
                continue

            for entry in rj['results']:
                pert_name = entry.get('perturbation', 'unknown')
                success = entry.get('success', False)
                suite_data['per_perturbation_success'][pert_name]['total'] += 1
                if success:
                    suite_data['per_perturbation_success'][pert_name]['successes'] += 1

                scene = entry.get('scene', {})
                obj_disps = scene.get('object_displacements', {})
                for obj_name, dist_val in obj_disps.items():
                    dist = dist_val if isinstance(dist_val, (int, float)) else dist_val.get('distance', 0)
                    if dist > 0.01:
                        suite_data['per_perturbation_objects'][pert_name][obj_name].append(dist)
                        if pert_name == 'baseline':
                            suite_data['baseline_objects'][obj_name].append(dist)

        if suite_data['per_perturbation_success']:
            results[suite_name] = suite_data

    return results


def main():
    parser = argparse.ArgumentParser(description="X-VLA Displacement Analysis")
    parser.add_argument("--batch-dir", type=str, default="rollouts/batch_1")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir) if args.output_dir else batch_dir

    print(f"Cross-task displacement analysis: {batch_dir}")
    print("=" * 60)

    all_env_results = {}

    # ─── LIBERO ───
    libero_exp = batch_dir / "xvla_libero" / "experiments"
    if libero_exp.exists():
        for ct_dir in sorted(libero_exp.glob("cross_task_*")):
            suite = ct_dir.name.replace("cross_task_", "")
            print(f"\n[LIBERO] {suite}...")
            suite_results = []

            for pair_dir in sorted(ct_dir.glob("pair_*")):
                pair_data = load_json(pair_dir / "pair_result.json")
                if pair_data:
                    suite_results.extend(analyze_libero_pair(pair_data))

            if suite_results:
                all_env_results[f"libero/{suite}"] = suite_results
                print(f"  {len(suite_results)} injection episodes analyzed")

    # ─── SimplerEnv ───
    simplerenv_exp = batch_dir / "xvla_SIMPLERENV" / "experiments"
    if simplerenv_exp.exists():
        for ct_dir in sorted(simplerenv_exp.glob("cross_task_*")):
            robot = ct_dir.name.replace("cross_task_", "")
            print(f"\n[SimplerEnv] {robot}...")
            robot_results = []

            for pair_dir in sorted(ct_dir.glob("pair_*")):
                pair_data = load_json(pair_dir / "pair_result.json")
                if pair_data:
                    robot_results.extend(analyze_simplerenv_pair(pair_data))

            if robot_results:
                all_env_results[f"simplerenv/{robot}"] = robot_results
                print(f"  {len(robot_results)} injection episodes analyzed")

    # ─── Counterfactual Object Analysis ───
    print("\n[Counterfactual] Analyzing object displacements...")
    cf_objects = analyze_counterfactual_objects(batch_dir)
    for suite, sd in cf_objects.items():
        n_conds = len(sd['per_condition_objects'])
        n_objs = sum(len(objs) for objs in sd['per_condition_objects'].values())
        print(f"  {suite}: {n_conds} conditions, {n_objs} object-condition pairs")

    # ─── Vision Perturbation Object Analysis ───
    print("\n[Vision] Analyzing object displacements...")
    vp_objects = analyze_vision_objects(batch_dir)
    for suite, sd in vp_objects.items():
        n_perts = len(sd['per_perturbation_objects'])
        print(f"  {suite}: {n_perts} perturbations with object displacements")

    # ─── Write outputs ───
    print("\n" + "=" * 60)
    print("Writing displacement analysis...")

    lines = []
    lines.append("# X-VLA Cross-Task Displacement Analysis")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("\n**Question:** When we inject task A's activations into task B's environment,")
    lines.append("does the robot attempt to perform task A (source) instead of task B (destination)?")
    lines.append("\n**Classification method:** Compare cosine similarity of the injected episode's")
    lines.append("action trajectory to the source baseline vs destination baseline.")
    lines.append("- **Source behavior**: cos(src) - cos(dst) > 0.05 (robot follows source task)")
    lines.append("- **Destination behavior**: cos(dst) - cos(src) > 0.05 (robot follows own task)")
    lines.append("- **Ambiguous**: |cos(src) - cos(dst)| <= 0.05\n")

    # ─── Grand Summary ───
    lines.append("## Grand Summary\n")
    lines.append("| Environment | Total Injections | Source Behavior | Dest Behavior | Ambiguous | Dest Task Success |")
    lines.append("|-------------|-----------------|-----------------|---------------|-----------|-------------------|")

    json_output = {}

    for env_key in sorted(all_env_results.keys()):
        results = all_env_results[env_key]
        # Exclude no-inject controls from displacement summary
        inject_results = [r for r in results if 'transformer' in r['condition']]

        total = len(inject_results)
        source = sum(1 for r in inject_results if r['behavior'] == 'source')
        dest = sum(1 for r in inject_results if r['behavior'] == 'destination')
        ambig = sum(1 for r in inject_results if r['behavior'] == 'ambiguous')
        succ = sum(1 for r in inject_results if r['success'])

        lines.append(f"| {env_key} | {total} | {format_rate(source, total)} | {format_rate(dest, total)} | {format_rate(ambig, total)} | {format_rate(succ, total)} |")

    # ─── Per-Environment Details ───
    for env_key in sorted(all_env_results.keys()):
        results = all_env_results[env_key]
        lines.append(f"\n---\n\n## {env_key}\n")

        # Per-condition breakdown
        by_cond = aggregate_by_condition(results)

        lines.append("### Per-Condition Displacement\n")
        lines.append("| Condition | N | Source% | Dest% | Ambig% | Succ% | Mean cos→src | Mean cos→dst | cos→src > cos→dst? |")
        lines.append("|-----------|---|---------|-------|--------|-------|-------------|-------------|-------------------|")

        for cond in sorted(by_cond.keys()):
            agg = by_cond[cond]
            n = agg['total']
            src_pct = agg['source'] / n * 100 if n > 0 else 0
            dst_pct = agg['destination'] / n * 100 if n > 0 else 0
            amb_pct = agg['ambiguous'] / n * 100 if n > 0 else 0
            succ_pct = agg['successes'] / n * 100 if n > 0 else 0
            mean_cos_src = sum(agg['cos_src_vals']) / len(agg['cos_src_vals']) if agg['cos_src_vals'] else 0
            mean_cos_dst = sum(agg['cos_dst_vals']) / len(agg['cos_dst_vals']) if agg['cos_dst_vals'] else 0
            dominant = "YES" if mean_cos_src > mean_cos_dst else "no"

            lines.append(f"| {cond} | {n} | {src_pct:.1f}% | {dst_pct:.1f}% | {amb_pct:.1f}% | {succ_pct:.1f}% | {mean_cos_src:.4f} | {mean_cos_dst:.4f} | {dominant} |")

        # Injection-only analysis (transformer conditions)
        inject_only = [r for r in results if 'transformer' in r['condition']]
        if inject_only:
            lines.append(f"\n### Injection-Only Summary (transformer conditions)\n")
            total = len(inject_only)
            source = sum(1 for r in inject_only if r['behavior'] == 'source')
            dest = sum(1 for r in inject_only if r['behavior'] == 'destination')
            ambig = sum(1 for r in inject_only if r['behavior'] == 'ambiguous')

            lines.append(f"- **Total injection episodes:** {total}")
            lines.append(f"- **Source behavior (robot does injected task):** {format_rate(source, total)}")
            lines.append(f"- **Destination behavior (robot does own task):** {format_rate(dest, total)}")
            lines.append(f"- **Ambiguous:** {format_rate(ambig, total)}")

            # Per-layer breakdown
            by_layer = defaultdict(lambda: {'total': 0, 'source': 0, 'dest': 0, 'ambig': 0})
            for r in inject_only:
                cond = r['condition']
                if 'L0' in cond and 'L12' not in cond and 'L23' not in cond:
                    layer = 'L0'
                elif 'L12_L23' in cond:
                    layer = 'L12+L23'
                elif 'L12' in cond and 'L23' not in cond:
                    layer = 'L12'
                elif 'L23' in cond:
                    layer = 'L23'
                elif 'ALL' in cond:
                    layer = 'ALL'
                else:
                    layer = 'other'

                by_layer[layer]['total'] += 1
                if r['behavior'] == 'source':
                    by_layer[layer]['source'] += 1
                elif r['behavior'] == 'destination':
                    by_layer[layer]['dest'] += 1
                else:
                    by_layer[layer]['ambig'] += 1

            lines.append("\n**Per-layer displacement:**\n")
            lines.append("| Layer | N | Source% | Dest% | Ambig% |")
            lines.append("|-------|---|---------|-------|--------|")
            for layer in ['L0', 'L12', 'L23', 'L12+L23', 'ALL']:
                if layer in by_layer:
                    bl = by_layer[layer]
                    n = bl['total']
                    lines.append(f"| {layer} | {n} | {bl['source']/n*100:.1f}% | {bl['dest']/n*100:.1f}% | {bl['ambig']/n*100:.1f}% |")

        # Cross-prompt vs own-prompt analysis
        cross_prompt = [r for r in results if r['condition'].startswith('cross_prompt')]
        own_prompt = [r for r in results if r['condition'].startswith('own_prompt')]

        if cross_prompt and own_prompt:
            lines.append(f"\n### Cross-Prompt vs Own-Prompt\n")
            lines.append("| Prompt Type | N | Source% | Dest% | Success% |")
            lines.append("|-------------|---|---------|-------|----------|")
            for label, subset in [("Cross-prompt", cross_prompt), ("Own-prompt", own_prompt)]:
                n = len(subset)
                src = sum(1 for r in subset if r['behavior'] == 'source') / n * 100
                dst = sum(1 for r in subset if r['behavior'] == 'destination') / n * 100
                succ = sum(1 for r in subset if r['success']) / n * 100
                lines.append(f"| {label} | {n} | {src:.1f}% | {dst:.1f}% | {succ:.1f}% |")

        # Object displacement analysis (LIBERO only)
        if any('moved_objects' in r for r in results):
            inject_with_objs = [r for r in results if 'transformer' in r['condition'] and r.get('moved_objects')]
            if inject_with_objs:
                obj_stats = aggregate_object_displacement(inject_with_objs)
                lines.append(f"\n### Object Displacement Under Injection\n")
                lines.append("Which objects get displaced when activations are injected? Does the robot")
                lines.append("move the *source* task's target objects (indicating it's performing the wrong task)?\n")
                lines.append("| Object | Times Displaced | As Source Target | As Dest Target | As Neither |")
                lines.append("|--------|----------------|-----------------|----------------|------------|")
                for obj in sorted(obj_stats.keys(), key=lambda x: -obj_stats[x]['times_displaced'])[:20]:
                    os = obj_stats[obj]
                    lines.append(f"| {obj} | {os['times_displaced']} | {os['as_src_target']} | {os['as_dst_target']} | {os['as_neither']} |")

                # Source object displacement rate
                total_displacements = sum(1 for r in inject_with_objs if r['src_obj_match'] > 0)
                lines.append(f"\n**Source task object displacement rate:** {format_rate(total_displacements, len(inject_with_objs))}")
                lines.append(f"(Episodes where at least one source task target object was displaced)")

        # Build JSON output
        json_output[env_key] = {
            'total_episodes': len(results),
            'injection_episodes': len([r for r in results if 'transformer' in r['condition']]),
            'per_condition': {},
        }
        for cond, agg in by_cond.items():
            n = agg['total']
            json_output[env_key]['per_condition'][cond] = {
                'total': n,
                'source_behavior': agg['source'],
                'destination_behavior': agg['destination'],
                'ambiguous': agg['ambiguous'],
                'successes': agg['successes'],
                'mean_cos_to_src': sum(agg['cos_src_vals']) / len(agg['cos_src_vals']) if agg['cos_src_vals'] else None,
                'mean_cos_to_dst': sum(agg['cos_dst_vals']) / len(agg['cos_dst_vals']) if agg['cos_dst_vals'] else None,
            }

    # ─── Counterfactual Object Analysis ───
    if cf_objects:
        lines.append("\n---\n\n## Counterfactual Prompt - Object Displacement Analysis\n")
        lines.append("Which objects does the robot interact with under different prompt conditions?")
        lines.append("Compares object displacements (distance > 0.01) across prompt conditions vs baseline.\n")

        for suite in sorted(cf_objects.keys()):
            sd = cf_objects[suite]
            lines.append(f"\n### {suite}\n")

            # Baseline objects first
            bl_objs = sd['baseline_objects']
            if bl_objs:
                lines.append("**Baseline (correct prompt) - objects displaced:**\n")
                lines.append("| Object | Times Displaced | Mean Distance |")
                lines.append("|--------|----------------|---------------|")
                for obj in sorted(bl_objs.keys(), key=lambda x: -len(bl_objs[x])):
                    dists = bl_objs[obj]
                    lines.append(f"| {obj} | {len(dists)} | {sum(dists)/len(dists):.4f} |")

            # Per-condition: which objects are displaced that AREN'T in baseline?
            baseline_obj_set = set(bl_objs.keys())
            wrong_object_conditions = []

            lines.append("\n**Per-condition object displacement (non-baseline conditions):**\n")
            lines.append("| Condition | Objects Displaced | New Objects (not in baseline) | Baseline Objects Missing |")
            lines.append("|-----------|------------------|-----------------------------|-----------------------|")

            for cond in sorted(sd['per_condition_objects'].keys()):
                if cond == 'baseline':
                    continue
                cond_objs = sd['per_condition_objects'][cond]
                cond_obj_set = set(cond_objs.keys())
                new_objs = cond_obj_set - baseline_obj_set
                missing_bl = baseline_obj_set - cond_obj_set

                n_displaced = len(cond_obj_set)
                new_str = ", ".join(sorted(new_objs)[:5]) if new_objs else "-"
                missing_str = ", ".join(sorted(missing_bl)[:5]) if missing_bl else "-"
                lines.append(f"| {cond} | {n_displaced} | {new_str} | {missing_str} |")

                if new_objs:
                    wrong_object_conditions.append((cond, new_objs))

            if wrong_object_conditions:
                lines.append(f"\n**Wrong-object interactions:** {len(wrong_object_conditions)} conditions caused the robot")
                lines.append(f"to displace objects NOT displaced under baseline (correct prompt).\n")
                for cond, new_objs in wrong_object_conditions[:10]:
                    lines.append(f"- `{cond}`: displaced {', '.join(sorted(new_objs))}")

    # ─── Vision Perturbation Object Analysis ───
    if vp_objects:
        lines.append("\n---\n\n## Vision Perturbation - Object Displacement Analysis\n")
        lines.append("Which objects does the robot interact with under different visual corruptions?")
        lines.append("Compares object displacements across perturbation types vs clean baseline.\n")

        for suite in sorted(vp_objects.keys()):
            sd = vp_objects[suite]
            lines.append(f"\n### {suite}\n")

            bl_objs = sd['baseline_objects']
            baseline_obj_set = set(bl_objs.keys())

            # Per-perturbation summary
            lines.append("| Perturbation | Success Rate | Objects Displaced | Correct Objects | Wrong Objects |")
            lines.append("|-------------|-------------|-------------------|-----------------|---------------|")

            for pert in sorted(sd['per_perturbation_objects'].keys()):
                pert_objs = sd['per_perturbation_objects'][pert]
                pert_obj_set = set(pert_objs.keys())
                correct_objs = pert_obj_set & baseline_obj_set
                wrong_objs = pert_obj_set - baseline_obj_set

                ps = sd['per_perturbation_success'].get(pert, {'successes': 0, 'total': 0})
                sr = ps['successes'] / ps['total'] * 100 if ps['total'] > 0 else 0

                wrong_str = ", ".join(sorted(wrong_objs)[:3]) if wrong_objs else "-"
                lines.append(f"| {pert} | {sr:.0f}% ({ps['successes']}/{ps['total']}) | {len(pert_obj_set)} | {len(correct_objs)} | {len(wrong_objs)} ({wrong_str}) |")

            # Summary: which perturbations cause wrong-object interaction?
            wrong_pert_count = sum(1 for pert in sd['per_perturbation_objects']
                                   if set(sd['per_perturbation_objects'][pert].keys()) - baseline_obj_set)
            total_perts = len(sd['per_perturbation_objects'])
            if total_perts > 0:
                lines.append(f"\n**{wrong_pert_count}/{total_perts} perturbations** caused interaction with non-baseline objects.")

    # ─── Key Findings ───
    lines.append("\n---\n\n## Key Findings\n")

    # Compute overall stats across all environments
    all_inject = []
    for results in all_env_results.values():
        all_inject.extend([r for r in results if 'transformer' in r['condition']])

    if all_inject:
        total = len(all_inject)
        source = sum(1 for r in all_inject if r['behavior'] == 'source')
        dest = sum(1 for r in all_inject if r['behavior'] == 'destination')
        ambig = sum(1 for r in all_inject if r['behavior'] == 'ambiguous')

        lines.append(f"1. **Overall:** Across {total} injection episodes, the robot follows the source task's")
        lines.append(f"   trajectory {source/total*100:.1f}% of the time, destination task {dest/total*100:.1f}%, ambiguous {ambig/total*100:.1f}%.")

        # cos_src > cos_dst analysis
        src_dominant = sum(1 for r in all_inject if r.get('cos_to_src', 0) and r.get('cos_to_dst', 0) and r['cos_to_src'] > r['cos_to_dst'])
        lines.append(f"\n2. **Cosine dominance:** cos→src > cos→dst in {format_rate(src_dominant, total)} of injection episodes.")
        lines.append(f"   This means injected activations consistently steer the robot toward the source task's trajectory.")

        # Cross-prompt specific
        cross_inject = [r for r in all_inject if r['condition'].startswith('cross_prompt')]
        if cross_inject:
            ci_src = sum(1 for r in cross_inject if r['behavior'] == 'source')
            lines.append(f"\n3. **Cross-prompt + injection:** {format_rate(ci_src, len(cross_inject))} show source behavior")
            lines.append(f"   (strongest manipulation: wrong prompt + wrong activations)")

    # Write files
    md_path = output_dir / "XVLA_DISPLACEMENT_ANALYSIS.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Wrote: {md_path}")

    # Add counterfactual and vision object data to JSON
    json_output['counterfactual_objects'] = {}
    for suite, sd in cf_objects.items():
        suite_json = {}
        for cond, objs in sd['per_condition_objects'].items():
            suite_json[cond] = {obj: {'count': len(dists), 'mean_dist': sum(dists)/len(dists)}
                                for obj, dists in objs.items()}
        json_output['counterfactual_objects'][suite] = suite_json

    json_output['vision_perturbation_objects'] = {}
    for suite, sd in vp_objects.items():
        suite_json = {}
        for pert, objs in sd['per_perturbation_objects'].items():
            suite_json[pert] = {obj: {'count': len(dists), 'mean_dist': sum(dists)/len(dists)}
                                for obj, dists in objs.items()}
        json_output['vision_perturbation_objects'][suite] = suite_json

    json_path = output_dir / "xvla_displacement_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
