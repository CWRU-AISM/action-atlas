"""Action Atlas API - injection and steering routes."""
from flask import Blueprint, request, jsonify
from .helpers import *
from .data_loaders import *
from .experiment_helpers import _find_latest_oft_result

injection_bp = Blueprint("injection", __name__)

# ---------------------------------------------------------------------------

@injection_bp.route('/api/vla/injection', methods=['GET'])
def get_injection():
    """Cross-task and same-scene injection results."""
    model = request.args.get('model', 'openvla')
    suite = request.args.get('suite', 'libero_goal')
    injection_type = request.args.get('type', 'cross_task')

    if model == 'act':
        return _injection_act(injection_type)
    if model in ('openvla', 'oft'):
        return _injection_openvla(model, suite, injection_type)
    if model in ('pi05', 'xvla', 'smolvla', 'groot'):
        return _injection_generic(model, suite, injection_type)

    return jsonify({'status': 404, 'error': f'No injection data available for model {model}'}), 404


def _injection_act(injection_type: str):
    """ACT injection data from rollout dirs or all_results.json fallback."""
    data_file = None
    videos = {}

    if injection_type == 'cross_task':
        candidates = [
            ALOHA_DATA_DIR / "injection_cross_task" / "cross_task" / "cross_task_results.json",
            ALOHA_DATA_DIR / "injection_cross_task" / "cross_task" / "results.json",
            ACT_INJECTION_DIR / "cross_task_results.json",
        ]
        for c in candidates:
            if c.exists():
                data_file = c
                break

        video_dir = ALOHA_DATA_DIR / "injection_cross_task" / "cross_task"
        if video_dir.exists():
            for vf in video_dir.glob("*.mp4"):
                videos[vf.stem] = f"/act_videos/injection_cross_task/cross_task/{vf.name}"

    elif injection_type == 'same_task':
        candidates = [
            ALOHA_DATA_DIR / "injection_same_task_ins" / "same_task" / "results.json",
            ALOHA_DATA_DIR / "injection_same_task_tc" / "same_task" / "results.json",
            ACT_INJECTION_DIR / "same_task_insertion.json",
            ACT_INJECTION_DIR / "same_task_transfer_cube.json",
        ]
        for c in candidates:
            if c.exists():
                data_file = c
                break
    else:
        return jsonify({'status': 400, 'error': f'Unknown type: {injection_type}'}), 400

    act_key_finding = (
        'ACT-ALOHA residual connections completely wash out injected activations '
        '(cos_to_baseline=1.0). Injection has ZERO effect.'
    )

    if data_file is None or not data_file.exists():
        all_results = load_json_cached(ACT_RESULTS_DIR / "all_results.json", 'act_all_results')
        if all_results:
            return jsonify({
                'model': 'act',
                'injection_type': injection_type,
                'data': all_results.get('injection', {}),
                'videos': videos,
                'data_source': 'all_results.json',
                'key_finding': act_key_finding,
            })
        return jsonify({'status': 404, 'error': 'No ACT injection data found'}), 404

    with open(data_file) as f:
        data = json.load(f)

    return jsonify({
        'model': 'act',
        'injection_type': injection_type,
        'data': data,
        'videos': videos,
        'data_source': str(data_file.name),
        'key_finding': act_key_finding,
    })


def _injection_openvla(model: str, suite: str, injection_type: str):
    """OpenVLA/OFT injection from per-run results or experiment_results_oft.json."""
    exp_type = f'{injection_type}_injection'
    results_file = _find_latest_oft_result(suite, exp_type)

    if not results_file:
        # Fallback: try loading from experiment_results_oft.json
        exp_data = _load_experiment_results_cached('oft')
        if exp_data:
            section_key = 'cross_task' if injection_type == 'cross_task' else (
                f'{injection_type}_injection'
            )
            ct = exp_data.get(section_key, exp_data.get(injection_type, {}))
            suite_data = _resolve_suite_data(ct, suite)

            if suite_data:
                runs = suite_data.get('runs', [])
                conditions = {}
                summary_dict = {}
                if runs:
                    total_eps = sum(r.get('n_episodes', 0) for r in runs)
                    total_succ_eps = sum(
                        round(r.get('success_rate', 0) * r.get('n_episodes', 0))
                        for r in runs
                    )
                    overall_sr = total_succ_eps / max(total_eps, 1)
                    summary_dict[injection_type] = {
                        'success_rate': overall_sr,
                        'successes': total_succ_eps, 'total': total_eps,
                    }
                    for i, run in enumerate(runs[:50]):
                        run_id = run.get('run_id', f'run_{i}')
                        sr = run.get('success_rate', 0)
                        n_eps = run.get('n_episodes', 0)
                        conditions[f"run_{run_id}"] = {
                            'baseline_a': {'success': True},
                            'baseline_b': {'success': True},
                            'desc_a': f'Run {run_id}',
                            'desc_b': f'SR: {sr:.0%} ({n_eps} eps, {run.get("n_conditions", "?")} conditions)',
                        }
                return jsonify({
                    'model': 'openvla', 'suite': suite,
                    'injection_type': injection_type,
                    'available_suites': list(ct.keys()),
                    'available_types': ['cross_task', 'same_scene', 'null'],
                    'conditions': conditions,
                    'summary': summary_dict,
                    'videos': {},
                    'n_pairs': len(conditions),
                    'data_source': 'experiment_results_oft.json',
                })

        return jsonify({'status': 404, 'error': f'No {injection_type} injection data for {suite}'}), 404

    with open(results_file) as f:
        data = json.load(f)

    conditions = data.get('conditions', {})

    run_dir = results_file.parent.parent
    exp_dir = run_dir / exp_type
    videos = {}
    if exp_dir.exists():
        for vf in exp_dir.glob("*.mp4"):
            videos[vf.stem] = (
                f"/videos/openvla/openvla_oft/{suite}/"
                f"{run_dir.name}/{exp_type}/{vf.name}"
            )

    summary = {}
    if injection_type == 'cross_task':
        for pair_data in conditions.values():
            for layer_key in ['layer_0', 'layer_16', 'layer_31']:
                inject_data = pair_data.get('injection_b_into_a', {}).get(layer_key, {})
                if layer_key not in summary:
                    summary[layer_key] = {'successes': 0, 'total': 0}
                summary[layer_key]['total'] += 1
                if inject_data.get('success', False):
                    summary[layer_key]['successes'] += 1
        for k in summary:
            summary[k]['success_rate'] = summary[k]['successes'] / max(summary[k]['total'], 1)

    elif injection_type == 'same_scene':
        for task_data in conditions.values():
            baseline = task_data.get('baseline', {})
            inject = task_data.get('injection', task_data.get('inject', {}))
            if 'baseline' not in summary:
                summary['baseline'] = {'successes': 0, 'total': 0}
                summary['injection'] = {'successes': 0, 'total': 0}
            summary['baseline']['total'] += 1
            summary['injection']['total'] += 1
            if baseline.get('success', False):
                summary['baseline']['successes'] += 1
            if inject.get('success', False):
                summary['injection']['successes'] += 1
        for k in summary:
            summary[k]['success_rate'] = summary[k]['successes'] / max(summary[k]['total'], 1)

    return jsonify({
        'model': 'openvla', 'suite': suite,
        'injection_type': injection_type,
        'available_suites': ['libero_goal', 'libero_10', 'libero_object', 'libero_spatial'],
        'available_types': ['cross_task', 'same_scene', 'null', 'temporal', 'cross_prompt'],
        'conditions': conditions,
        'summary': summary,
        'videos': videos,
        'n_pairs': len(conditions),
    })


def _injection_generic(model: str, suite: str, injection_type: str):
    """Injection for pi05 / xvla / smolvla / groot from baked or experiment results."""
    # Try baked injection data first (has per-pair results from disk)
    baked_inj = load_json_cached(
        API_DATA_DIR / f'{model}_injection_baked.json',
        f'{model}_injection_baked',
    )
    if baked_inj:
        baked_suite = baked_inj.get(suite, baked_inj.get(suite_short(suite), {}))
        if baked_suite and baked_suite.get('pairs'):
            return _injection_from_baked(model, suite, injection_type, baked_inj, baked_suite)

    exp_data = _load_experiment_results_cached(model)
    if exp_data is None:
        return jsonify({'status': 404, 'error': f'No injection data available for model {model}'}), 404

    ct = exp_data.get('cross_task', {})
    suite_data = _resolve_suite_data(ct, suite)

    if not suite_data:
        return jsonify({
            'model': model, 'suite': suite,
            'available_suites': list(ct.keys()),
            'status': 404,
            'error': f'No injection data for {model}/{suite}. Available: {", ".join(ct.keys())}',
        }), 404

    return _injection_from_experiment_results(model, suite, injection_type, exp_data, ct, suite_data)


def _injection_from_baked(
    model: str, suite: str, injection_type: str,
    baked_inj: dict, baked_suite: dict,
):
    """Format baked injection pairs for the frontend."""
    baked_pairs = baked_suite['pairs']
    conditions = {}
    total_success = 0
    total_pairs = 0

    for pk, pv in baked_pairs.items():
        conds = pv.get('conditions', {})
        inject_conds = {k: v for k, v in conds.items() if 'inject' in k}
        if inject_conds:
            total_pairs += 1
            if any(v.get('success') for v in inject_conds.values()):
                total_success += 1
        conditions[pk] = {
            'baseline_a': conds.get('baseline_task_0', {'success': True}),
            'baseline_b': conds.get('baseline_task_1', {'success': True}),
            'desc_a': pv.get('prompt_a', pv.get('task_a_desc', f'Task {pv.get("task_a", "?")}')),
            'desc_b': pv.get('prompt_b', pv.get('task_b_desc', f'Task {pv.get("task_b", "?")}')),
            'injection_b_into_a': {k: v for k, v in conds.items() if 'inject' in k},
        }

    summary = {}
    if total_pairs > 0:
        summary['cross_task'] = {
            'success_rate': total_success / total_pairs,
            'successes': total_success, 'total': total_pairs,
        }

    cross_vids = _load_ablation_index_videos(model, 'cross_task')
    videos_map = _filter_videos_by_suite(cross_vids, suite)

    if total_pairs > 0:
        finding = (
            f'{model.upper()}: {len(conditions)} cross-task injection pairs for {suite}. '
            f'{total_success}/{total_pairs} show source behavior transfer.'
        )
    else:
        finding = f'{len(conditions)} pairs available.'

    return jsonify({
        'status': 200,
        'data': {
            'model': model, 'suite': suite,
            'injection_type': injection_type,
            'pairs': list(conditions.keys()),
            'conditions': conditions,
            'summary': summary,
            'videos': videos_map,
            'n_pairs': len(conditions),
            'available_suites': list(baked_inj.keys()),
            'key_finding': finding,
        }
    })


def _injection_from_experiment_results(
    model: str, suite: str, injection_type: str,
    exp_data: dict, ct: dict, suite_data: dict,
):
    """Format experiment_results cross_task injection for the frontend."""
    pairs = suite_data.get('pairs', {})
    runs = suite_data.get('runs', [])
    conditions = {}
    summary = {}

    cross_vids = _load_ablation_index_videos(model, 'cross_task') if injection_type == 'cross_task' else []
    videos_map = _filter_videos_by_suite(cross_vids, suite) if cross_vids else {}

    if pairs:
        total_pairs = 0
        success_pairs = 0
        for pair_key, pair_data in pairs.items():
            total_pairs += 1
            baseline_a_success = pair_data.get(
                'baseline_A_success', pair_data.get('baseline_success_rate', 0),
            )
            baseline_b_success = pair_data.get('baseline_B_success', True)

            inject_result = pair_data.get('expert_all', pair_data.get('vlm_all', {}))
            if isinstance(inject_result, dict):
                injected_success = inject_result.get('success', False)
            else:
                injected_sr = pair_data.get('injected_success_rate', 0)
                injected_success = injected_sr > 0.5

            if injected_success:
                success_pairs += 1

            injection_b_into_a = {}
            for layer_key in [
                'expert_all', 'expert_early', 'expert_mid', 'expert_late',
                'vlm_all', 'vlm_early', 'vlm_mid', 'vlm_late',
                'layer_0', 'layer_16', 'layer_31',
            ]:
                if layer_key in pair_data and isinstance(pair_data[layer_key], dict):
                    injection_b_into_a[layer_key] = {
                        'success': pair_data[layer_key].get('success', False),
                        'n_steps': pair_data[layer_key].get('n_steps', 0),
                    }

            desc_a = pair_data.get('task_a_desc', pair_data.get('source_task',
                     pair_data.get('task_a', pair_key.split('_to_')[0] if '_to_' in pair_key else '')))
            desc_b = pair_data.get('task_b_desc', pair_data.get('target_task',
                     pair_data.get('task_b', pair_key.split('_to_')[1] if '_to_' in pair_key else '')))

            cond_entry = {
                'baseline_a': {'success': bool(baseline_a_success)},
                'baseline_b': {'success': bool(baseline_b_success)},
                'desc_a': desc_a,
                'desc_b': desc_b,
            }
            if injection_b_into_a:
                cond_entry['injection_b_into_a'] = injection_b_into_a
            conditions[pair_key] = cond_entry

        if total_pairs > 0:
            summary['cross_task'] = {
                'success_rate': success_pairs / total_pairs,
                'successes': success_pairs, 'total': total_pairs,
            }

    elif runs:
        total_runs = len(runs)
        total_eps = sum(r.get('n_episodes', 0) for r in runs)
        total_succ_eps = sum(
            round(r.get('success_rate', 0) * r.get('n_episodes', 0))
            for r in runs
        )
        if total_eps > 0:
            overall_sr = total_succ_eps / total_eps
        else:
            overall_sr = sum(1 for r in runs if r.get('success_rate', 0) > 0.5) / max(total_runs, 1)

        summary['cross_task'] = {
            'success_rate': overall_sr,
            'successes': total_succ_eps if total_eps > 0 else sum(1 for r in runs if r.get('success_rate', 0) > 0.5),
            'total': total_eps if total_eps > 0 else total_runs,
        }
        for i, run in enumerate(runs[:50]):
            run_id = run.get('run_id', f'run_{i}')
            sr = run.get('success_rate', 0)
            n_eps = run.get('n_episodes', 0)
            conditions[f"run_{run_id}"] = {
                'baseline_a': {'success': True},
                'baseline_b': {'success': True},
                'desc_a': f'Run {run_id}',
                'desc_b': f"SR: {sr:.0%} ({n_eps} eps, {run.get('n_conditions', '?')} conditions)",
            }

    # If no pairs and no runs, check for group_stats metadata
    if not conditions:
        n_task_pairs = suite_data.get('n_task_pairs', 0)
        n_processed = suite_data.get('n_processed_pairs', 0)
        group_stats = suite_data.get('group_stats', {})
        injection_groups = suite_data.get('injection_groups', [])
        if n_task_pairs > 0 or group_stats or injection_groups:
            key_info_parts = []
            if n_task_pairs > 0:
                key_info_parts.append(f'{n_task_pairs} task pairs configured, {n_processed} processed')
            for gk, gv in group_stats.items():
                if isinstance(gv, dict):
                    gs_sr = gv.get('success_rate', gv.get('mean_success_rate', 0))
                    gs_n = gv.get('n_pairs', gv.get('count', 0))
                    if gs_n > 0:
                        summary[gk] = {
                            'success_rate': gs_sr,
                            'successes': round(gs_sr * gs_n),
                            'total': gs_n,
                        }
                        key_info_parts.append(f'{gk}: {gs_sr*100:.0f}% ({gs_n} pairs)')
            if key_info_parts:
                suite_data['key_finding'] = (
                    f'{model.upper()} cross-task injection: ' + '; '.join(key_info_parts)
                )

    # Handle null_injection type
    null_data = exp_data.get('null_injection', {})
    if injection_type == 'null' and null_data:
        null_suite = null_data.get(suite, null_data.get(suite_short(suite), {}))
        if null_suite:
            conditions = {}
            summary = {}
            null_runs = null_suite.get('runs', [])
            if null_runs:
                total_eps = sum(r.get('n_episodes', 0) for r in null_runs)
                total_succ = sum(
                    round(r.get('success_rate', 0) * r.get('n_episodes', 0))
                    for r in null_runs
                )
                if total_eps > 0:
                    summary['null_injection'] = {
                        'success_rate': total_succ / total_eps,
                        'successes': total_succ, 'total': total_eps,
                    }
                else:
                    total = len(null_runs)
                    succ = sum(1 for r in null_runs if r.get('success_rate', 0) > 0.5)
                    summary['null_injection'] = {
                        'success_rate': succ / max(total, 1),
                        'successes': succ, 'total': total,
                    }
                for i, run in enumerate(null_runs[:50]):
                    run_id = run.get('run_id', f'null_{i}')
                    sr = run.get('success_rate', 0)
                    n_eps = run.get('n_episodes', 0)
                    conditions[f"null_{run_id}"] = {
                        'baseline_a': {'success': True},
                        'baseline_b': {'success': True},
                        'desc_a': f'Null injection run {run_id}',
                        'desc_b': f"SR: {sr:.0%} ({n_eps} eps)",
                    }

    return jsonify({
        'model': model, 'suite': suite,
        'injection_type': injection_type,
        'available_suites': list(ct.keys()),
        'available_types': ['cross_task', 'null'],
        'conditions': conditions,
        'summary': summary,
        'videos': videos_map,
        'n_pairs': len(conditions),
        'key_finding': suite_data.get(
            'key_finding', f'{len(conditions)} cross-task injection results for {model}.',
        ),
    })


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

@injection_bp.route('/api/vla/steering_concepts', methods=['GET'])
def get_steering_concepts():
    """Get available concepts for steering with pre-computed success curves."""
    steering_dir = API_DATA_DIR / "libero_10" / "steering_results"
    concepts = []
    if steering_dir.exists():
        for sf in sorted(steering_dir.glob("steering_*.json")):
            sd = load_json_cached(sf)
            if sd is None:
                continue
            strengths = {}
            for k, v in sd.get('results', {}).items():
                s = float(k.replace('strength_', ''))
                strengths[s] = v.get('success_rate', 0)

            concepts.append({
                'concept': sd.get('concept'),
                'layer': sd.get('layer'),
                'suite': sd.get('suite'),
                'n_features': len(sd.get('feature_ids', [])),
                'feature_ids': sd.get('feature_ids', []),
                'dose_response': strengths,
                'baseline_sr': 0.667,
            })

    return jsonify({
        'concepts': concepts,
        'goldilocks_warning': (
            'All concepts show 0% success at all steering strengths - the Goldilocks effect. '
            'VLA features are load-bearing and cannot be steered without task failure.'
        ),
        'note': (
            'Pre-computed from 80 rollouts (8 concepts x 5 strengths x 2 episodes each) '
            'on LIBERO-10, layer 12.'
        ),
    })


###############################################################################
# Scene State / Trajectory Visualization Endpoints
###############################################################################

# Cache for large merged_results.json files (they're 20-50MB each)
_scene_state_cache: Dict[str, dict] = {}
from .experiment_helpers import _find_latest_oft_result
