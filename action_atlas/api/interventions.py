# Action Atlas API - interventions routes
import json
import re
from pathlib import Path
from typing import Dict, Optional

from flask import Blueprint, request, jsonify

from .helpers import (
    ABLATION_INDEX_FILES, ACT_GRID_ABLATION_DIR, ACT_INJECTION_DIR,
    ACT_RESULTS_DIR, ALOHA_DATA_DIR, OFT_DATA_DIR,
    load_json_cached, normalize_suite, suite_short,
)
from .data_loaders import *
from .success_tracking import *
from .concepts import _ablation_summary_cache
from .experiment_helpers import _find_latest_oft_result

interventions_bp = Blueprint("interventions", __name__)

# Baked data lives alongside this module in action_atlas/api/data/
API_DATA_DIR = Path(__file__).parent / "data"

# Metaworld suite variants, tried in order when the requested suite is 'metaworld'
_METAWORLD_KEYS = [
    'metaworld_easy', 'metaworld_medium',
    'metaworld_hard', 'metaworld_very_hard',
]


def _natural_sort_key(k: str) -> list:
    # Sort key that handles embedded integers naturally (L2 before L10)
    parts = re.split(r'(\d+)', k)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _load_experiment_results_cached(model: str) -> Optional[dict]:
    # Load experiment_results_<model>.json with caching
    path = API_DATA_DIR / f"experiment_results_{model}.json"
    return load_json_cached(path, f"experiment_results_{model}")


def _resolve_suite_data(data: dict, suite: str) -> dict:
    # Look up suite data trying the original key, the alternate form, and metaworld variants
    suite_data = data.get(suite, {})
    if not suite_data:
        alt = normalize_suite(suite) if not suite.startswith('libero_') else suite_short(suite)
        suite_data = data.get(alt, {})
    if not suite_data and suite == 'metaworld':
        for mw_key in _METAWORLD_KEYS:
            suite_data = data.get(mw_key, {})
            if suite_data:
                break
    return suite_data


def _load_ablation_index_videos(model: str, experiment_type: str) -> list:
    # Load and cache videos of a given experiment_type from a model's ablation index
    cache_key = f'injection_videos_{model}_{experiment_type}'
    if cache_key in _ablation_summary_cache:
        return _ablation_summary_cache[cache_key]

    fname = ABLATION_INDEX_FILES.get(model)
    if not fname:
        return []

    idx_data = load_json_cached(API_DATA_DIR / fname, f"ablation_index_{model}")
    if idx_data is None:
        return []

    all_vids = idx_data.get('videos', []) if isinstance(idx_data, dict) else idx_data
    filtered = [v for v in all_vids if v.get('experiment_type') == experiment_type]
    _ablation_summary_cache[cache_key] = filtered
    return filtered


def _filter_videos_by_suite(videos: list, suite: str, limit: int = 50) -> dict:
    # Filter video list by suite and build a filename -> path map
    result = {}
    for v in videos:
        if v.get('suite', '') == suite or suite in v.get('path', ''):
            fn = v.get('filename', v.get('path', '').split('/')[-1])
            result[fn] = v.get('path', '')
            if len(result) >= limit:
                break
    return result
# Grid Ablation
@interventions_bp.route('/api/vla/grid_ablation', methods=['GET'])
def get_grid_ablation():
    # Interactive grid ablation data for ACT-ALOHA and OpenVLA-OFT
    model = request.args.get('model', 'act')
    task = request.args.get('task', 'AlohaInsertion-v0')
    suite = request.args.get('suite', 'libero_goal')

    if model == 'act':
        return _grid_ablation_act(task)
    if model == 'openvla':
        return _grid_ablation_openvla(suite)
    if model in ('xvla', 'smolvla', 'groot', 'pi05'):
        return _grid_ablation_generic(model, suite)

    return jsonify({'status': 404, 'error': f'No grid ablation data for model {model}'}), 404


def _grid_ablation_act(task: str):
    # ACT-ALOHA grid ablation from rollout dirs or JSON fallback
    base_dir = ALOHA_DATA_DIR / f"grid_ablation_{task}"
    grid = {}
    baseline = None
    noise_results = {}
    data_source = None

    if base_dir.exists():
        data_source = 'rollout_dirs'
        for cell_dir in sorted(base_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            summary = load_json_cached(cell_dir / "summary.json")
            if summary is None:
                continue

            name = cell_dir.name
            entry = {
                'success_rate': summary.get('success_rate', 0),
                'mean_reward': round(summary.get('mean_reward', 0), 1),
                'std_reward': round(summary.get('std_reward', 0), 1),
                'n_episodes': summary.get('n_episodes', 0),
                'has_videos': (cell_dir / "videos").exists(),
            }

            if name == 'baseline':
                baseline = entry
            elif name.startswith('noise_'):
                noise_results[name] = entry
            elif name.startswith('grid_'):
                parts = name.split('_')
                row, col = int(parts[1]), int(parts[2])
                grid[f"{row}_{col}"] = {
                    **entry, 'row': row, 'col': col,
                    'bbox': summary.get('perturbation_kwargs', {}).get('bbox', []),
                }

    if not grid and ACT_GRID_ABLATION_DIR.exists():
        data_source = 'json_results'
        for jf in sorted(ACT_GRID_ABLATION_DIR.glob(f"{task}*.json")):
            suffix = jf.stem.replace(f"{task}_", "")
            data = load_json_cached(jf)
            if data is None:
                continue

            entry = {
                'success_rate': data.get('success_rate', 0),
                'mean_reward': round(data.get('mean_reward', 0), 1),
                'std_reward': round(data.get('std_reward', 0), 1),
                'n_episodes': data.get('n_episodes', 0),
                'has_videos': False,
            }

            if suffix == 'baseline':
                baseline = entry
            elif suffix.startswith('noise_'):
                noise_results[suffix] = entry
            elif suffix.startswith('grid_'):
                parts = suffix.split('_')
                if len(parts) >= 3:
                    row, col = int(parts[1]), int(parts[2])
                    grid[f"{row}_{col}"] = {
                        **entry, 'row': row, 'col': col,
                        'bbox': data.get('perturbation_kwargs', {}).get('bbox', []),
                    }

    if not grid and not baseline:
        return jsonify({'status': 404, 'error': f'No grid ablation data for {task}'}), 404

    if grid:
        min_cell = min(grid.values(), key=lambda x: x['success_rate'])
        max_cell = max(grid.values(), key=lambda x: x['success_rate'])
    else:
        min_cell = max_cell = None

    return jsonify({
        'model': 'act',
        'task': task,
        'available_tasks': ['AlohaInsertion-v0', 'AlohaTransferCube-v0'],
        'grid_size': 4,
        'grid': grid,
        'baseline': baseline,
        'noise': noise_results,
        'critical_cell': f"{min_cell['row']}_{min_cell['col']}" if min_cell else None,
        'critical_sr': min_cell['success_rate'] if min_cell else None,
        'best_cell': f"{max_cell['row']}_{max_cell['col']}" if max_cell else None,
        'video_base_path': f'/videos/act/{task.replace("gym_aloha/", "")}',
        'data_source': data_source,
    })


def _grid_ablation_openvla(suite: str):
    # OpenVLA-OFT grid ablation from per-run results files
    results_file = _find_latest_oft_result(suite, 'grid_ablation')
    if not results_file:
        return jsonify({'status': 404, 'error': f'No OFT grid ablation for {suite}'}), 404

    with open(results_file) as f:
        data = json.load(f)

    conditions = data.get('conditions', {})
    grid = {}
    for task_key, task_data in conditions.items():
        for cond_key, cond_val in task_data.items():
            if not cond_key.startswith('grid_'):
                continue
            if cond_key not in grid:
                grid[cond_key] = {
                    'successes': 0, 'total': 0,
                    'row': cond_val.get('row', 0), 'col': cond_val.get('col', 0),
                }
            grid[cond_key]['total'] += 1
            if cond_val.get('success'):
                grid[cond_key]['successes'] += 1

    for k in grid:
        grid[k]['success_rate'] = grid[k]['successes'] / max(grid[k]['total'], 1)

    return jsonify({
        'model': 'openvla',
        'suite': suite,
        'available_suites': ['libero_goal', 'libero_10', 'libero_object', 'libero_spatial'],
        'grid_size': 4,
        'grid': grid,
        'source_file': str(results_file.relative_to(OFT_DATA_DIR)),
    })


def _grid_ablation_generic(model: str, suite: str):
    # Grid ablation for xvla / smolvla / groot / pi05 from experiment_results JSON
    exp_data = _load_experiment_results_cached(model)
    if exp_data is None:
        return jsonify({'status': 404, 'error': f'No experiment results for {model}'}), 404

    ga = exp_data.get('grid_ablation', {})
    suite_data = _resolve_suite_data(ga, suite)

    if not suite_data:
        return jsonify({
            'model': model, 'suite': suite, 'available_suites': list(ga.keys()),
            'grid_size': 4, 'grid': {},
            'baseline': {'success_rate': 0, 'mean_reward': 0, 'n_episodes': 0},
            'grid_type': 'layer_ablation', 'empty': True,
            'message': f'No grid ablation data for {model}/{suite}',
        })

    per_layer = suite_data.get('per_layer', {}) or suite_data.get('per_condition', {})
    baseline_data = suite_data.get('baseline', suite_data.get('baseline_overall', {}))

    # Compute baseline success rate from flat {task: sr} or nested {task: {success_rate: sr}}
    baseline_sr = 0.0
    if isinstance(baseline_data, dict) and baseline_data:
        vals = []
        for v in baseline_data.values():
            if isinstance(v, (int, float)):
                vals.append(v)
            elif isinstance(v, dict) and 'success_rate' in v:
                vals.append(v['success_rate'])
        if vals:
            baseline_sr = sum(vals) / len(vals)

    layer_keys = sorted(per_layer.keys(), key=_natural_sort_key)
    max_layers = min(len(layer_keys), 32)
    grid_cols = 4 if max_layers <= 16 else 6

    grid = {}
    for idx, layer_key in enumerate(layer_keys[:max_layers]):
        row, col = divmod(idx, grid_cols)
        layer_data = per_layer[layer_key]
        sr = layer_data.get('overall_success_rate', 0)
        grid[f"{row}_{col}"] = {
            'row': row, 'col': col, 'success_rate': sr, 'mean_reward': sr,
            'n_episodes': layer_data.get('n_episodes', len(layer_data.get('per_task', {}))),
            'label': layer_key,
        }

    return jsonify({
        'model': model, 'suite': suite, 'available_suites': list(ga.keys()),
        'grid_size': grid_cols, 'grid': grid,
        'baseline': {
            'success_rate': baseline_sr, 'mean_reward': baseline_sr,
            'n_episodes': len(baseline_data) if isinstance(baseline_data, dict) else 0,
        },
        'grid_type': 'layer_ablation',
    })
# Counterfactual
@interventions_bp.route('/api/vla/counterfactual', methods=['GET'])
def get_counterfactual():
    # Counterfactual prompting results from OFT and Pi0.5 experiments
    model = request.args.get('model', 'openvla')
    suite = request.args.get('suite', 'libero_goal')

    if model == 'openvla':
        results_file = _find_latest_oft_result(suite, 'counterfactual_prompting')
        if results_file:
            return _counterfactual_openvla(results_file, suite)
        # No raw OFT results files; fall through to generic handler as 'oft'
        model = 'oft'

    if model == 'pi05':
        return _counterfactual_pi05(suite)

    if model in ('xvla', 'smolvla', 'groot', 'act', 'oft'):
        return _counterfactual_generic(model, suite)

    return jsonify({'status': 404, 'error': f'No counterfactual data for model {model}'}), 404


def _counterfactual_openvla(results_file: Path, suite: str):
    # Counterfactual from raw OFT per-run results files
    with open(results_file) as f:
        data = json.load(f)

    conditions = data.get('conditions', {})
    tasks = {}
    prompt_types = set()
    for task_key in sorted(conditions.keys()):
        task_data = conditions[task_key]
        task_id = task_key.replace('task_', '')
        tasks[task_id] = {}
        for prompt_type, result in task_data.items():
            prompt_types.add(prompt_type)
            tasks[task_id][prompt_type] = {
                'prompt': result.get('prompt', ''),
                'success': result.get('success', False),
                'n_steps': result.get('n_steps', 0),
            }

    summary = {}
    for pt in sorted(prompt_types):
        successes = sum(1 for t in tasks.values() if t.get(pt, {}).get('success', False))
        total = sum(1 for t in tasks.values() if pt in t)
        summary[pt] = {
            'success_rate': successes / max(total, 1),
            'successes': successes, 'total': total,
        }

    run_dir = results_file.parent.parent
    video_dir = run_dir / 'counterfactual_prompting'
    videos = {}
    if video_dir.exists():
        for vf in video_dir.glob("*.mp4"):
            videos[vf.stem] = (
                f"/videos/openvla/openvla_oft/{suite}/"
                f"{run_dir.name}/counterfactual_prompting/{vf.name}"
            )

    return jsonify({
        'model': 'openvla',
        'suite': suite,
        'available_suites': ['libero_goal', 'libero_10', 'libero_object', 'libero_spatial'],
        'prompt_types': sorted(list(prompt_types)),
        'tasks': tasks,
        'summary': summary,
        'videos': videos,
    })


def _counterfactual_pi05(suite: str):
    # Pi0.5 counterfactual data from validation results
    suite_key = suite_short(suite)

    valid_dir = Path(__file__).parent.parent / 'results' / 'valid' / 'counterfactual'
    results_file = None
    for f in sorted(valid_dir.glob(f'counterfactual_{suite}_*.json'), reverse=True):
        results_file = f
        break

    if not results_file:
        return jsonify({'status': 404, 'error': f'No Pi0.5 counterfactual data for {suite}'}), 404

    with open(results_file) as f:
        data = json.load(f)

    analyses = data.get('analyses', {})
    category_behaviors = analyses.get('category_behaviors', {})

    key_categories = [
        'baseline', 'null', 'object_swap', 'verb_swap', 'motor',
        'negation', 'spatial_swap', 'wrong_object', 'conflict', 'object_only',
    ]
    prompt_types = [c for c in key_categories if c in category_behaviors]

    tasks = {}
    summary = {}
    for cat in prompt_types:
        info = category_behaviors[cat]
        n_eps = info.get('n_episodes', 0)
        mean_steps = info.get('n_steps_mean', 250)
        std_steps = info.get('n_steps_std', 0)

        if mean_steps >= 249.5:
            sr = 0.0
        elif std_steps == 0 and mean_steps < 249:
            sr = 1.0
        else:
            sr = max(0.0, min(1.0, 1.0 - (mean_steps / 250.0)))

        successes = round(sr * n_eps)
        summary[cat] = {'success_rate': sr, 'successes': successes, 'total': n_eps}
        tasks[cat] = {
            cat: {
                'prompt': f'{cat} ({n_eps} episodes)',
                'success': sr > 0,
                'n_steps': int(mean_steps),
            }
        }

    layer_div = analyses.get('layer_divergence', {}).get('expert_layers', {})
    key_finding = None
    if layer_div:
        max_l2 = 0
        max_layer = ''
        for layer, vals in layer_div.items():
            l2 = vals.get('baseline_vs_null_l2', 0)
            if l2 > max_l2:
                max_l2 = l2
                max_layer = layer
        key_finding = (
            f"Pi0.5 counterfactual analysis: {data.get('n_episodes', 0)} episodes across "
            f"{len(prompt_types)} prompt categories. Max activation divergence at {max_layer} "
            f"(L2={max_l2:.1f}). Categories with n_steps=250 indicate complete task failure."
        )

    return jsonify({
        'model': 'pi05',
        'suite': suite,
        'available_suites': ['libero_goal', 'libero_10', 'libero_object', 'libero_spatial'],
        'prompt_types': prompt_types,
        'tasks': tasks,
        'summary': summary,
        'videos': {},
        'key_finding': key_finding,
    })


def _counterfactual_generic(model: str, suite: str):
    # Counterfactual for xvla / smolvla / groot / act / oft from baked or experiment results
    # Try baked SimplerEnv/custom counterfactual data first
    if 'simplerenv' in suite:
        baked = load_json_cached(
            API_DATA_DIR / f'{model}_simplerenv_counterfactual_baked.json',
            f'{model}_simplerenv_cf_baked',
        )
        if baked:
            suite_cf = baked.get(suite, {})
            if suite_cf:
                cf_tasks = suite_cf.get('tasks', [])
                categories = suite_cf.get('categories', {})
                formatted_tasks = []
                for t in cf_tasks:
                    task_entry = {'task': t.get('task', ''), 'prompt': t.get('prompt', '')}
                    for cname, cdata in t.get('conditions', {}).items():
                        task_entry[cname] = cdata
                    formatted_tasks.append(task_entry)

                summary = {}
                for cname, cdata in categories.items():
                    summary[cname] = {
                        'success_rate': cdata.get('success_rate', 0),
                        'total': cdata.get('total', 0),
                        'successes': cdata.get('successes', 0),
                    }
                return jsonify({
                    'data': {
                        'tasks': formatted_tasks,
                        'summary': summary,
                        'prompt_types': list(categories.keys()),
                        'model': model,
                        'suite': suite,
                    }
                })

    # Load counterfactual data from experiment_results_<model>.json
    exp_data = _load_experiment_results_cached(model)
    if exp_data is None:
        return jsonify({'status': 404, 'error': f'No experiment results for {model}'}), 404

    cf = exp_data.get('counterfactual', {})
    suite_data = _resolve_suite_data(cf, suite)

    if not suite_data:
        return jsonify({
            'model': model, 'suite': suite,
            'available_suites': list(cf.keys()),
            'prompt_types': [], 'tasks': {}, 'summary': {}, 'videos': {},
            'empty': True,
            'message': f'No counterfactual data for {model}/{suite}',
        })

    per_prompt = suite_data.get('per_prompt_type', suite_data.get('per_condition', {}))
    runs_list = suite_data.get('runs', [])

    summary = {}
    tasks: Dict[str, dict] = {}

    if per_prompt:
        prompt_types_list = sorted(per_prompt.keys())
        for pt in prompt_types_list:
            pt_data = per_prompt[pt]
            sr = pt_data.get('overall_success_rate', pt_data.get('success_rate', 0))
            n_tasks = pt_data.get('n_tasks', len(pt_data.get('per_task', {})))
            n_success = round(sr * n_tasks) if sr <= 1.0 else round(sr / 100 * n_tasks)
            summary[pt] = {
                'success_rate': sr if sr <= 1.0 else sr / 100,
                'successes': n_success, 'total': n_tasks,
            }
            for tid, tsr in pt_data.get('per_task', {}).items():
                task_str = str(tid)
                if task_str not in tasks:
                    tasks[task_str] = {}
                task_success = tsr > 0.5 if isinstance(tsr, (int, float)) else bool(tsr)
                tasks[task_str][pt] = {
                    'prompt': f'{pt} ({task_str})',
                    'success': task_success,
                    'n_steps': 0,
                }

    elif runs_list:
        prompt_types_list = ['counterfactual']
        total_episodes = 0
        total_successes = 0
        for i, run in enumerate(runs_list):
            run_id = run.get('run_id', f'run_{i}')
            sr = run.get('success_rate', 0)
            n_eps = run.get('n_episodes', 0)
            n_succ = round(sr * n_eps) if n_eps > 0 else 0
            total_episodes += n_eps
            total_successes += n_succ
            tasks[str(i)] = {
                'counterfactual': {
                    'prompt': f'Run {run_id} ({n_eps} episodes, {run.get("n_conditions", "?")} conditions)',
                    'success': sr > 0.5,
                    'n_steps': n_eps,
                }
            }
        if total_episodes > 0:
            summary['counterfactual'] = {
                'success_rate': total_successes / total_episodes,
                'successes': total_successes, 'total': total_episodes,
            }
        else:
            summary['counterfactual'] = {'success_rate': 0, 'successes': 0, 'total': len(runs_list)}

    else:
        prompt_types_list = _build_counterfactual_from_per_task(suite_data, tasks, summary)

    key_finding = None
    if summary:
        baseline_sr = summary.get('baseline', {}).get('success_rate', 0)
        worst_cat = min(summary.items(), key=lambda x: x[1]['success_rate'])
        if baseline_sr > 0 and worst_cat[0] != 'baseline':
            key_finding = (
                f"{model.upper()} counterfactual: baseline {baseline_sr*100:.0f}% success, "
                f"worst category '{worst_cat[0]}' at {worst_cat[1]['success_rate']*100:.0f}% "
                f"({worst_cat[1].get('total', 0)} episodes)."
            )

    return jsonify({
        'model': model, 'suite': suite,
        'available_suites': list(cf.keys()),
        'prompt_types': prompt_types_list,
        'tasks': tasks,
        'summary': summary,
        'videos': {},
        'key_finding': key_finding,
    })


def _build_counterfactual_from_per_task(
    suite_data: dict, tasks: Dict[str, dict], summary: Dict[str, dict],
) -> list:
    """
    Build counterfactual tasks/summary from per_task dict with conditions inside each task.

    Used for X-VLA, GR00T, SmolVLA formats. Mutates *tasks* and *summary* in place
    and returns the sorted list of prompt types.
    """
    per_task_raw = suite_data.get('per_task', {})
    if isinstance(per_task_raw, list):
        per_task_raw = {}
    categories = suite_data.get('categories', [])

    if isinstance(categories, dict):
        prompt_types_set = set(categories.keys())
    elif isinstance(categories, list):
        prompt_types_set = set(categories)
    else:
        prompt_types_set = set()

    for tid, task_info in per_task_raw.items():
        task_str = str(tid)
        conditions = task_info.get('conditions', {})
        if task_str not in tasks:
            tasks[task_str] = {}
        for cond_name, cond_data in conditions.items():
            prompt_types_set.add(cond_name)
            cond_sr = cond_data.get('success_rate', 0)
            tasks[task_str][cond_name] = {
                'prompt': cond_data.get('prompt', f'{cond_name} ({task_str})'),
                'success': cond_sr > 0.5 if isinstance(cond_sr, (int, float)) else bool(cond_sr),
                'n_steps': int(cond_data.get('avg_steps', 0)),
            }

    prompt_types_list = sorted(prompt_types_set)

    for pt in prompt_types_list:
        total = 0
        successes = 0
        for task_data in tasks.values():
            if pt in task_data:
                total += 1
                if task_data[pt].get('success'):
                    successes += 1

        total_eps = 0
        success_eps = 0
        for task_info in per_task_raw.values():
            cond_data = task_info.get('conditions', {}).get(pt, {})
            n_ep = cond_data.get('n_episodes', 0)
            sr = cond_data.get('success_rate', 0)
            total_eps += n_ep
            success_eps += round(sr * n_ep) if n_ep > 0 else 0

        if total_eps > 0:
            summary[pt] = {
                'success_rate': success_eps / total_eps,
                'successes': success_eps, 'total': total_eps,
            }
        elif total > 0:
            summary[pt] = {
                'success_rate': successes / total,
                'successes': successes, 'total': total,
            }
        elif isinstance(categories, dict) and pt in categories:
            cat_info = categories[pt]
            if isinstance(cat_info, dict):
                cat_total = cat_info.get('n_episodes', 0)
                cat_sr = cat_info.get('success_rate', 0)
                cat_successes = round(cat_sr * cat_total) if cat_total > 0 else 0
                summary[pt] = {
                    'success_rate': cat_sr,
                    'successes': cat_successes, 'total': cat_total,
                }
            else:
                summary[pt] = {'success_rate': 0, 'successes': 0, 'total': 0}
        else:
            summary[pt] = {'success_rate': 0, 'successes': 0, 'total': 0}

    # Filter out prompt types with 0 total episodes
    prompt_types_list = [pt for pt in prompt_types_list if summary.get(pt, {}).get('total', 0) > 0]
    for pt in list(summary):
        if summary[pt].get('total', 0) == 0:
            del summary[pt]

    # If per_task was empty but categories has stats, build synthetic task entries
    # so the frontend task grid is populated (SmolVLA categories-only format)
    if not tasks and isinstance(categories, dict):
        for cat_name, cat_info in categories.items():
            if not isinstance(cat_info, dict):
                continue
            cat_total = cat_info.get('n_episodes', 0)
            if cat_total <= 0:
                continue
            cat_sr = cat_info.get('success_rate', 0)
            avg_steps = cat_info.get('avg_steps', 0)
            tasks[cat_name] = {
                cat_name: {
                    'prompt': f'{cat_name} ({cat_total} episodes)',
                    'success': cat_sr > 0.5,
                    'n_steps': int(avg_steps),
                }
            }
            per_source = cat_info.get('per_source', {})
            for src_key, src_info in per_source.items():
                src_n = src_info.get('n_episodes', 0)
                src_sr = src_info.get('success_rate', 0)
                tasks[f'{cat_name}_{src_key}'] = {
                    cat_name: {
                        'prompt': f'{cat_name}: {src_key} ({src_n} eps)',
                        'success': src_sr > 0.5,
                        'n_steps': 0,
                    }
                }

    return prompt_types_list
# Injection