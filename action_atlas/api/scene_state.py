"""Action Atlas API - scene_state routes."""
from flask import Blueprint, request, jsonify, send_file, abort, make_response, redirect
from .helpers import *
from .data_loaders import *

scene_state_bp = Blueprint("scene_state", __name__)


def _load_merged_results(suite: str, seed: str = 'seed123') -> Optional[dict]:
    """
    Load merged_results.json for Pi0.5 cross-task experiments. Uses cache.

    For suites without a merged_results.json (e.g. LIBERO-10), merges all
    available batch results.json files into a single dict.
    """
    cache_key = f"{suite}_{seed}"
    if cache_key in _scene_state_cache:
        return _scene_state_cache[cache_key]

    # Map suite name to directory paths that may contain merged_results.json
    suite_dirs = {
        'goal': [
            PI05_ROLLOUTS_DIR / 'cross_task_goal' / f'cross_task_libero_goal_{seed}_20260127_195312',
            PI05_ROLLOUTS_DIR / 'cross_task_goal' / 'cross_task_libero_goal_seed123_20260127_195312',
        ],
        'spatial': [
            PI05_ROLLOUTS_DIR / 'cross_task_spatial' / 'cross_task_libero_spatial_20260126_225619',
        ],
        '10': [
            PI05_ROLLOUTS_DIR / 'cross_task_10' / 'cross_task_libero_10_20260127_051354',
            PI05_ROLLOUTS_DIR / 'cross_task_10' / 'cross_task_libero_10_20260127_085109',
        ],
        'object': [
            PI05_ROLLOUTS_DIR / 'groot_feb13_pi05' / 'object',
        ],
    }

    # groot Feb 13-14 data uses results.json (not merged_results.json) with pairs at top level
    # 45 pairs each, 8 conditions × 2 directions, with full trajectory + video data
    pi05_suites = {
        'object': PI05_ROLLOUTS_DIR / 'groot_feb13_pi05' / 'object' / 'results.json',
        '10': PI05_ROLLOUTS_DIR / 'groot_feb13_pi05' / '10' / 'results.json',
    }

    paths_to_try = suite_dirs.get(suite, [])

    # Try groot results.json (raw format with pairs at top level, no "results" wrapper)
    pi05_path = pi05_suites.get(suite)
    if pi05_path and pi05_path.exists():
        try:
            with open(pi05_path) as f:
                raw = json.load(f)
            # Wrap raw pair data in expected format
            if 'results' not in raw and isinstance(raw, dict):
                data = {'results': raw, 'suite': suite}
            else:
                data = raw
            _scene_state_cache[cache_key] = data
            print(f"Loaded groot results from {pi05_path} ({len(data.get('results', {}))} pairs)")
            return data
        except Exception as e:
            print(f"Error loading groot {pi05_path}: {e}")

    # First try merged_results.json (goal / spatial)
    for base_dir in paths_to_try:
        merged_path = base_dir / 'merged_results.json'
        if merged_path.exists():
            try:
                with open(merged_path) as f:
                    data = json.load(f)
                _scene_state_cache[cache_key] = data
                print(f"Loaded merged_results from {merged_path}")
                return data
            except Exception as e:
                print(f"Error loading {merged_path}: {e}")
                continue

    # Fallback: merge per-batch results.json files (LIBERO-10)
    merged_results: dict = {}
    merged_meta: dict = {}
    for base_dir in paths_to_try:
        if not base_dir.exists():
            continue
        for batch_dir in sorted(base_dir.iterdir()):
            batch_json = batch_dir / 'results.json'
            if not batch_json.exists():
                continue
            try:
                with open(batch_json) as f:
                    batch_data = json.load(f)
                # Capture metadata from first valid batch
                if not merged_meta:
                    merged_meta = {
                        'suite': batch_data.get('suite', suite),
                        'task_pairs': batch_data.get('task_pairs', []),
                        'seed': batch_data.get('seed', seed),
                        'max_steps': batch_data.get('max_steps', 520),
                        'task_prompts': batch_data.get('task_prompts', {}),
                    }
                else:
                    # Merge task_prompts
                    merged_meta.setdefault('task_prompts', {}).update(
                        batch_data.get('task_prompts', {})
                    )
                # Merge pair results (later batches can add new pairs)
                for pair_id, pair_data in batch_data.get('results', {}).items():
                    if pair_id not in merged_results:
                        merged_results[pair_id] = pair_data
                    # Don't overwrite existing pairs - first occurrence wins
                print(f"Merged batch {batch_json}")
            except Exception as e:
                print(f"Error loading batch {batch_json}: {e}")
                continue

    if merged_results:
        data = {**merged_meta, 'results': merged_results}
        _scene_state_cache[cache_key] = data
        print(f"Assembled {len(merged_results)} pairs for suite={suite} from batch files")
        return data

    # Final fallback: load pre-processed lightweight summary (Fly.io deployment)
    # Pi0.5 files use short suite names (goal.json, not libero_goal.json)
    suite_short = suite.replace('libero_', '') if suite.startswith('libero_') else suite
    baked_path = Path(__file__).parent / 'data' / 'pi05_scene_state' / f'{suite_short}.json'
    if not baked_path.exists():
        baked_path = Path(__file__).parent / 'data' / 'pi05_scene_state' / f'{suite}.json'
    if baked_path.exists():
        try:
            with open(baked_path) as f:
                data = json.load(f)
            # Reshape baked summary into the expected format with 'results' dict
            if 'pairs' in data and 'results' not in data:
                results = {}
                for pair in data['pairs']:
                    pair_key = pair.get('key', f"pair_{pair.get('task_a')}_{pair.get('task_b')}")
                    results[pair_key] = pair
                data['results'] = results
            _scene_state_cache[cache_key] = data
            print(f"Loaded baked scene state summary from {baked_path}")
            return data
        except Exception as e:
            print(f"Error loading baked scene state {baked_path}: {e}")

    print(f"No scene state data found for suite={suite}, seed={seed}")
    return None


def _load_oft_scene_state(suite: str) -> Optional[dict]:
    oft_dir = Path(__file__).parent / 'data' / 'oft_scene_state'
    suite_map = {
        'goal': 'libero_goal', 'object': 'libero_object',
        'spatial': 'libero_spatial', '10': 'libero_10',
    }
    file_suite = suite_map.get(suite, suite)
    candidates = [
        oft_dir / f'{suite}.json',
        oft_dir / f'{file_suite}_cross_task.json',
        oft_dir / f'{file_suite}_baseline.json',
        oft_dir / f'{file_suite}.json',
    ]
    baked_path = next((p for p in candidates if p.exists()), None)
    if baked_path is None:
        return None
    try:
        with open(baked_path) as f:
            data = json.load(f)
        if 'pairs' in data and 'results' not in data:
            results = {}
            for pair in data['pairs']:
                pair_key = pair.get('key', f"pair_{pair.get('task_a')}_{pair.get('task_b')}")
                results[pair_key] = pair
            data['results'] = results
        return data
    except Exception as e:
        print(f"Error loading OFT scene state {baked_path}: {e}")
        return None


_model_scene_state_cache: Dict[str, dict] = {}


def _load_model_scene_state(model: str, suite: str, experiment_type: str = 'baseline') -> Optional[dict]:
    """Load baked scene state data for SmolVLA, X-VLA, or GR00T."""
    model_dirs = {
        'xvla': 'xvla_scene_state',
        'smolvla': 'smolvla_scene_state',
        'groot': 'groot_scene_state',
        'act': 'act_scene_state',
    }
    dir_name = model_dirs.get(model)
    if not dir_name:
        return None

    # Map suite names to file names
    suite_map = {
        'goal': 'libero_goal', 'object': 'libero_object',
        'spatial': 'libero_spatial', '10': 'libero_10',
        'long': 'libero_long',
        'simplerenv_widowx': 'simplerenv_widowx',
        'simplerenv_google_robot': 'simplerenv_google_robot',
        'metaworld': 'metaworld',
        'metaworld_easy': 'metaworld_easy',
        'metaworld_medium': 'metaworld_medium',
        'metaworld_hard': 'metaworld_hard',
        'metaworld_very_hard': 'metaworld_very_hard',
    }
    file_suite = suite_map.get(suite, suite)

    # Try different file patterns
    data_dir = Path(__file__).parent / 'data' / dir_name
    candidates = [
        data_dir / f'{file_suite}_{experiment_type}.json',
        data_dir / f'{file_suite}_cross_task.json',
        data_dir / f'{file_suite}.json',
    ]

    cache_key = f"{model}_{suite}_{experiment_type}"
    if cache_key in _model_scene_state_cache:
        return _model_scene_state_cache[cache_key]

    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if 'difficulties' in data and 'results' not in data:
                    # MetaWorld aggregated format: difficulties -> {easy/medium/hard/very_hard}
                    # Each difficulty level can contain either 'pairs' (cross_task) or 'tasks'
                    # (grid_ablation, counterfactual, vision_perturbation).
                    has_pairs = any(
                        'pairs' in dd for dd in data['difficulties'].values()
                    )
                    if has_pairs:
                        # Flatten all difficulty levels' pairs into a single results dict,
                        # tagging each pair with its difficulty level.
                        results = {}
                        for difficulty, diff_data in data['difficulties'].items():
                            for pair in diff_data.get('pairs', []):
                                task_a = pair.get('task_a', '')
                                task_b = pair.get('task_b', '')
                                pair_key = pair.get('key', f"pair_{task_a}_{task_b}")
                                pair_entry = dict(pair)
                                pair_entry['difficulty'] = difficulty
                                # Normalize field names: task_a_desc/task_b_desc -> prompt_a/prompt_b
                                if 'prompt_a' not in pair_entry and 'task_a_desc' in pair_entry:
                                    pair_entry['prompt_a'] = pair_entry['task_a_desc']
                                if 'prompt_b' not in pair_entry and 'task_b_desc' in pair_entry:
                                    pair_entry['prompt_b'] = pair_entry['task_b_desc']
                                results[pair_key] = pair_entry
                        data['results'] = results
                    else:
                        # Flatten difficulties with tasks format: merge all tasks
                        # across difficulties, prefixing task_id with difficulty,
                        # then convert to pairs via _convert_tasks_to_pairs.
                        merged_tasks = []
                        for difficulty, diff_data in data['difficulties'].items():
                            for task in diff_data.get('tasks', []):
                                task_entry = dict(task)
                                # Prefix task_id with difficulty to keep them unique
                                orig_id = task_entry.get('task_id', task_entry.get('task_name', ''))
                                task_entry['task_id'] = f"{difficulty}/{orig_id}"
                                desc = task_entry.get('task_description', str(orig_id))
                                task_entry['task_description'] = f"[{difficulty}] {desc}"
                                task_entry['difficulty'] = difficulty
                                merged_tasks.append(task_entry)
                        data['tasks'] = merged_tasks
                        data = _convert_tasks_to_pairs(data)
                elif 'pairs' in data and 'results' not in data:
                    results = {}
                    for pair in data['pairs']:
                        pair_key = pair.get('key', f"pair_{pair.get('task_a')}_{pair.get('task_b')}")
                        results[pair_key] = pair
                    data['results'] = results
                elif 'tasks' in data and 'results' not in data:
                    # Convert tasks-based format (SmolVLA/GR00T baseline/ablation)
                    # into pairs+results format for the scene state UI.
                    data = _convert_tasks_to_pairs(data)
                _model_scene_state_cache[cache_key] = data
                print(f"Loaded {model} scene state from {path}")
                return data
            except Exception as e:
                print(f"Error loading {model} scene state {path}: {e}")
                continue
    return None


def _convert_tasks_to_pairs(data: dict) -> dict:
    """
    Convert tasks-based scene state format to pairs+results format.

    Tasks format has per-task trials with trajectories.  We generate
    pairwise entries so the scene-state UI can display baseline
    trajectories side-by-side for every task pair.
    """
    tasks = data.get('tasks', [])
    if not tasks:
        return data

    results: Dict[str, dict] = {}
    task_prompts: Dict[str, str] = {}

    for t in tasks:
        task_prompts[str(t['task_id'])] = t.get('task_description', f"Task {t['task_id']}")

    for i, ta in enumerate(tasks):
        for j, tb in enumerate(tasks):
            if i >= j:
                continue
            pair_key = f"pair_{ta['task_id']}_{tb['task_id']}"
            conditions: Dict[str, dict] = {}

            # Build baseline conditions from the first successful trial
            # (or first trial if none succeeded).
            for label, task_data in [('baseline_task_0', ta), ('baseline_task_1', tb)]:
                trials = task_data.get('trials', [])
                if not trials:
                    continue
                # Prefer a successful trial
                trial = next((tr for tr in trials if tr.get('success')), trials[0])
                conditions[label] = {
                    'n_steps': trial.get('n_steps', 0),
                    'success': trial.get('success', False),
                    'robot_eef_trajectory': trial.get('robot_eef_trajectory', []),
                    'object_trajectories': trial.get('object_trajectories', {}),
                    'object_displacements': trial.get('object_displacements', {}),
                }

            results[pair_key] = {
                'key': pair_key,
                'task_a': ta['task_id'],
                'task_b': tb['task_id'],
                'prompt_a': ta.get('task_description', f"Task {ta['task_id']}"),
                'prompt_b': tb.get('task_description', f"Task {tb['task_id']}"),
                'conditions': conditions,
            }

    data['results'] = results
    data['task_prompts'] = task_prompts
    return data


@scene_state_bp.route('/api/vla/scene_state/pairs', methods=['GET'])
def get_scene_state_pairs():
    """List available pairs for cross-task scene state data."""
    suite = request.args.get('suite', 'goal')
    seed = request.args.get('seed', 'seed123')
    model = request.args.get('model', 'pi05')
    experiment_type = request.args.get('experiment_type', '')

    # Discover available experiment types for this model/suite
    available_experiment_types = []
    if model in ('xvla', 'smolvla', 'groot', 'act'):
        model_dirs = {
            'xvla': 'xvla_scene_state',
            'smolvla': 'smolvla_scene_state',
            'groot': 'groot_scene_state',
            'act': 'act_scene_state',
        }
        dir_name = model_dirs.get(model, '')
        suite_map = {
            'goal': 'libero_goal', 'object': 'libero_object',
            'spatial': 'libero_spatial', '10': 'libero_10',
            'long': 'libero_long',
            'metaworld': 'metaworld',
            'metaworld_easy': 'metaworld_easy',
            'metaworld_medium': 'metaworld_medium',
            'metaworld_hard': 'metaworld_hard',
            'metaworld_very_hard': 'metaworld_very_hard',
        }
        file_suite = suite_map.get(suite, suite)
        data_dir = Path(__file__).parent / 'data' / dir_name
        if data_dir.exists():
            prefix = f'{file_suite}_'
            for f in data_dir.glob(f'{prefix}*.json'):
                # Extract experiment type from filename: {suite}_{experiment_type}.json
                exp_type = f.stem[len(prefix):]
                if exp_type and not exp_type.endswith('_compact'):
                    available_experiment_types.append(exp_type)
            available_experiment_types = sorted(set(available_experiment_types))

    if model == 'openvla':
        data = _load_oft_scene_state(suite)
    elif model in ('xvla', 'smolvla', 'groot', 'act'):
        if experiment_type:
            data = _load_model_scene_state(model, suite, experiment_type)
        else:
            # Default: try cross_task first, then baseline, then first available
            data = _load_model_scene_state(model, suite, 'cross_task')
            if data is None:
                data = _load_model_scene_state(model, suite, 'baseline')
            if data is None and available_experiment_types:
                data = _load_model_scene_state(model, suite, available_experiment_types[0])
    else:
        data = _load_merged_results(suite, seed)
    if data is None:
        # Return empty but valid response instead of 404 (data may not exist yet, e.g. MetaWorld)
        return jsonify({'pairs': [], 'model': model, 'suite': suite,
                        'message': f'No scene state data available yet for {model}/{suite}'})

    results = data.get('results', {})
    task_prompts = data.get('task_prompts', {})

    pairs = []
    for pair_id in sorted(results.keys()):
        pair_data = results[pair_id]
        task_a = pair_data.get('task_a', '')
        task_b = pair_data.get('task_b', '')
        prompt_a = pair_data.get('prompt_a', task_prompts.get(str(task_a), f'Task {task_a}'))
        prompt_b = pair_data.get('prompt_b', task_prompts.get(str(task_b), f'Task {task_b}'))

        # Collect available conditions dynamically from actual data keys.
        # Supports three layouts:
        #   1) Old top-level: pair_data has keys like baseline_task_0, inject_0_into_1 (nested sub-dict)
        #   2) New baked format: pair_data['conditions'] is a flat dict with keys like
        #      "baseline_task_0", "inject_0_into_1/cross_prompt_pali_ALL"
        #   3) Hybrid: both may exist (conditions dict takes precedence for extras)
        conditions = []
        conditions_dict = pair_data.get('conditions', {})

        # --- From the 'conditions' sub-dict (new baked format, flat keys) ---
        for key in sorted(conditions_dict.keys()):
            if key.startswith('baseline_task_'):
                conditions.append(key)
        for key in sorted(conditions_dict.keys()):
            if not key.startswith('baseline_task_'):
                conditions.append(key)

        # --- From top-level pair_data keys (old format) ---
        # Baselines: any key matching baseline_task_*
        for key in sorted(pair_data.keys()):
            if key.startswith('baseline_task_') and key not in conditions:
                conditions.append(key)
        # Injection conditions: any key matching inject_*_into_* (nested sub-dict)
        for key in sorted(pair_data.keys()):
            if key.startswith('inject_') and '_into_' in key:
                inject_data = pair_data[key]
                if isinstance(inject_data, dict):
                    for cond_name in sorted(inject_data.keys()):
                        flat_key = f"{key}/{cond_name}"
                        if flat_key not in conditions:
                            conditions.append(flat_key)

        pair_entry = {
            'id': pair_id,
            'task_a': task_a,
            'task_b': task_b,
            'prompt_a': prompt_a,
            'prompt_b': prompt_b,
            'conditions': conditions,
        }
        # Preserve difficulty level from flattened difficulties format
        if 'difficulty' in pair_data:
            pair_entry['difficulty'] = pair_data['difficulty']
        pairs.append(pair_entry)

    response = {
        'suite': suite,
        'seed': seed,
        'total_pairs': len(pairs),
        'pairs': pairs,
        'available_seeds': ['seed123', 'seed42', 'seed456'],
    }
    if available_experiment_types:
        response['experiment_types'] = available_experiment_types
        response['active_experiment_type'] = experiment_type if experiment_type else (
            'cross_task' if 'cross_task' in available_experiment_types else
            available_experiment_types[0] if available_experiment_types else ''
        )
    return jsonify(response)


@scene_state_bp.route('/api/vla/scene_state', methods=['GET'])
def get_scene_state():
    """Get trajectory data for a specific pair and condition."""
    suite = request.args.get('suite', 'goal')
    pair = request.args.get('pair', 'pair_0_1')
    condition = request.args.get('condition', 'baseline_task_0')
    seed = request.args.get('seed', 'seed123')
    model = request.args.get('model', 'pi05')

    experiment_type = request.args.get('experiment_type', '')

    if model == 'openvla':
        data = _load_oft_scene_state(suite)
    elif model in ('xvla', 'smolvla', 'groot', 'act'):
        if experiment_type:
            data = _load_model_scene_state(model, suite, experiment_type)
        else:
            data = _load_model_scene_state(model, suite, 'cross_task')
            if data is None:
                data = _load_model_scene_state(model, suite, 'baseline')
    else:
        data = _load_merged_results(suite, seed)
    if data is None:
        # Return empty but valid response instead of 404 (data may not exist yet, e.g. MetaWorld)
        return jsonify({'model': model, 'suite': suite, 'pair': pair, 'condition': condition,
                        'robot_eef_trajectory': [], 'object_trajectories': {},
                        'n_steps': 0, 'empty': True,
                        'message': f'No scene state data available yet for {model}/{suite}'})

    results = data.get('results', {})
    if pair not in results:
        return jsonify({'model': model, 'suite': suite, 'pair': pair, 'condition': condition,
                        'robot_eef_trajectory': [], 'object_trajectories': {},
                        'n_steps': 0, 'empty': True,
                        'message': f'Pair {pair} not found in {model}/{suite}'})

    pair_data = results[pair]

    # Navigate to the condition.
    # Supports three layouts:
    #   1) New baked: pair_data['conditions'] is a flat dict with keys like
    #      "baseline_task_0" or "inject_0_into_1/cross_prompt_pali_ALL"
    #   2) Old format: top-level keys "baseline_task_N" or nested
    #      pair_data['inject_X_into_Y']['cond_name']
    conditions_dict = pair_data.get('conditions', {})
    cond_data = None

    # Try 1: flat lookup in 'conditions' sub-dict (new baked format)
    if condition in conditions_dict and isinstance(conditions_dict[condition], dict):
        cond_data = conditions_dict[condition]

    # Try 2: nested navigation for slash-separated conditions (old format)
    if cond_data is None and '/' in condition:
        parts = condition.split('/', 1)
        inject_key = parts[0]
        cond_name = parts[1]
        if inject_key in pair_data and isinstance(pair_data[inject_key], dict) and cond_name in pair_data[inject_key]:
            cond_data = pair_data[inject_key][cond_name]

    # Try 3: direct top-level key (old format baselines)
    if cond_data is None and condition in pair_data and isinstance(pair_data[condition], dict):
        cond_data = pair_data[condition]

    # Try 4: fuzzy baseline matching - handle naming differences between models
    if cond_data is None and 'baseline' in condition:
        # Look for any condition containing 'baseline'
        all_conds = list(conditions_dict.keys()) if conditions_dict else []
        # Also check top-level pair_data keys
        all_conds += [k for k in pair_data.keys() if isinstance(pair_data[k], dict)]
        baseline_conds = [c for c in all_conds if 'baseline' in c.lower()]
        if baseline_conds:
            # Pick the first matching baseline
            best = baseline_conds[0]
            cond_data = conditions_dict.get(best) or pair_data.get(best)
            condition = best  # Update for response

    if cond_data is None:
        available = list(conditions_dict.keys()) if conditions_dict else [k for k in pair_data.keys() if isinstance(pair_data.get(k), dict)]
        return jsonify({'error': f'Condition {condition} not found in {pair}', 'available_conditions': available[:20]}), 404

    if not isinstance(cond_data, dict):
        return jsonify({'error': f'Invalid condition data for {condition}'}), 400

    scene = cond_data.get('scene', {})

    # Baked data may have trajectory fields at the condition level (not nested under 'scene')
    eef_traj = scene.get('robot_eef_trajectory', []) or cond_data.get('robot_eef_trajectory', [])
    obj_traj = scene.get('object_trajectories', {}) or cond_data.get('object_trajectories', {})
    obj_disp = scene.get('object_displacements', {}) or cond_data.get('object_displacements', {})
    n_steps = scene.get('n_steps', 0) or cond_data.get('n_steps', 0)

    # Subsample EEF trajectory if too large (for frontend performance)
    max_points = 500
    if len(eef_traj) > max_points:
        step = len(eef_traj) / max_points
        eef_traj = [eef_traj[int(i * step)] for i in range(max_points)]

    # Check if a video exists for this condition (groot data has per-condition mp4s)
    video_url = None
    if '/' in condition:
        # injection condition: inject_X_into_Y/cond_name → groot_feb13_pi05/{suite}/{pair}/inject_X_into_Y/cond_name.mp4
        omen_video = PI05_ROLLOUTS_DIR / 'groot_feb13_pi05' / suite / pair / (condition + '.mp4')
        if omen_video.exists():
            video_url = f'/api/vla/video/pi05/groot_feb13_pi05/{suite}/{pair}/{condition}.mp4'

    return jsonify({
        'pair': pair,
        'condition': condition,
        'suite': suite,
        'task_a': pair_data.get('task_a'),
        'task_b': pair_data.get('task_b'),
        'prompt_a': pair_data.get('prompt_a', ''),
        'prompt_b': pair_data.get('prompt_b', ''),
        'steps': cond_data.get('steps', n_steps),
        'success': cond_data.get('success', False),
        'video_url': video_url,
        'scene': {
            'n_steps': n_steps,
            'robot_eef_trajectory': eef_traj,
            'object_trajectories': obj_traj,
            'object_displacements': obj_disp,
            'initial_state': scene.get('initial_state', {}) or cond_data.get('initial_state', {}),
            'final_state': scene.get('final_state', {}) or cond_data.get('final_state', {}),
        },
    })
# Concept Ablation Scene State (SmolVLA, GR00T)
_concept_ablation_scene_cache: Dict[str, dict] = {}


@scene_state_bp.route('/api/vla/action_trajectories/files', methods=['GET'])
def get_action_trajectory_files():
    """List available OFT ablation files for action trajectory comparison."""
    if not OFT_ABLATION_DIR.exists():
        return jsonify({'files': [], 'error': 'Ablation directory not found'}), 404

    files = []
    for f in sorted(OFT_ABLATION_DIR.glob('ablation_L*.json')):
        name = f.stem  # e.g., ablation_L04_libero_goal
        parts = name.split('_', 2)  # ['ablation', 'L04', 'libero_goal']
        if len(parts) >= 3:
            layer = parts[1]
            suite = parts[2]
            files.append({
                'filename': f.name,
                'layer': layer,
                'suite': suite,
                'label': f"Layer {layer} - {suite}",
            })

    return jsonify({'files': files, 'total': len(files)})


@scene_state_bp.route('/api/vla/action_trajectories', methods=['GET'])
def get_action_trajectories():
    """Get OFT action trajectory data for baseline vs ablated comparison."""
    layer = request.args.get('layer', 'L04')
    suite = request.args.get('suite', 'libero_goal')
    concept = request.args.get('concept', '')
    task_id = request.args.get('task_id', '0')
    episode = int(request.args.get('episode', 0))

    data = _load_oft_ablation(layer, suite)
    if data is None:
        return jsonify({'error': f'No ablation data for {layer}_{suite}'}), 404

    tasks = data.get('tasks', {})
    baseline = data.get('baseline', {})

    # If no concept specified, list available concepts
    if not concept:
        available = []
        for c_name, c_data in tasks.items():
            concept_tasks = list(c_data.get('tasks', {}).keys())
            available.append({
                'concept': c_name,
                'n_tasks': len(concept_tasks),
                'tasks': concept_tasks,
            })
        return jsonify({
            'layer': layer,
            'suite': suite,
            'concepts': available,
            'baseline_info': {
                'n_episodes': baseline.get('n_episodes', 0),
                'success_rate': baseline.get('success_rate', 0),
            },
        })

    # Get specific concept data
    if concept not in tasks:
        return jsonify({'error': f'Concept {concept} not found'}), 404

    concept_data = tasks[concept]
    concept_tasks = concept_data.get('tasks', {})

    if task_id not in concept_tasks:
        return jsonify({'error': f'Task {task_id} not found for concept {concept}'}), 404

    task_data = concept_tasks[task_id]
    actions = task_data.get('actions', [])

    if episode >= len(actions):
        episode = 0

    # Get the ablated trajectory
    ablated_traj = actions[episode] if actions else []

    # Get baseline trajectory from baseline data
    # Baseline structure varies:
    #   - baseline.tasks.{task_id}.actions  (if tasks sub-key exists)
    #   - baseline.{task_id}.actions        (if task IDs are direct keys)
    #   - may not have actions at all (only success_rate/successes)
    baseline_traj = []
    if 'tasks' in baseline:
        baseline_actions = baseline['tasks'].get(task_id, {}).get('actions', [])
        baseline_traj = baseline_actions[episode] if episode < len(baseline_actions) else []
    elif task_id in baseline:
        task_baseline = baseline[task_id]
        if isinstance(task_baseline, dict) and 'actions' in task_baseline:
            baseline_actions = task_baseline['actions']
            baseline_traj = baseline_actions[episode] if episode < len(baseline_actions) else []

    # Subsample for frontend performance
    max_steps = 300
    if len(ablated_traj) > max_steps:
        step = len(ablated_traj) / max_steps
        ablated_traj = [ablated_traj[int(i * step)] for i in range(max_steps)]
    if len(baseline_traj) > max_steps:
        step = len(baseline_traj) / max_steps
        baseline_traj = [baseline_traj[int(i * step)] for i in range(max_steps)]

    # Baseline success rate for this task
    baseline_sr = 0
    if task_id in baseline:
        baseline_sr = baseline[task_id].get('success_rate', 0) if isinstance(baseline[task_id], dict) else 0

    # Action dimension labels
    dim_labels = ['dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'gripper']

    return jsonify({
        'layer': layer,
        'suite': suite,
        'concept': concept,
        'task_id': task_id,
        'episode': episode,
        'n_episodes': len(actions),
        'success_rate': task_data.get('success_rate', 0),
        'baseline_success_rate': baseline_sr,
        'delta': task_data.get('delta', 0),
        'baseline_trajectory': baseline_traj,
        'ablated_trajectory': ablated_traj,
        'dim_labels': dim_labels,
        'baseline_steps': len(baseline_traj),
        'ablated_steps': len(ablated_traj),
        'baseline_available': len(baseline_traj) > 0,
    })
# Experiment Results API (SmolVLA, X-VLA, GR00T)
# Cache for experiment results (loaded once on first request)
_experiment_results_cache: Dict = {}


