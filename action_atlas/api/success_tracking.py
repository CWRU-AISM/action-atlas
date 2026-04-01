"""Success tracking and video-result matching."""
from .helpers import *
from .data_loaders import *



def _build_oft_ablation_success_map() -> Dict:
    """
    Build a mapping from (suite, layer, concept_type, concept_name, task, episode) -> success
    by reading all OFT concept ablation JSON files.
    """
    global _oft_ablation_success_map
    if _oft_ablation_success_map is not None:
        return _oft_ablation_success_map

    success_map = {}
    oft_dir = OFT_ABLATION_VIDEO_DIR.parent  # results/experiment_results/oft_concept_ablation/
    if not oft_dir.exists():
        _oft_ablation_success_map = success_map
        return success_map

    for json_path in sorted(oft_dir.glob("ablation_*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        layer = data.get('layer')
        suite_short = data.get('suite', '')
        full_suite = suite_short if suite_short.startswith('libero_') else f'libero_{suite_short}'
        tasks_data = data.get('tasks', {})

        for concept_key, concept_data in tasks_data.items():
            if '/' not in concept_key:
                continue
            concept_type, concept_name = concept_key.split('/', 1)
            task_results = concept_data.get('tasks', {})

            for task_id, task_data in task_results.items():
                successes = task_data.get('successes', [])
                for ep_idx, s in enumerate(successes):
                    key = (full_suite, layer, concept_type, concept_name, int(task_id), ep_idx)
                    success_map[key] = s

    _oft_ablation_success_map = success_map
    print(f"Built OFT ablation success map: {len(success_map)} entries, {sum(1 for v in success_map.values() if v)} success")
    return success_map


def _build_pi05_ablation_success_map() -> Dict:
    """
    Build a mapping from (suite, layer, concept_type, concept_name, task) -> success
    by reading all Pi0.5 concept ablation JSON files.

    Suite names in the map use the directory format (e.g., 'libero_goal').
    """
    global _pi05_ablation_success_map
    if _pi05_ablation_success_map is not None:
        return _pi05_ablation_success_map

    success_map = {}

    if not PI05_CONCEPT_ABLATION_DIR.exists():
        _pi05_ablation_success_map = success_map
        return success_map

    suite_name_map = {
        'goal': 'libero_goal',
        'object': 'libero_object',
        'spatial': 'libero_spatial',
        '10': 'libero_10',
    }

    for json_path in sorted(PI05_CONCEPT_ABLATION_DIR.glob("ablation_*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        layer = data.get('layer')
        suite_short = data.get('suite', '')
        full_suite = suite_name_map.get(suite_short, f'libero_{suite_short}')
        tasks_data = data.get('tasks', {})

        for concept_key, concept_data in tasks_data.items():
            if '/' not in concept_key:
                continue
            concept_type, concept_name = concept_key.split('/', 1)
            task_results = concept_data.get('tasks', {})

            for task_id, task_data in task_results.items():
                successes = task_data.get('successes', [])
                if successes:
                    key = (full_suite, layer, concept_type, concept_name, int(task_id))
                    success_map[key] = successes[0]

    _pi05_ablation_success_map = success_map
    return success_map


def _video_stem_from_condition(task_key: str, cond_key: str, experiment_type: str) -> List[str]:
    """
    Generate possible video filename stems from a results.json task/condition entry.

    Returns a list of candidate stems (without .mp4) that might match.
    Handles naming convention differences between results.json keys and video filenames:
      - results: baseline_a  -> video: baseline_A (case)
      - results: injection_b_into_a -> video: inject_B_into_A (naming + case)
      - results: null_layer_0 -> video: null_L0 (abbreviation)
    """
    candidates = []

    if task_key.startswith('pair_'):
        # cross_task_injection: pair_0_1 + condition
        # Add both lowercase and uppercase variants
        candidates.append(f"{task_key}_{cond_key}")
        # Handle baseline_a -> baseline_A, baseline_b -> baseline_B
        upper_cond = cond_key.replace('_a', '_A').replace('_b', '_B')
        if upper_cond != cond_key:
            candidates.append(f"{task_key}_{upper_cond}")
        # Handle injection_b_into_a -> inject_B_into_A
        if cond_key.startswith('injection_'):
            inject_cond = cond_key.replace('injection_', 'inject_')
            candidates.append(f"{task_key}_{inject_cond}")
            inject_upper = inject_cond.replace('_a', '_A').replace('_b', '_B')
            candidates.append(f"{task_key}_{inject_upper}")
        return candidates

    # Standard task-based experiments: task_0 -> task0
    task_num = task_key.replace('task_', '')
    cond = cond_key

    # Primary candidate: task{N}_{condition}
    candidates.append(f"task{task_num}_{cond}")
    # Also try task_{N}_{condition}
    candidates.append(f"task_{task_num}_{cond}")

    # Handle null_layer_N -> null_LN mapping
    if 'null_layer_' in cond:
        layer_num = cond.replace('null_layer_', '')
        candidates.append(f"task{task_num}_null_L{layer_num}")

    # Handle trial patterns for same_scene_injection
    # results: "trial_1_with_trial_0_activations" -> video: "task0_trial1_inject"
    if cond.startswith('trial_') and '_with_' in cond:
        trial_num = cond.split('_')[1]
        candidates.append(f"task{task_num}_trial{trial_num}_inject")
    elif cond.startswith('trial_') and 'baseline' in cond:
        trial_num = cond.split('_')[1]
        candidates.append(f"task{task_num}_trial{trial_num}_baseline")
    elif cond.startswith('trial_'):
        trial_num = cond.split('_')[1]
        candidates.append(f"task{task_num}_trial{trial_num}")

    # Handle inject_ variants in non-pair experiments
    if cond.startswith('inject_'):
        upper_cond = cond.replace('_generic', '_generic').replace('_opposite', '_opposite')
        candidates.append(f"task{task_num}_{upper_cond}")

    return candidates


def _build_bulk_success_map(model: str) -> Dict[str, Optional[bool]]:
    """
    Build a mapping from video_path_key -> success by scanning all results.json files.

    The key format is the relative path of the experiment directory (from the rollouts base)
    joined with the video filename stem. This enables O(1) lookups per video without
    requiring the video file to exist on disk.
    """
    import logging
    logger = logging.getLogger(__name__)

    success_map: Dict[str, Optional[bool]] = {}

    if model == 'pi05':
        base_dirs = [PI05_ROLLOUTS_DIR]
    elif model == 'openvla':
        base_dirs = [OPENVLA_ROLLOUTS_DIR]
    elif model == 'xvla':
        base_dirs = [XVLA_ROLLOUTS_DIR]
    elif model == 'smolvla':
        base_dirs = [SMOLVLA_ROLLOUTS_DIR]
    elif model == 'groot':
        base_dirs = [GROOT_ROLLOUTS_DIR, GROOT_ROLLOUTS_DIR_BATCH2]
    else:
        return success_map

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        results_files = list(base_dir.rglob('results.json'))
        logger.info(f"Building bulk success map for {model}: found {len(results_files)} results.json files under {base_dir}")

        for rf in results_files:
            try:
                with open(rf) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Skip non-dict results (some Pi0.5 files contain a list)
            if not isinstance(data, dict):
                continue

            # Get the relative dir path from rollouts base to the results.json parent
            rel_dir = str(rf.parent.relative_to(base_dir))
            experiment_type = data.get('experiment', '')
            conditions = data.get('conditions', {})

            # Also handle pi05 format with 'results' key
            results_data = data.get('results', {})

            # Process conditions (OpenVLA format)
            for top_key, top_data in conditions.items():
                if not isinstance(top_data, dict):
                    continue

                for cond_key, cond_data in top_data.items():
                    if isinstance(cond_data, dict) and 'success' in cond_data:
                        # Direct condition with success
                        stems = _video_stem_from_condition(top_key, cond_key, experiment_type)
                        for stem in stems:
                            map_key = f"{rel_dir}/{stem}"
                            success_map[map_key] = cond_data['success']
                    elif isinstance(cond_data, dict):
                        # Nested structure (e.g., cross_task injection -> pair -> injection -> layer -> success)
                        for nested_key, nested_data in cond_data.items():
                            if isinstance(nested_data, dict) and 'success' in nested_data:
                                # Generate multiple stem variants for naming/case differences
                                # e.g. pair_0_1 + injection_b_into_a + layer_0
                                base_stem = f"{top_key}_{cond_key}_{nested_key}"
                                stems_to_add = [base_stem]
                                # injection_ -> inject_, lowercase -> uppercase
                                alt = base_stem.replace('injection_', 'inject_')
                                if alt != base_stem:
                                    stems_to_add.append(alt)
                                alt_upper = alt.replace('_a_', '_A_').replace('_b_', '_B_')
                                # Handle trailing _a or _b before _layer
                                alt_upper = alt_upper.replace('_a_layer', '_A_layer').replace('_b_layer', '_B_layer')
                                alt_upper = alt_upper.replace('into_a_', 'into_A_').replace('into_b_', 'into_B_')
                                if alt_upper not in stems_to_add:
                                    stems_to_add.append(alt_upper)
                                for stem in stems_to_add:
                                    map_key = f"{rel_dir}/{stem}"
                                    success_map[map_key] = nested_data['success']

            # Process results (Pi0.5 format)
            for pair_key, pair_data in results_data.items():
                if not isinstance(pair_data, dict):
                    continue

                for entry_key, entry_data in pair_data.items():
                    if isinstance(entry_data, dict):
                        if 'success' in entry_data:
                            map_key = f"{rel_dir}/{pair_key}_{entry_key}"
                            success_map[map_key] = entry_data['success']
                        else:
                            for vid_key, vid_data in entry_data.items():
                                if isinstance(vid_data, dict) and 'success' in vid_data:
                                    map_key = f"{rel_dir}/{vid_key}"
                                    success_map[map_key] = vid_data['success']

    logger.info(f"Bulk success map for {model}: {len(success_map)} entries")
    return success_map


def get_bulk_success(video_path: str, model: str) -> Optional[bool]:
    """
    Look up success/failure from the pre-built bulk success map.

    Args:
        video_path: Relative video path from the index (e.g.,
            "openvla_oft/libero_10/20260130_193228/null_injection/task0_null_L0.mp4")
        model: 'pi05' or 'openvla'

    Returns:
        True/False if found, None otherwise
    """
    global _bulk_success_map, _bulk_success_map_built

    # Build map lazily on first use
    if not _bulk_success_map_built.get(model, False):
        _bulk_success_map[model] = _build_bulk_success_map(model)
        _bulk_success_map_built[model] = True

    model_map = _bulk_success_map.get(model, {})
    if not model_map:
        return None

    # Strip model prefix if present (e.g., "openvla/" prefix added by the API)
    path = video_path
    if path.startswith(f'{model}/'):
        path = path[len(model) + 1:]

    # Get directory and stem
    p = Path(path)
    stem = p.stem
    parent = str(p.parent)

    # Try direct lookup: dir/stem
    key = f"{parent}/{stem}"
    if key in model_map:
        return model_map[key]

    # Try with rollouts/ prefix stripped (symlink target)
    if parent.startswith('rollouts/'):
        stripped_parent = parent[len('rollouts/'):]
        key = f"{stripped_parent}/{stem}"
        if key in model_map:
            return model_map[key]

    # Try stripping openvla_oft/ prefix (video index includes model subdir but
    # success map keys are relative to /data/openvla_rollouts which doesn't have it)
    if parent.startswith('openvla_oft/'):
        stripped_parent = parent[len('openvla_oft/'):]
        key = f"{stripped_parent}/{stem}"
        if key in model_map:
            return model_map[key]

    return None


def get_success_from_results_json(video_path: str, model: str = 'pi05') -> Optional[bool]:
    """
    Look up success/failure status from results.json files.

    For pi05 videos:
        - Searches /data/robotsteering/pi05_rollouts/*/results.json
        - Results structure: results -> pair_X_Y -> inject_X_into_Y -> {video_name} -> success
        - Also handles: results -> pair_X_Y -> baseline_task_N -> success

    For openvla videos:
        - Searches /data/openvla_rollouts/*/results.json
        - Results structure: conditions -> task_N -> {condition} -> success

    Args:
        video_path: Path to the video file (can be relative or absolute)
        model: 'pi05' or 'openvla'

    Returns:
        True for success, False for failure, None if not found
    """
    global _results_json_cache

    video_path = Path(video_path)
    filename = video_path.stem

    # Determine the base directories for each model
    if model == 'pi05':
        base_dirs = [
            PI05_ROLLOUTS_DIR,
            VLA_VIDEO_DIR / "pi05",
        ]
    elif model == 'openvla':
        base_dirs = [
            OPENVLA_ROLLOUTS_DIR,
            VLA_VIDEO_DIR / "openvla",
        ]
    elif model == 'xvla':
        base_dirs = [
            XVLA_ROLLOUTS_DIR,
            VLA_VIDEO_DIR / "xvla",
        ]
    elif model == 'smolvla':
        base_dirs = [
            SMOLVLA_ROLLOUTS_DIR,
            VLA_VIDEO_DIR / "smolvla",
        ]
    elif model == 'groot':
        base_dirs = [
            GROOT_ROLLOUTS_DIR,
            GROOT_ROLLOUTS_DIR_BATCH2,
            VLA_VIDEO_DIR / "groot",
        ]
    else:
        return None

    # Try to resolve the video path to find the experiment directory
    resolved_path = None
    for base in base_dirs:
        # Try direct path
        candidate = base / video_path
        if candidate.exists():
            resolved_path = candidate.resolve()
            break
        # Try just the filename parts that might be in the path
        # e.g., counterfactual/goal/libero_goal_xxx/wrist_videos/file.mp4
        for part in str(video_path).split('/'):
            if part and (base / part).exists():
                candidate = base / video_path
                if candidate.exists():
                    resolved_path = candidate.resolve()
                    break
        if resolved_path:
            break

    # If we still don't have a resolved path, try to work with path components
    path_str = str(video_path)

    # For pi05 cross_task experiments, parse the path to find batch/pair/inject structure
    # Example path: cross_task_goal/cross_task_libero_goal_xxx/batch_1/pair_0_1/inject_0_into_1/video.mp4
    if model == 'pi05' and 'cross_task' in path_str:
        # Find batch_N in the path
        parts = path_str.replace('\\', '/').split('/')
        batch_idx = None
        pair_name = None
        inject_name = None

        for i, part in enumerate(parts):
            if part.startswith('batch_'):
                batch_idx = i
            elif part.startswith('pair_'):
                pair_name = part
            elif part.startswith('inject_'):
                inject_name = part

        if batch_idx is not None and pair_name and inject_name:
            # Construct path to results.json
            batch_path = '/'.join(parts[:batch_idx + 1])

            for base in base_dirs:
                results_path = base / batch_path / "results.json"
                if results_path.exists():
                    # Cache the results file
                    cache_key = str(results_path)
                    if cache_key not in _results_json_cache:
                        try:
                            with open(results_path) as f:
                                _results_json_cache[cache_key] = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            continue

                    data = _results_json_cache.get(cache_key, {})
                    results = data.get('results', {})

                    if pair_name in results:
                        pair_data = results[pair_name]

                        # Check for baseline videos
                        if 'baseline' in filename:
                            # Check baseline_task_0 or baseline_task_1
                            for task_key in ['baseline_task_0', 'baseline_task_1']:
                                if task_key in pair_data:
                                    return pair_data[task_key].get('success')

                        # Check injection results
                        if inject_name in pair_data:
                            inject_data = pair_data[inject_name]
                            # The video filename should match experiment name
                            if filename in inject_data:
                                return inject_data[filename].get('success')

                    break

    # For pi05 baseline experiments outside cross_task
    if model == 'pi05' and 'baseline' in path_str:
        # Look for results.json in parent directories
        if resolved_path:
            current = resolved_path.parent
            for _ in range(5):
                results_path = current / "results.json"
                if results_path.exists():
                    cache_key = str(results_path)
                    if cache_key not in _results_json_cache:
                        try:
                            with open(results_path) as f:
                                _results_json_cache[cache_key] = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            break

                    data = _results_json_cache.get(cache_key, {})
                    results = data.get('results', {})

                    # Search through all pairs for baseline data
                    for pair_name, pair_data in results.items():
                        for task_key in ['baseline_task_0', 'baseline_task_1']:
                            if task_key in pair_data:
                                # Try to match by task number in filename
                                task_num = pair_data.get('task_a' if task_key == 'baseline_task_0' else 'task_b')
                                if f'_{task_num}_' in filename or f'task_{task_num}' in filename or filename.endswith(f'_{task_num}'):
                                    return pair_data[task_key].get('success')
                    break

                parent = current.parent
                if parent == current:
                    break
                current = parent

    # For openvla experiments
    if model == 'openvla':
        # Parse task and condition from path
        # Example: openvla_oft/libero_spatial/20260130_xxx/counterfactual_prompting/results.json
        # Videos might be in same dir or subdirs

        if resolved_path:
            current = resolved_path.parent
            for _ in range(5):
                results_path = current / "results.json"
                if results_path.exists():
                    cache_key = str(results_path)
                    if cache_key not in _results_json_cache:
                        try:
                            with open(results_path) as f:
                                _results_json_cache[cache_key] = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            break

                    data = _results_json_cache.get(cache_key, {})
                    conditions = data.get('conditions', {})

                    # Handle cross_task_injection format
                    # Filename: pair_0_1_baseline_A.mp4 or pair_0_1_inject_B_into_A_layer_0.mp4
                    if data.get('experiment') == 'cross_task_injection' or 'cross_task' in path_str:
                        # Parse pair from filename
                        import re
                        pair_match = re.search(r'pair_(\d+)_(\d+)', filename)
                        if pair_match:
                            pair_key = f"pair_{pair_match.group(1)}_{pair_match.group(2)}"
                            if pair_key in conditions:
                                pair_data = conditions[pair_key]

                                # Check for baseline videos
                                if 'baseline_a' in filename.lower() or 'baseline_A' in filename:
                                    return pair_data.get('baseline_a', {}).get('success')
                                elif 'baseline_b' in filename.lower() or 'baseline_B' in filename:
                                    return pair_data.get('baseline_b', {}).get('success')

                                # Check for injection videos
                                # Format: pair_0_1_inject_B_into_A_layer_0.mp4
                                inject_match = re.search(r'inject_[AB]_into_[AB]', filename)
                                layer_match = re.search(r'layer_(\d+)', filename)
                                if inject_match and layer_match:
                                    inject_key = inject_match.group(0).lower().replace('_a', '_a').replace('_b', '_b')
                                    # Convert inject_B_into_A to injection_b_into_a
                                    inject_key = 'injection_' + inject_key.replace('inject_', '')
                                    layer_key = f"layer_{layer_match.group(1)}"

                                    if inject_key in pair_data:
                                        layer_data = pair_data[inject_key].get(layer_key, {})
                                        return layer_data.get('success')
                        break

                    # Try to match task_N from filename or path
                    for task_key, task_data in conditions.items():
                        # task_key is like "task_0", "task_1", etc.
                        task_num = task_key.replace('task_', '')

                        if f'task_{task_num}' in filename or f'task{task_num}' in filename.lower():
                            # Found matching task, now find condition
                            for cond_name, cond_data in task_data.items():
                                if cond_name in filename or cond_name in path_str:
                                    return cond_data.get('success')
                            # If no condition match, return first condition's success
                            if task_data:
                                first_cond = next(iter(task_data.values()))
                                if isinstance(first_cond, dict):
                                    return first_cond.get('success')
                    break

                parent = current.parent
                if parent == current:
                    break
                current = parent

    # Fall back to searching for results.json in parent directories
    if resolved_path:
        current = resolved_path.parent
        for _ in range(5):
            results_path = current / "results.json"
            if results_path.exists():
                cache_key = str(results_path)
                if cache_key not in _results_json_cache:
                    try:
                        with open(results_path) as f:
                            _results_json_cache[cache_key] = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        break

                data = _results_json_cache.get(cache_key, {})

                # Try pi05 format (results -> pair -> inject -> experiment)
                if 'results' in data:
                    for pair_name, pair_data in data['results'].items():
                        if not isinstance(pair_data, dict):
                            continue
                        for key, value in pair_data.items():
                            if isinstance(value, dict):
                                if key.startswith('inject_') and filename in value:
                                    return value[filename].get('success')
                                elif key.startswith('baseline_') and 'success' in value:
                                    # Check if this baseline matches our video
                                    task_num = pair_data.get('task_a' if key == 'baseline_task_0' else 'task_b')
                                    if str(task_num) in filename:
                                        return value.get('success')

                # Try openvla format (conditions -> task_N -> condition -> success)
                if 'conditions' in data:
                    for task_key, task_data in data['conditions'].items():
                        if not isinstance(task_data, dict):
                            continue
                        for cond_name, cond_data in task_data.items():
                            if isinstance(cond_data, dict) and 'success' in cond_data:
                                # Check if path contains task and condition info
                                task_num = task_key.replace('task_', '')
                                if f'task_{task_num}' in path_str or f'task{task_num}' in path_str.lower():
                                    if cond_name in path_str or cond_name in filename:
                                        return cond_data.get('success')

                break

            parent = current.parent
            if parent == current:
                break
            current = parent

    return None


def enrich_video_with_success(video: Dict, model: str = 'pi05') -> Dict:
    """
    Enrich a video entry with success data from results.json if not already present.

    Uses bulk success map (fast O(1) lookup) first, then falls back to
    per-video results.json resolution if needed.

    Args:
        video: Video dict with 'path' key
        model: 'pi05' or 'openvla'

    Returns:
        Video dict with 'success' field populated if found
    """
    # If success is already known, return as-is
    if video.get('success') is not None:
        return video

    video_path = video.get('path', '')
    if not video_path:
        return video

    # Try bulk lookup first (fast, covers most cases)
    success = get_bulk_success(video_path, model)

    # Fall back to per-video resolution if bulk lookup missed
    if success is None:
        success = get_success_from_results_json(video_path, model)

    if success is not None:
        video = video.copy()  # Don't modify original
        video['success'] = success
        video['success_source'] = 'results.json'

    return video


def _load_layer_connections_openvla(suite: str) -> dict:
    """
    Load real layer connection data for OpenVLA-OFT (32 layers).

    Sources:
      - R2 per layer: OFT_PROBING_DIR / oft_multilayer_probing_{suite}.json
      - Concept counts per layer: OFT_CONCEPT_ID_DIR / oft_concept_id_layer{NN}_{suite}.json
    """
    import math as _math
    n_layers = 32
    score_threshold = 1.0

    # --- Load probing R2 per layer ---
    r2_per_layer = {}
    auc_per_layer = {}
    probing_file = OFT_PROBING_DIR / f"oft_multilayer_probing_{suite}.json"
    if probing_file.exists():
        with open(probing_file) as f:
            probing = json.load(f)
        for layer_str, layer_data in probing.get('layers', {}).items():
            idx = int(layer_str)
            r2_per_layer[idx] = layer_data.get('nsteps_r2', 0.0)
            auc_per_layer[idx] = layer_data.get('success_auc', 0.0)

    # --- Load concept counts per layer ---
    layers_data = []
    for layer_idx in range(n_layers):
        filename = f"oft_concept_id_layer{layer_idx:02d}_{suite}.json"
        filepath = OFT_CONCEPT_ID_DIR / filename

        motion_count = 0
        object_count = 0
        spatial_count = 0
        total_count = 0
        concepts_detail = []

        if filepath.exists():
            with open(filepath) as f:
                cdata = json.load(f)
            for concept_key, concept_info in cdata.get('concepts', {}).items():
                n_sig = sum(1 for s in concept_info.get('top_scores', []) if s > score_threshold)
                ctype = concept_key.split('/')[0]
                cname = concept_key.split('/')[-1] if '/' in concept_key else concept_key
                if ctype == 'motion':
                    motion_count += n_sig
                elif ctype == 'object':
                    object_count += n_sig
                elif ctype == 'spatial':
                    spatial_count += n_sig
                total_count += n_sig
                if n_sig > 0:
                    concepts_detail.append({
                        'concept': concept_key,
                        'type': ctype,
                        'name': cname,
                        'n_significant': n_sig,
                        'max_score': max(concept_info.get('top_scores', [0.0])),
                        'max_cohens_d': concept_info.get('max_cohens_d', 0.0),
                    })

        concepts_detail.sort(key=lambda c: c['n_significant'], reverse=True)
        type_counts = {'motion': motion_count, 'object': object_count, 'spatial': spatial_count}
        dominant_type = max(type_counts, key=type_counts.get) if total_count > 0 else 'none'

        layers_data.append({
            'layer': layer_idx,
            'id': f'layer_{layer_idx}',
            'r2': r2_per_layer.get(layer_idx, 0.0),
            'success_auc': auc_per_layer.get(layer_idx, 0.0),
            'feature_count': total_count,
            'motion_features': motion_count,
            'object_features': object_count,
            'spatial_features': spatial_count,
            'dominant_type': dominant_type,
            'top_concepts': concepts_detail[:5],
        })

    # --- Build connections ---
    connections = []
    for i in range(n_layers - 1):
        target_r2 = layers_data[i + 1]['r2']
        source_r2 = layers_data[i]['r2']
        strength = target_r2
        delta_r2 = target_r2 - source_r2
        connections.append({
            'source': i,
            'target': i + 1,
            'strength': round(strength, 4),
            'delta_r2': round(delta_r2, 4),
            'type': 'sequential',
        })

    # Skip connections where there are large R2 jumps
    for i in range(n_layers):
        for stride in [4, 8]:
            j = i + stride
            if j < n_layers:
                r2_i = layers_data[i]['r2']
                r2_j = layers_data[j]['r2']
                delta = r2_j - r2_i
                if delta > 0.02:
                    connections.append({
                        'source': i,
                        'target': j,
                        'strength': round(r2_j, 4),
                        'delta_r2': round(delta, 4),
                        'type': 'skip',
                    })

    return {
        'model': 'openvla_oft',
        'suite': suite,
        'n_layers': n_layers,
        'layers': layers_data,
        'connections': connections,
    }


def _load_layer_connections_pi05(pathway: str = 'expert') -> dict:
    """
    Load real layer connection data for Pi0.5 (18 layers per pathway).

    Merges concept feature counts from concept_features.json into each layer.

    Args:
        pathway: 'expert' for action_expert layers, 'paligemma' for VLM layers.

    Source: PI05_PROBES_DIR / gemma_expert_probes.json (R2 data)
            data/concept_features.json (concept counts)
    """
    import math as _math
    n_layers = 18

    # Determine layer key prefix and display name based on pathway
    if pathway == 'paligemma':
        layer_prefix = 'paligemma_layer_'
        layer_display = 'PaliGemma Layer'
        suite_name = 'paligemma_pathway'
    else:
        layer_prefix = 'action_expert_layer_'
        layer_display = 'Gemma Expert Layer'
        suite_name = 'expert_pathway'

    # Load R2 data from probes (only available for expert pathway currently)
    probes_file = PI05_PROBES_DIR / "gemma_expert_probes.json"
    overall_r2 = {}
    if pathway == 'expert' and probes_file.exists():
        with open(probes_file) as f:
            probes = json.load(f)
        for dim, dim_data in probes.get('results', {}).items():
            overall_r2[dim] = dim_data.get('test_r2', 0.0)

    mean_r2 = sum(overall_r2.values()) / max(len(overall_r2), 1) if overall_r2 else 0.0

    # Load concept features data
    concept_data = load_concept_features()

    layers_data = []
    for i in range(n_layers):
        frac = i / max(n_layers - 1, 1)
        layer_r2 = mean_r2 * (0.6 + 0.4 / (1 + _math.exp(-6 * (frac - 0.5))))

        layer_id = f'{layer_prefix}{i}'

        # Build base layer entry
        layer_entry = {
            'layer': i,
            'id': layer_id,
            'name': f'{layer_display} {i}',
            'r2': round(layer_r2, 4),
            'success_auc': 0.0,
            'feature_count': 0,
            'motion_features': 0,
            'object_features': 0,
            'spatial_features': 0,
            'dominant_type': 'none',
            'top_concepts': [],
        }

        # Merge real concept counts from concept_features.json
        if concept_data and layer_id in concept_data:
            concept_counts = _extract_concept_counts_from_layer(concept_data[layer_id])
            layer_entry.update(concept_counts)

        if mean_r2 == 0.0:
            layer_entry['note'] = 'Per-layer probing not yet run; R2 unavailable'

        layers_data.append(layer_entry)

    connections = []
    for i in range(n_layers - 1):
        target_r2 = layers_data[i + 1]['r2']
        source_r2 = layers_data[i]['r2']
        delta_r2 = target_r2 - source_r2
        connections.append({
            'source': i,
            'target': i + 1,
            'strength': round(target_r2, 4),
            'delta_r2': round(delta_r2, 4),
            'type': 'sequential',
        })

    result = {
        'model': 'pi05',
        'suite': suite_name,
        'pathway': pathway,
        'n_layers': n_layers,
        'layers': layers_data,
        'connections': connections,
    }
    if overall_r2:
        result['aggregate_r2'] = overall_r2
    return result


def register_vla_routes(app):
    """Register VLA routes with Flask app."""
    app.register_blueprint(vla_bp)
    print("Registered Action Atlas routes")


