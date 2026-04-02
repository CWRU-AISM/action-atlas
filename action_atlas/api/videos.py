"""Action Atlas API - videos routes."""
from flask import Blueprint, request, jsonify, send_file, abort, make_response, redirect
from .helpers import *
from .data_loaders import *
from .success_tracking import *

videos_bp = Blueprint("videos", __name__)


def load_video_index(model: str) -> Optional[Dict]:
    """Load video index for a specific model."""
    video_index_path = Path(__file__).parent / "data" / "videos" / model / "index.json"
    if not video_index_path.exists():
        return None
    with open(video_index_path) as f:
        return json.load(f)


@videos_bp.route('/api/vla/videos', methods=['GET'])
def get_vla_videos():
    """
    Get list of videos with paths and metadata.

    Query params:
        model: pi05 or openvla (default: pi05)
        experiment_type: Filter by experiment type (e.g., counterfactual, ablation)
        suite: Filter by task suite (e.g., libero_10, goal)
        success: Filter by success status (true/false)
        limit: Maximum number of videos to return (default: unlimited)

    Returns list of videos with paths and metadata.
    """
    model = request.args.get('model', 'pi05')
    experiment_type = request.args.get('experiment_type')
    suite = request.args.get('suite')
    success_param = request.args.get('success')
    limit_param = request.args.get('limit')

    # Parse success parameter
    success_filter = None
    if success_param is not None:
        success_filter = success_param.lower() in ('true', '1', 'yes')

    # Try to load from index.json first
    video_index = load_video_index(model)

    if video_index is not None:
        videos = video_index.get('videos', [])

        # Append concept ablation videos (from results/ directory or baked index)
        has_ablation_in_index = any(v.get('experiment_type') == 'concept_ablation' for v in videos)

        # OFT concept ablation: append from directory or baked index
        if model in ('openvla', 'openvla_oft') and not has_ablation_in_index:
            oft_success_map = _build_oft_ablation_success_map()
            if OFT_ABLATION_VIDEO_DIR.exists():
                for suite_path in sorted(OFT_ABLATION_VIDEO_DIR.iterdir()):
                    if not suite_path.is_dir():
                        continue
                    suite_name = suite_path.name
                    for vp in sorted(suite_path.glob("*.mp4")):
                        fn = vp.stem
                        parts = fn.split("_")
                        layer_num = None
                        concept_type = None
                        concept_name = None
                        task_num = None
                        ep_num = None
                        if len(parts) >= 6 and parts[0] == 'ablation':
                            if parts[1].startswith('L'):
                                try:
                                    layer_num = int(parts[1][1:])
                                except ValueError:
                                    pass
                            concept_type = parts[2]
                            task_idx = None
                            for i, p in enumerate(parts):
                                if p.startswith('task'):
                                    task_idx = i
                                    try:
                                        task_num = int(p.replace('task', ''))
                                    except ValueError:
                                        pass
                                elif p.startswith('ep'):
                                    try:
                                        ep_num = int(p.replace('ep', ''))
                                    except ValueError:
                                        pass
                            if task_idx and task_idx > 3:
                                concept_name = '_'.join(parts[3:task_idx])
                        success_key = (suite_name, layer_num, concept_type, concept_name, task_num, ep_num)
                        videos.append({
                            'path': f'oft_ablation/{suite_name}/{vp.name}',
                            'experiment_type': 'concept_ablation',
                            'suite': suite_name,
                            'subtype': 'ablation',
                            'concept': f"{concept_type}/{concept_name}" if concept_type and concept_name else fn,
                            'concept_type': concept_type,
                            'layer': layer_num,
                            'task': task_num,
                            'episode': ep_num,
                            'success': oft_success_map.get(success_key),
                            'model': 'openvla',
                        })
            else:
                baked_index = Path(__file__).parent / "data" / "oft_ablation_index.json"
                if baked_index.exists():
                    with open(baked_index) as f:
                        idx_data = json.load(f)
                    videos.extend(idx_data.get('videos', []))

        # Pi0.5 concept ablation: append from directory or baked index
        if model == 'pi05' and not has_ablation_in_index:
            if PI05_ABLATION_VIDEO_DIR.exists():
                pi05_success_map = _build_pi05_ablation_success_map()
                for suite_path in sorted(PI05_ABLATION_VIDEO_DIR.iterdir()):
                    if not suite_path.is_dir():
                        continue
                    suite_name = suite_path.name
                    for vp in sorted(suite_path.glob("*.mp4")):
                        fn = vp.stem
                        parts = fn.split("_")
                        layer_num = None
                        concept_type = None
                        concept_name = None
                        task_num = None
                        ep_num = None
                        if len(parts) >= 6 and parts[0] == 'ablation':
                            if parts[1].startswith('L'):
                                try:
                                    layer_num = int(parts[1][1:])
                                except ValueError:
                                    pass
                            concept_type = parts[2]
                            task_idx = None
                            for i, p in enumerate(parts):
                                if p.startswith('task'):
                                    task_idx = i
                                    try:
                                        task_num = int(p.replace('task', ''))
                                    except ValueError:
                                        pass
                                elif p.startswith('ep'):
                                    try:
                                        ep_num = int(p.replace('ep', ''))
                                    except ValueError:
                                        pass
                            if task_idx and task_idx > 3:
                                concept_name = '_'.join(parts[3:task_idx])
                        full_concept = f"{concept_type}/{concept_name}" if concept_type and concept_name else fn
                        success_key = (suite_name, layer_num, concept_type, concept_name, task_num)
                        success_val = pi05_success_map.get(success_key)
                        videos.append({
                            'path': f'pi05_ablation/{suite_name}/{vp.name}',
                            'experiment_type': 'concept_ablation',
                            'suite': suite_name,
                            'subtype': 'ablation',
                            'concept': full_concept,
                            'task': task_num,
                            'layer': layer_num,
                            'success': success_val,
                            'model': 'pi05',
                        })
            else:
                baked_index = Path(__file__).parent / "data" / "pi05_ablation_index.json"
                if baked_index.exists():
                    with open(baked_index) as f:
                        idx_data = json.load(f)
                    videos.extend(idx_data.get('videos', []))

        # Normalize suite names: "goal" → "libero_goal", etc.
        _suite_normalize = {'goal': 'libero_goal', 'object': 'libero_object', 'spatial': 'libero_spatial', '10': 'libero_10'}
        for v in videos:
            s = v.get('suite', '')
            if s in _suite_normalize:
                v['suite'] = _suite_normalize[s]

        # Apply experiment_type and suite filters first for performance
        if experiment_type:
            videos = [v for v in videos if v.get('experiment_type') == experiment_type]
        if suite:
            videos = [v for v in videos if v.get('suite') == suite]

        # Always enrich with success data from bulk results.json lookup.
        # The bulk lookup is O(1) per video (pre-built hash map), so this
        # is fast even for 19K+ videos.
        enriched_videos = []
        for v in videos:
            enriched = enrich_video_with_success(v, model)
            enriched_videos.append(enriched)
        videos = enriched_videos

        # Apply success filter if requested
        if success_filter is not None:
            videos = [v for v in videos if v.get('success') == success_filter]

        # Get unique values for filters
        all_videos = video_index.get('videos', [])
        experiment_types = list(set(v.get('experiment_type', '') for v in all_videos if v.get('experiment_type')))
        suites = list(set(v.get('suite', '') for v in all_videos if v.get('suite')))
        concepts = list(set(v.get('concept', '') for v in all_videos if v.get('concept')))

        # Transform video paths to include model prefix for the video serving endpoint
        # The paths in index.json are relative to the rollouts symlink
        # Skip paths that already have a special prefix (e.g., pi05_ablation/, oft_ablation/)
        for v in videos:
            path = v.get('path', '')
            if path and not path.startswith(f'{model}/') and not path.startswith(('pi05_ablation/', 'oft_ablation/')):
                v['path'] = f'{model}/{path}'

        # Apply limit if specified (after all filtering and transformations)
        total_before_limit = len(videos)
        if limit_param is not None:
            try:
                limit_val = int(limit_param)
                if limit_val > 0:
                    videos = videos[:limit_val]
            except (ValueError, TypeError):
                pass

        return jsonify({
            'videos': videos,
            'total': total_before_limit,
            'filters': {
                'models': [model],
                'experiment_types': sorted(experiment_types),
                'suites': sorted(suites),
                'concepts': sorted(concepts)
            }
        })

    # For new models (xvla, smolvla, groot, act): scan local rollout dirs + baked index
    if model in ('xvla', 'smolvla', 'groot', 'act'):
        import re as _re
        videos = []
        _scan_patterns = {
            'xvla': [
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/baselines', 'baseline', None),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/counterfactual_libero_goal_v2', 'counterfactual', 'libero_goal'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/counterfactual_libero_object_v2', 'counterfactual', 'libero_object'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/counterfactual_libero_spatial_v2', 'counterfactual', 'libero_spatial'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/counterfactual_libero_10', 'counterfactual', 'libero_10'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/cross_task_libero_goal', 'cross_task', 'libero_goal'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/cross_task_libero_object', 'cross_task', 'libero_object'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/cross_task_libero_spatial', 'cross_task', 'libero_spatial'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/cross_task_libero_10', 'cross_task', 'libero_10'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/grid_ablation_libero_goal', 'grid_ablation', 'libero_goal'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/grid_ablation_libero_object', 'grid_ablation', 'libero_object'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/grid_ablation_libero_spatial', 'grid_ablation', 'libero_spatial'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/grid_ablation_libero_10', 'grid_ablation', 'libero_10'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/vision_libero_goal', 'vision_perturbation', 'libero_goal'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/vision_libero_object', 'vision_perturbation', 'libero_object'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/vision_libero_spatial', 'vision_perturbation', 'libero_spatial'),
                (XVLA_ROLLOUTS_DIR, 'xvla_libero/experiments/vision_libero_10', 'vision_perturbation', 'libero_10'),
                (XVLA_ROLLOUTS_DIR, 'xvla_concept_ablation', 'concept_ablation', None),
                (XVLA_ROLLOUTS_DIR, 'xvla_concept_steering', 'steering', None),
            ],
            'smolvla': [
                # LIBERO (batch 1)
                (SMOLVLA_LIBERO_DIR, 'baselines', 'baseline', None),
                (SMOLVLA_LIBERO_DIR, 'counterfactual', 'counterfactual', None),
                (SMOLVLA_LIBERO_DIR, 'cross_task', 'cross_task', None),
                (SMOLVLA_LIBERO_DIR, 'grid_ablation', 'grid_ablation', None),
                (SMOLVLA_LIBERO_DIR, 'concept_ablation', 'concept_ablation', None),
                # MetaWorld (batch 2)
                (SMOLVLA_ROLLOUTS_DIR, 'metaworld_baseline', 'baseline', None),
                (SMOLVLA_ROLLOUTS_DIR, 'metaworld_cross_task', 'cross_task', None),
                (SMOLVLA_ROLLOUTS_DIR, 'metaworld_grid_ablation', 'grid_ablation', None),
                (SMOLVLA_ROLLOUTS_DIR, 'metaworld_vision_perturbation', 'vision_perturbation', None),
                (SMOLVLA_ROLLOUTS_DIR, 'metaworld_counterfactual_v2_easy', 'counterfactual', None),
                (SMOLVLA_ROLLOUTS_DIR, 'metaworld_counterfactual_v2_medium', 'counterfactual', None),
            ],
            'groot': [
                (GROOT_ROLLOUTS_DIR, 'libero_goal/baselines', 'baseline', 'libero_goal'),
                (GROOT_ROLLOUTS_DIR, 'libero_object/baselines', 'baseline', 'libero_object'),
                (GROOT_ROLLOUTS_DIR, 'libero_long/baselines', 'baseline', 'libero_long'),
                (GROOT_ROLLOUTS_DIR, 'libero_goal/counterfactual', 'counterfactual', 'libero_goal'),
                (GROOT_ROLLOUTS_DIR, 'libero_object/counterfactual', 'counterfactual', 'libero_object'),
                (GROOT_ROLLOUTS_DIR, 'libero_long/counterfactual', 'counterfactual', 'libero_long'),
                (GROOT_ROLLOUTS_DIR, 'libero_goal/visual_perturbation', 'vision_perturbation', 'libero_goal'),
                (GROOT_ROLLOUTS_DIR, 'libero_object/visual_perturbation', 'vision_perturbation', 'libero_object'),
                (GROOT_ROLLOUTS_DIR, 'libero_long/visual_perturbation', 'vision_perturbation', 'libero_long'),
                (GROOT_ROLLOUTS_DIR, 'sae_feature_ablation', 'concept_ablation', None),
                (GROOT_ROLLOUTS_DIR, 'sae_fraction_to_failure', 'fraction_to_failure', None),
                (GROOT_ROLLOUTS_DIR, 'sae_temporal_ablation', 'temporal_ablation', None),
                (GROOT_ROLLOUTS_DIR_BATCH2, 'sae_steering', 'steering', None),
                (GROOT_ROLLOUTS_DIR_BATCH2, 'sae_fraction_to_failure', 'fraction_to_failure', None),
                (GROOT_ROLLOUTS_DIR_BATCH2, 'sae_temporal_ablation', 'temporal_ablation', None),
                (GROOT_ROLLOUTS_DIR_BATCH2, 'sae_cross_suite_ablation', 'cross_suite_ablation', None),
            ],
            'act': [],
        }
        MAX_LOCAL_SCAN = 50000
        MAX_PER_SOURCE = 5000  # Prevent any single source from monopolizing
        local_count = 0
        for base_dir, subpath, exp_override, fixed_suite in _scan_patterns.get(model, []):
            if local_count >= MAX_LOCAL_SCAN:
                break
            scan_dir = base_dir / subpath
            if not scan_dir.exists():
                continue
            source_count = 0
            for vp in sorted(scan_dir.rglob("*.mp4")):
                if local_count >= MAX_LOCAL_SCAN or source_count >= MAX_PER_SOURCE:
                    break
                fn = vp.stem
                rel = str(vp.relative_to(base_dir))
                detected_exp = exp_override
                if detected_exp is None:
                    parts = rel.split('/')
                    dir_name = parts[1] if len(parts) >= 2 and parts[0] == 'experiments' else (parts[0] if parts else '')
                    for kw, et in [('counterfactual', 'counterfactual'), ('cross_task', 'cross_task'),
                                   ('vision_perturbation', 'vision_perturbation'), ('baseline', 'baseline'),
                                   ('grid_ablation', 'grid_ablation'), ('ablation', 'concept_ablation'),
                                   ('steering', 'steering')]:
                        if kw in dir_name:
                            detected_exp = et
                            break
                    if detected_exp is None:
                        detected_exp = dir_name
                detected_suite = fixed_suite
                if detected_suite is None:
                    for sn in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial', 'libero_long',
                               'metaworld_easy', 'metaworld_medium', 'metaworld_hard', 'metaworld_very_hard', 'metaworld']:
                        if sn in rel:
                            detected_suite = sn
                            break
                    # For MetaWorld dirs, detect from path
                    if detected_suite is None and 'metaworld' in str(base_dir).lower() or 'metaworld' in subpath:
                        detected_suite = 'metaworld'
                task_match = _re.search(r'task[_]?(\d+)', rel)
                task_num = int(task_match.group(1)) if task_match else None
                success_val = True if 'success' in fn.lower() else (False if 'fail' in fn.lower() else None)
                entry = {
                    'path': f'{model}/{rel}', 'filename': vp.name,
                    'experiment_type': detected_exp or 'unknown', 'suite': detected_suite,
                    'task': task_num, 'success': success_val, 'model': model,
                }
                if experiment_type and entry['experiment_type'] != experiment_type:
                    continue
                if suite and entry.get('suite') != suite:
                    continue
                if success_filter is not None and entry.get('success') != success_filter:
                    continue
                videos.append(entry)
                local_count += 1
                source_count += 1

        # Also load from baked ablation index (groot has 137K+ video entries with paths)
        index_map = {
            'xvla': 'xvla_ablation_index.json',
            'smolvla': 'smolvla_ablation_index.json',
            'groot': 'groot_ablation_index.json',
        }
        if model in index_map:
            baked_index = Path(__file__).parent / "data" / index_map[model]
            if baked_index.exists():
                with open(baked_index) as f:
                    idx_data = json.load(f)
                all_vids = idx_data.get('videos', []) if isinstance(idx_data, dict) else (idx_data if isinstance(idx_data, list) else [])
                if all_vids:
                    _sn = {'goal': 'libero_goal', 'object': 'libero_object', 'spatial': 'libero_spatial', '10': 'libero_10'}
                    for v in all_vids:
                        s = v.get('suite', '')
                        if s in _sn:
                            v['suite'] = _sn[s]
                    if experiment_type:
                        all_vids = [v for v in all_vids if v.get('experiment_type') == experiment_type]
                    if suite:
                        all_vids = [v for v in all_vids if v.get('suite') == suite]
                    if success_filter is not None:
                        all_vids = [v for v in all_vids if v.get('success') == success_filter]
                    videos.extend(all_vids)

        experiment_types = list(set(v.get('experiment_type', '') for v in videos if v.get('experiment_type')))
        suites = list(set(v.get('suite', '') for v in videos if v.get('suite')))
        concepts = list(set(v.get('concept', '') for v in videos if v.get('concept')))
        total_before_limit = len(videos)
        if limit_param is not None:
            try:
                limit_val = int(limit_param)
                if limit_val > 0:
                    videos = videos[:limit_val]
            except (ValueError, TypeError):
                pass
        return jsonify({
            'videos': videos, 'total': total_before_limit,
            'filters': {'models': [model], 'experiment_types': sorted(experiment_types),
                        'suites': sorted(suites), 'concepts': sorted(concepts)},
        })

    # Fallback: scan video directories if no index exists
    video_base = Path(__file__).parent / "data" / "videos"
    videos = []

    # Check for model-specific directory or use goal as default for pi05
    if model == 'pi05':
        search_dirs = [
            video_base / "pi05",
            video_base / "goal"
        ]
    else:
        search_dirs = [video_base / model]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Scan subdirectories (ablation, baseline, reconstruction, etc.)
        for subdir in search_dir.iterdir():
            if not subdir.is_dir():
                continue

            detected_experiment_type = subdir.name

            # Skip if experiment_type filter doesn't match
            if experiment_type and detected_experiment_type != experiment_type:
                continue

            for video_file in subdir.glob("*.mp4"):
                filename = video_file.stem

                # Detect suite from path or filename
                detected_suite = None
                for possible_suite in ['libero_10', 'libero_90', 'goal', 'spatial', 'concepts']:
                    if possible_suite in str(video_file) or possible_suite in filename:
                        detected_suite = possible_suite
                        break

                # Skip if suite filter doesn't match
                if suite and detected_suite != suite:
                    continue

                video_entry = {
                    'path': f"/videos/{model}/{subdir.name}/{video_file.name}",
                    'filename': video_file.name,
                    'experiment_type': detected_experiment_type,
                    'suite': detected_suite,
                    'success': None,  # Will be enriched from results.json
                    'model': model
                }

                # Enrich with success data from results.json
                video_entry = enrich_video_with_success(video_entry, model)

                # Fallback: parse success from filename if not found in results.json
                if video_entry.get('success') is None:
                    video_entry['success'] = 'success' in filename.lower()

                # Skip if success filter doesn't match
                if success_filter is not None and video_entry.get('success') != success_filter:
                    continue

                videos.append(video_entry)

    # Return in same format as index.json path for frontend consistency
    experiment_types = list(set(v.get('experiment_type', '') for v in videos if v.get('experiment_type')))
    suites = list(set(v.get('suite', '') for v in videos if v.get('suite')))
    concepts = list(set(v.get('concept', '') for v in videos if v.get('concept')))

    # Apply limit if specified
    total_before_limit = len(videos)
    if limit_param is not None:
        try:
            limit_val = int(limit_param)
            if limit_val > 0:
                videos = videos[:limit_val]
        except (ValueError, TypeError):
            pass

    return jsonify({
        'videos': videos,
        'total': total_before_limit,
        'filters': {
            'models': [model],
            'experiment_types': sorted(experiment_types),
            'suites': sorted(suites),
            'concepts': sorted(concepts),
        },
        'note': 'Generated from directory scan (no index.json found)'
    })


@videos_bp.route('/api/vla/video/<path:video_path>', methods=['GET'])
def serve_vla_video(video_path: str):
    """
    Serve video files from the data/videos directory.

    This endpoint serves video files from the symlinked directories under data/videos.
    It supports:
    - pi05 videos at data/videos/pi05/
    - openvla videos at data/videos/openvla/rollouts/ (symlinked to /data/openvla_rollouts)
    - Various experiment types (counterfactual, ablation, baseline, etc.)

    Security:
    - Path traversal attacks are prevented by checking for '..' in the path
    - Only .mp4 files are served
    - Files must exist within the VLA_VIDEO_DIR

    CORS headers are added for cross-origin video streaming support.

    Args:
        video_path: Relative path to the video file. Examples:
            - "pi05/counterfactual/video.mp4"
            - "openvla/rollouts/libero_10/activations_xxx/task_0/video.mp4"
            - "openvla/libero_10/activations_xxx/task_0/video.mp4" (auto-prefixed with rollouts/)

    Returns:
        Video file with appropriate headers for streaming, or 404 if not found.
    """
    # Security: Prevent path traversal attacks
    if '..' in video_path or video_path.startswith('/'):
        abort(400, description="Invalid video path")

    # Handle OFT concept ablation video paths (resolve from results directory)
    if video_path.startswith('oft_ablation/'):
        # Path format: oft_ablation/{suite}/{filename}
        parts = video_path[len('oft_ablation/'):].split('/', 1)
        if len(parts) == 2:
            suite, filename = parts
            ablation_path = OFT_ABLATION_VIDEO_DIR / suite / filename
            if ablation_path.exists() and ablation_path.suffix.lower() == '.mp4':
                response = make_response(send_file(
                    ablation_path, mimetype='video/mp4', as_attachment=False, download_name=filename
                ))
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
        if TIGRIS_PUBLIC_URL:
            return redirect(f"{TIGRIS_PUBLIC_URL}/{video_path}", code=302)
        abort(404, description="OFT ablation video not found")

    # Handle Pi0.5 concept ablation video paths (resolve from results directory)
    if video_path.startswith('pi05_ablation/'):
        # Path format: pi05_ablation/{suite}/{filename}
        parts = video_path[len('pi05_ablation/'):].split('/', 1)
        if len(parts) == 2:
            suite, filename = parts
            ablation_path = PI05_ABLATION_VIDEO_DIR / suite / filename
            if ablation_path.exists() and ablation_path.suffix.lower() == '.mp4':
                response = make_response(send_file(
                    ablation_path, mimetype='video/mp4', as_attachment=False, download_name=filename
                ))
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Cache-Control'] = 'public, max-age=3600'
                return response
        if TIGRIS_PUBLIC_URL:
            return redirect(f"{TIGRIS_PUBLIC_URL}/{video_path}", code=302)
        abort(404, description="Pi0.5 ablation video not found")

    # Handle openvla paths - the index.json paths are relative to /data/openvla_rollouts
    # which is symlinked at data/videos/openvla/rollouts
    if video_path.startswith('openvla/'):
        # Extract the path after "openvla/"
        openvla_subpath = video_path[len('openvla/'):]
        # If path doesn't already start with "rollouts/", prepend it
        if not openvla_subpath.startswith('rollouts/'):
            video_path = f'openvla/rollouts/{openvla_subpath}'

    # Handle GR00T video paths from ablation index
    # Index paths: groot_fraction_to_failure/libero_object/dit_L00/filename.mp4
    # Disk paths: /data/groot_rollouts{,_BATCH2}/sae_fraction_to_failure/libero_object/dit_L00/videos/filename.mp4
    if video_path.startswith('groot_'):
        groot_path_map = {
            'groot_fraction_to_failure/': 'sae_fraction_to_failure/',
            'groot_steering/': 'sae_steering/',
            'groot_temporal_ablation/': 'sae_temporal_ablation/',
            'groot_cross_suite_ablation/': 'sae_cross_suite_ablation/',
            'groot_feature_ablation/': 'sae_feature_ablation/',
        }
        for idx_prefix, disk_prefix in groot_path_map.items():
            if video_path.startswith(idx_prefix):
                subpath = video_path[len(idx_prefix):]
                # Insert 'videos/' before the filename: suite/layer/filename -> suite/layer/videos/filename
                parts = subpath.rsplit('/', 1)
                if len(parts) == 2:
                    subpath_with_videos = parts[0] + '/videos/' + parts[1]
                else:
                    subpath_with_videos = subpath
                # Check both BATCH2 (more videos) and BATCH1
                for base_dir in [GROOT_ROLLOUTS_DIR_BATCH2, GROOT_ROLLOUTS_DIR]:
                    # Try with videos/ subdirectory first
                    candidate = base_dir / disk_prefix / subpath_with_videos
                    if candidate.exists() and candidate.is_file():
                        response = make_response(send_file(
                            candidate, mimetype='video/mp4', as_attachment=False, download_name=candidate.name
                        ))
                        response.headers['Access-Control-Allow-Origin'] = '*'
                        response.headers['Cache-Control'] = 'public, max-age=3600'
                        return response
                    # Try without videos/ subdirectory
                    candidate = base_dir / disk_prefix / subpath
                    if candidate.exists() and candidate.is_file():
                        response = make_response(send_file(
                            candidate, mimetype='video/mp4', as_attachment=False, download_name=candidate.name
                        ))
                        response.headers['Access-Control-Allow-Origin'] = '*'
                        response.headers['Cache-Control'] = 'public, max-age=3600'
                        return response
                break  # Only match one prefix
        # If no file found, fall through to Tigris or 404

    # Normalize the path and ensure it's within the video directory
    # Use resolve() to follow symlinks and get the real path
    requested_path = VLA_VIDEO_DIR / video_path

    # Try the primary path (works with symlinks on the dev machine)
    resolved_path = None
    try:
        candidate = requested_path.resolve()
        if candidate.exists() and candidate.is_file():
            resolved_path = candidate
    except (ValueError, RuntimeError):
        pass

    # Fallback: try direct paths (Docker / environments without symlinks)
    if resolved_path is None:
        fallback_map = {
            'pi05/': [PI05_ROLLOUTS_DIR],
            'pi05_baseline/': [PI05_BASELINE_DIR],
            'openvla/rollouts/': [OPENVLA_ROLLOUTS_DIR],
            'openvla/': [OPENVLA_ROLLOUTS_DIR],
            'aloha/': [ALOHA_ROLLOUTS_DIR],
            'xvla/': [XVLA_ROLLOUTS_DIR],
            'smolvla/': [SMOLVLA_ROLLOUTS_DIR, SMOLVLA_LIBERO_DIR],
            'groot/': [GROOT_ROLLOUTS_DIR, GROOT_ROLLOUTS_DIR_BATCH2],
            'goal/': [VLA_VIDEO_DIR / 'goal'],
        }
        for prefix, base_dirs in fallback_map.items():
            if video_path.startswith(prefix):
                subpath = video_path[len(prefix):]
                for base_dir in base_dirs:
                    candidate = base_dir / subpath
                    if candidate.exists() and candidate.is_file():
                        resolved_path = candidate
                        break
                if resolved_path:
                    break

    if resolved_path is None:
        # Fallback: redirect to Tigris object storage (Fly.io deployment)
        if TIGRIS_PUBLIC_URL:
            tigris_key = video_path
            # Pi05 index paths are relative to pi05/ dir - need pi05/ prefix for Tigris.
            # Known top-level Tigris prefixes that already map correctly:
            tigris_top_prefixes = ('aloha/', 'openvla/', 'pi05/', 'pi05_ablation/',
                                   'pi05_baseline/', 'oft_ablation/',
                                   'xvla/', 'smolvla/', 'groot/',
                                   'groot_fraction_to_failure/',
                                   'xvla_ablation/', 'smolvla_ablation/')
            # Vision perturbation paths need source dir remapping to strip
            # the "main/" or "batch2/" subdirectory, regardless of whether
            # the pi05/ prefix is already present.
            #   pi05/vision_perturbation/main/...  → pi05/vision_perturbation/...
            #   pi05/vision_perturbation/batch2/... → pi05/vision_perturbation_batch2/...
            #   vision_perturbation/main/...       → pi05/vision_perturbation/...
            #   vision_perturbation/batch2/...      → pi05/vision_perturbation_batch2/...
            if tigris_key.startswith('pi05/vision_perturbation/main/'):
                tigris_key = 'pi05/vision_perturbation/' + tigris_key[len('pi05/vision_perturbation/main/'):]
            elif tigris_key.startswith('pi05/vision_perturbation/batch2/'):
                tigris_key = 'pi05/vision_perturbation_batch2/' + tigris_key[len('pi05/vision_perturbation/batch2/'):]
            elif tigris_key.startswith('vision_perturbation/main/'):
                tigris_key = 'pi05/vision_perturbation/' + tigris_key[len('vision_perturbation/main/'):]
            elif tigris_key.startswith('vision_perturbation/batch2/'):
                tigris_key = 'pi05/vision_perturbation_batch2/' + tigris_key[len('vision_perturbation/batch2/'):]
            elif not any(tigris_key.startswith(p) for p in tigris_top_prefixes):
                # All other pi05 paths (counterfactual/, cross_task_goal/, etc.)
                tigris_key = 'pi05/' + tigris_key
            return redirect(f"{TIGRIS_PUBLIC_URL}/{tigris_key}", code=302)
        abort(404, description="Video not found")

    if not resolved_path.suffix.lower() == '.mp4':
        abort(400, description="Only MP4 files are supported")

    # Get file size for Content-Length header
    file_size = resolved_path.stat().st_size

    # Create response with video file
    response = make_response(send_file(
        resolved_path,
        mimetype='video/mp4',
        as_attachment=False,
        download_name=resolved_path.name
    ))

    # Add CORS headers for video streaming
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Range, Content-Type'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range, Accept-Ranges'

    # Add headers for video streaming/seeking support
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['Content-Length'] = file_size
    response.headers['Cache-Control'] = 'public, max-age=3600'

    return response


@videos_bp.route('/api/vla/video/<path:video_path>', methods=['OPTIONS'])
def serve_vla_video_options(video_path: str):
    """
    Handle OPTIONS preflight requests for video CORS.

    This endpoint handles the CORS preflight request that browsers send
    before making cross-origin requests with custom headers (like Range).
    """
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Range, Content-Type'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response


@videos_bp.route('/api/vla/video/oft_ablation/<suite>/<filename>', methods=['GET'])
def serve_oft_ablation_video(suite: str, filename: str):
    if '..' in suite or '..' in filename:
        abort(400, description="Invalid path")
    video_path = OFT_ABLATION_VIDEO_DIR / suite / filename
    if not video_path.exists() or not video_path.suffix.lower() == '.mp4':
        if TIGRIS_PUBLIC_URL:
            return redirect(f"{TIGRIS_PUBLIC_URL}/oft_ablation/{suite}/{filename}", code=302)
        abort(404, description="Video not found")
    response = make_response(send_file(
        video_path, mimetype='video/mp4', as_attachment=False, download_name=filename
    ))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response


