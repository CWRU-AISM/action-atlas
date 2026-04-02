"""Action Atlas API - perturbation routes."""
from flask import Blueprint, request, jsonify, send_file, abort, make_response, redirect
from .helpers import *
from .data_loaders import *

perturbation_bp = Blueprint("perturbation", __name__)


def apply_perturbation(image: Image.Image, perturbation_type: str, strength: int) -> Image.Image:
    """
    Apply a perturbation to a PIL Image.

    Args:
        image: PIL Image to perturb
        perturbation_type: Type of perturbation to apply
        strength: Strength of perturbation (0-100)

    Returns:
        Perturbed PIL Image
    """
    # Ensure strength is in valid range
    strength = max(0, min(100, strength))

    # Convert to RGB if necessary for processing
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if perturbation_type == 'noise':
        # Add Gaussian noise
        img_array = np.array(image, dtype=np.float32)
        noise_level = strength * 2.55  # Scale to 0-255 range
        noise = np.random.normal(0, noise_level, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    elif perturbation_type == 'blur':
        # Apply Gaussian blur
        # Map strength 0-100 to radius 0-20
        radius = strength * 0.2
        if radius > 0:
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    elif perturbation_type == 'crop':
        # Center crop
        # Map strength 0-100 to crop ratio 1.0-0.2 (higher strength = more cropping)
        crop_ratio = 1.0 - (strength * 0.008)  # 0 -> 1.0, 100 -> 0.2
        crop_ratio = max(0.2, crop_ratio)

        width, height = image.size
        new_width = int(width * crop_ratio)
        new_height = int(height * crop_ratio)

        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        cropped = image.crop((left, top, right, bottom))
        # Resize back to original size
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    elif perturbation_type == 'h_flip':
        # Horizontal flip (binary operation)
        if strength >= 50:
            return ImageOps.mirror(image)
        return image

    elif perturbation_type == 'v_flip':
        # Vertical flip (binary operation)
        if strength >= 50:
            return ImageOps.flip(image)
        return image

    elif perturbation_type == 'rotate':
        # Rotation - map strength 0-100 to degrees 0-360
        angle = strength * 3.6
        return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(128, 128, 128))

    elif perturbation_type == 'grayscale':
        # Grayscale conversion with blending based on strength
        gray = ImageOps.grayscale(image).convert('RGB')
        if strength >= 100:
            return gray
        # Blend between original and grayscale
        blend_factor = strength / 100.0
        return Image.blend(image, gray, blend_factor)

    elif perturbation_type == 'invert':
        # Color inversion with blending based on strength
        inverted = ImageOps.invert(image)
        if strength >= 100:
            return inverted
        # Blend between original and inverted
        blend_factor = strength / 100.0
        return Image.blend(image, inverted, blend_factor)

    elif perturbation_type == 'brightness':
        # Brightness adjustment
        # Map strength: 0 -> 0.0 (black), 50 -> 1.0 (no change), 100 -> 2.0 (bright)
        factor = strength / 50.0
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    elif perturbation_type == 'contrast':
        # Contrast adjustment
        # Map strength: 0 -> 0.0 (flat gray), 50 -> 1.0 (no change), 100 -> 2.0 (high contrast)
        factor = strength / 50.0
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    elif perturbation_type == 'saturation':
        # Saturation adjustment
        # Map strength: 0 -> 0.0 (grayscale), 50 -> 1.0 (no change), 100 -> 2.0 (vivid)
        factor = strength / 50.0
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    else:
        # Unknown perturbation type, return original
        return image


@perturbation_bp.route('/api/vla/perturbation_types', methods=['GET'])
def get_perturbation_types():
    """
    Get available perturbation types with descriptions.

    Returns list of perturbation types with metadata for UI display.
    Frontend expects: { perturbation_types: [{id, label, hasStrength, category, description}, ...] }
    """
    # Build array format expected by frontend PerturbationTesting component
    # Map backend categories to frontend categories:
    # degradation/geometric/color -> 'vision' (frontend uses 'vision' for all image-level perturbations)
    backend_to_frontend_category = {
        'degradation': 'vision',
        'geometric': 'vision',
        'color': 'vision',
    }
    types_array = []
    for type_id, info in PERTURBATION_TYPES.items():
        has_strength = info.get('default_strength', 50) != 100  # Binary ops have default 100
        backend_cat = info.get('category', 'vision')
        frontend_cat = backend_to_frontend_category.get(backend_cat, backend_cat)
        types_array.append({
            'id': type_id,
            'label': info['name'],
            'hasStrength': has_strength,
            'category': frontend_cat,
            'description': info.get('description', ''),
            'icon': type_id,
        })

    return jsonify({
        'perturbation_types': types_array,
        'categories': ['vision']
    })


# Cache for VP experiment results (loaded once per process)
_vp_results_cache: Dict[str, dict] = {}


def _load_vp_results(model: str) -> Optional[dict]:
    """Load baked VP results for a model, with caching."""
    if model in _vp_results_cache:
        return _vp_results_cache[model]

    data_dir = Path(__file__).parent / "data"
    if model in ('pi05', 'pi0.5'):
        fpath = data_dir / "vp_results_pi05.json"
    elif model in ('openvla', 'openvla_oft', 'oft'):
        fpath = data_dir / "vp_results_openvla.json"
    elif model == 'xvla':
        fpath = data_dir / "vp_results_xvla.json"
    elif model == 'smolvla':
        fpath = data_dir / "vp_results_smolvla.json"
    elif model == 'groot':
        fpath = data_dir / "vp_results_groot.json"
    else:
        return None

    if not fpath.exists():
        # Fallback: try to load from experiment_results_<model>.json which has VP data
        # in a different format (per_perturbation instead of suites/tasks/overall)
        canonical = model.replace('pi0.5', 'pi05').replace('openvla_oft', 'openvla').replace('oft', 'openvla')
        exp_path = data_dir / f"experiment_results_{canonical}.json"
        if exp_path.exists():
            try:
                with open(exp_path) as f:
                    exp_data = json.load(f)
                vp_section = exp_data.get('vision_perturbation', {})
                if vp_section:
                    # Convert experiment_results VP format to vp_results format
                    converted = _convert_experiment_vp_to_vp_results(canonical, vp_section)
                    _vp_results_cache[model] = converted
                    return converted
            except Exception as e:
                print(f"Error loading VP fallback from {exp_path}: {e}")
        return None

    with open(fpath) as f:
        data = json.load(f)
    _vp_results_cache[model] = data
    return data


def _convert_experiment_vp_to_vp_results(model: str, vp_section: dict) -> dict:
    """
    Convert experiment_results vision_perturbation format to vp_results format.

    experiment_results format:
      { "suite_name": { "per_perturbation": { "blur_light": { "overall_success_rate": 0.9, "per_task": {...} } } } }

    vp_results format:
      { "model": "...", "suites": { "suite_name": { "overall": { "blur_light": { "success_rate": 0.9 } }, "tasks": {} } } }
    """
    suites = {}
    all_perturbation_types = set()
    total_episodes = 0

    for suite_key, suite_data in vp_section.items():
        per_pert = suite_data.get('per_perturbation', {})
        if not per_pert:
            continue

        overall = {}
        tasks: Dict[str, dict] = {}

        for pert_name, pert_data in per_pert.items():
            all_perturbation_types.add(pert_name)
            sr = pert_data.get('overall_success_rate', pert_data.get('success_rate', 0))
            n_eps = pert_data.get('n_episodes', pert_data.get('n_tasks', 0) * 3)  # estimate
            total_episodes += n_eps
            overall[pert_name] = {
                'success_rate': sr if isinstance(sr, (int, float)) and sr <= 1.0 else sr / 100.0,
                'n_episodes': n_eps,
            }

            # Build per-task results if available
            per_task = pert_data.get('per_task', {})
            for task_id, task_sr in per_task.items():
                task_str = str(task_id)
                if task_str not in tasks:
                    tasks[task_str] = {}
                task_sr_val = task_sr if isinstance(task_sr, (int, float)) else (1.0 if task_sr else 0.0)
                if isinstance(task_sr_val, (int, float)) and task_sr_val > 1.0:
                    task_sr_val = task_sr_val / 100.0
                tasks[task_str][pert_name] = {
                    'success_rate': task_sr_val,
                }

        suites[suite_key] = {
            'overall': overall,
            'tasks': tasks,
        }

    return {
        'model': model,
        'total_episodes': total_episodes,
        'suites': suites,
        'perturbation_types': sorted(list(all_perturbation_types)),
    }


@perturbation_bp.route('/api/vla/vp_experiment_results', methods=['GET'])
def get_vp_experiment_results():
    """
    Return real VP experiment results for data-driven inference.

    Query params:
        model: model name (pi05, openvla, xvla, smolvla, groot; default: 'pi05')
        suite: optional suite filter (e.g., 'libero_goal')
        task: optional task index filter (e.g., '0')
        perturbation: optional perturbation type filter (e.g., 'blur_heavy')
    """
    model = request.args.get('model', 'pi05')
    suite = request.args.get('suite')
    task = request.args.get('task')
    perturbation = request.args.get('perturbation')

    data = _load_vp_results(model)
    if data is None:
        return jsonify({'error': f'No VP results found for model {model}'}), 404

    # If no filters, return the full dataset
    if not suite and task is None and not perturbation:
        return jsonify(data)

    # Filter down to requested scope
    result = {
        'model': data.get('model', model),
        'perturbation_types': data.get('perturbation_types', []),
    }

    suites = data.get('suites', {})

    # Pi0.5 uses 'goal'/'spatial', OFT uses 'libero_goal'/'libero_spatial'
    # Normalize suite name for lookup
    suite_key = suite
    if suite and suite not in suites:
        # Try stripping 'libero_' prefix or adding it
        alt = suite.replace('libero_', '') if suite.startswith('libero_') else f'libero_{suite}'
        if alt in suites:
            suite_key = alt

    if suite_key and suite_key in suites:
        suite_data = suites[suite_key]

        if task is not None:
            task_str = str(task)
            task_data = suite_data.get('tasks', {}).get(task_str, {})

            if perturbation:
                # Return specific perturbation result for this task
                if perturbation in task_data:
                    cond_data = task_data[perturbation]
                    # Also include baseline for comparison
                    baseline_data = task_data.get('baseline', {})
                    result['result'] = {
                        **cond_data,
                        'perturbation': perturbation,
                        'baseline_success_rate': baseline_data.get('success_rate'),
                        'baseline_avg_n_steps': baseline_data.get('avg_n_steps'),
                        'baseline_video_path': baseline_data.get('video_path'),
                    }
                else:
                    result['result'] = None
                    result['available_perturbations'] = list(task_data.keys())
            else:
                result['task_results'] = task_data
        else:
            if perturbation:
                # Return suite-level overall for this perturbation
                overall = suite_data.get('overall', {})
                if perturbation in overall:
                    baseline_overall = overall.get('baseline', {})
                    result['result'] = {
                        **overall[perturbation],
                        'perturbation': perturbation,
                        'baseline_success_rate': baseline_overall.get('success_rate'),
                    }
                else:
                    result['result'] = None
                    result['available_perturbations'] = list(overall.keys())
            else:
                result['suite_data'] = suite_data
    elif suite:
        result['error'] = f'Suite {suite} not found'
        result['available_suites'] = list(suites.keys())
    else:
        # No suite filter but have perturbation filter - return overall across suites
        if perturbation:
            cross_suite = {}
            for s_name, s_data in suites.items():
                overall = s_data.get('overall', {})
                if perturbation in overall:
                    cross_suite[s_name] = overall[perturbation]
            result['by_suite'] = cross_suite
        else:
            result['suites'] = suites

    return jsonify(result)


@perturbation_bp.route('/api/vla/perturb', methods=['POST'])
def apply_perturbation_endpoint():
    """
    Apply perturbation to an image.

    Request JSON:
        image: Base64 encoded image (with or without data URL prefix)
        perturbation_type: Type of perturbation (noise, blur, crop, h_flip, v_flip, rotate, grayscale, invert, brightness)
        strength: Perturbation strength (0-100)

    Returns:
        Base64 encoded perturbed image
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 400,
                'error': {'code': 'MISSING_DATA', 'message': 'Request body required'}
            }), 400

        # Extract parameters
        image_data = data.get('image')
        perturbation_type = data.get('perturbation_type', 'noise')
        strength = data.get('strength', 50)

        if not image_data:
            return jsonify({
                'status': 400,
                'error': {'code': 'MISSING_IMAGE', 'message': 'image parameter required'}
            }), 400

        # Validate perturbation type
        if perturbation_type not in PERTURBATION_TYPES:
            return jsonify({
                'status': 400,
                'error': {
                    'code': 'INVALID_PERTURBATION',
                    'message': f'Unknown perturbation type: {perturbation_type}. Valid types: {list(PERTURBATION_TYPES.keys())}'
                }
            }), 400

        # Validate strength
        try:
            strength = int(strength)
            if strength < 0 or strength > 100:
                raise ValueError("Strength must be 0-100")
        except (ValueError, TypeError) as e:
            return jsonify({
                'status': 400,
                'error': {'code': 'INVALID_STRENGTH', 'message': 'strength must be an integer 0-100'}
            }), 400

        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({
                'status': 400,
                'error': {'code': 'INVALID_IMAGE', 'message': f'Failed to decode image: {str(e)}'}
            }), 400

        # Apply perturbation
        perturbed_image = apply_perturbation(image, perturbation_type, strength)

        # Encode result as base64
        buffer = io.BytesIO()
        perturbed_image.save(buffer, format='PNG')
        buffer.seek(0)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'perturbed_image': f'data:image/png;base64,{result_base64}',
            'perturbation_type': perturbation_type,
            'strength': strength,
            'original_size': list(image.size),
            'perturbed_size': list(perturbed_image.size)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Processing error: {str(e)}'
        }), 500


@perturbation_bp.route('/api/vla/extract_frame', methods=['POST'])
def extract_video_frame():
    """
    Extract a frame from a video file.

    Request JSON:
        video_path: Path to the video file (relative to data directory or absolute)
        frame_number: Frame number to extract (0-indexed)
        OR
        timestamp: Timestamp in seconds to extract frame from

    Returns:
        Base64 encoded image of the extracted frame
    """
    if not CV2_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'OpenCV (cv2) is not installed. Install with: pip install opencv-python'
        }), 503

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body required'
            }), 400

        video_path = data.get('video_path')
        frame_number = data.get('frame_number')
        timestamp = data.get('timestamp')

        if not video_path:
            return jsonify({
                'success': False,
                'error': 'video_path parameter required'
            }), 400

        if frame_number is None and timestamp is None:
            return jsonify({
                'success': False,
                'error': 'Either frame_number or timestamp required'
            }), 400

        # Resolve video path
        video_path_str = video_path
        video_path = Path(video_path)
        if not video_path.is_absolute():
            # Try relative to various data directories
            # VLA_VIDEO_DIR is data/videos, and videos are under pi05/ subdirectory
            possible_paths = [
                VLA_VIDEO_DIR / video_path,  # data/videos/pi05/counterfactual/...
                Path(__file__).parent / "data" / video_path,
                Path(__file__).parent / "data" / "videos" / video_path,
                Path(__file__).parent.parent / "outputs" / video_path,
                Path(__file__).parent.parent / "outputs" / "ablation_videos" / video_path,
            ]
            # Add direct fallback paths for model-prefixed paths (Docker / no symlinks)
            video_path_s = str(video_path)
            if video_path_s.startswith('pi05/'):
                possible_paths.append(PI05_ROLLOUTS_DIR / video_path_s[len('pi05/'):])
            elif video_path_s.startswith('pi05_baseline/'):
                possible_paths.append(PI05_BASELINE_DIR / video_path_s[len('pi05_baseline/'):])
            elif video_path_s.startswith('openvla/rollouts/'):
                possible_paths.append(OPENVLA_ROLLOUTS_DIR / video_path_s[len('openvla/rollouts/'):])
            elif video_path_s.startswith('openvla/'):
                possible_paths.append(OPENVLA_ROLLOUTS_DIR / video_path_s[len('openvla/'):])
            video_path = None
            for p in possible_paths:
                # Follow symlinks to check if target exists
                try:
                    if p.exists():
                        # Always resolve the path to handle symlinked parent directories
                        video_path = p.resolve()
                        break
                except Exception:
                    continue

            if video_path is None:
                return jsonify({
                    'success': False,
                    'error': f'Video file not found: {video_path_str}. Searched in: {[str(p) for p in possible_paths]}'
                }), 404

        if not video_path.exists():
            return jsonify({
                'success': False,
                'error': f'Video file not found: {video_path}'
            }), 404

        # Open video with cv2
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return jsonify({
                'success': False,
                'error': f'Failed to open video: {video_path}'
            }), 500

        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate frame number from timestamp if provided
            if timestamp is not None:
                frame_number = int(timestamp * fps)

            # Validate frame number
            frame_number = int(frame_number)
            if frame_number < 0:
                frame_number = 0
            if frame_number >= total_frames:
                frame_number = total_frames - 1

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read frame
            ret, frame = cap.read()

            if not ret:
                return jsonify({
                    'success': False,
                    'error': f'Failed to read frame {frame_number}'
                }), 500

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)

            # Encode as base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return jsonify({
                'success': True,
                'frame': f'data:image/png;base64,{result_base64}',
                'frame_number': frame_number,
                'timestamp': frame_number / fps if fps > 0 else 0,
                'video_info': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'duration': total_frames / fps if fps > 0 else 0
                }
            })

        finally:
            cap.release()

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
# Experiment Statistics Endpoint
import re


@perturbation_bp.route('/api/vla/perturbation_videos', methods=['GET'])
def get_perturbation_videos():
    """Find rollout videos matching a perturbation type to show what the robot did."""
    perturbation_type = request.args.get('type', 'noise')
    suite = request.args.get('suite', '')
    model = request.args.get('model', 'pi05')
    limit = int(request.args.get('limit', '6'))

    # Map frontend perturbation names to search terms found in subtypes OR filenames
    # Both pi05 and openvla encode perturbation info differently:
    #   pi05: subtype is broad ('crop','flip','rotate'), but filename has detail ('blur_heavy_full')
    #   openvla: subtype is '' (empty), but filename has detail ('task0_blur_heavy.mp4')
    perturbation_map = {
        'noise': ['noise_high', 'noise_low', 'noise_medium', 'noise'],
        'blur': ['blur_heavy', 'blur_light', 'blur_medium', 'blur_5', 'blur'],
        'h_flip': ['h_flip', 'horizontal_flip'],
        'v_flip': ['v_flip', 'vertical_flip'],
        'rotate': ['rotate_15', 'rotate_45', 'rotate'],
        'grayscale': ['grayscale', 'greyscale'],
        'crop': ['crop_right', 'crop_left', 'crop_top', 'crop_bottom', 'crop_50', 'crop_70', 'crop_90', 'center_crop', 'crop'],
        'brightness': ['bright_up', 'bright_down', 'brightness'],
        'contrast': ['contrast_up', 'contrast_down', 'contrast'],
        'saturation': ['saturation'],
        'invert': ['invert'],
    }

    search_terms = perturbation_map.get(perturbation_type, [perturbation_type])

    # Load video index - try model-specific directory first
    index_path = VLA_VIDEO_DIR / model / "index.json"
    if not index_path.exists():
        # Models without index.json: scan filesystem for VP videos directly
        vp_scan_dirs = {
            'groot': [
                (GROOT_ROLLOUTS_DIR / 'libero_goal' / 'visual_perturbation', 'libero_goal'),
                (GROOT_ROLLOUTS_DIR / 'libero_object' / 'visual_perturbation', 'libero_object'),
                (GROOT_ROLLOUTS_DIR / 'libero_long' / 'visual_perturbation', 'libero_long'),
            ],
            'smolvla': [
                (SMOLVLA_ROLLOUTS_DIR / 'metaworld_vision_perturbation', 'metaworld'),
            ],
            'xvla': [
                (XVLA_ROLLOUTS_DIR / 'xvla_libero' / 'experiments' / 'vision_libero_goal', 'libero_goal'),
                (XVLA_ROLLOUTS_DIR / 'xvla_libero' / 'experiments' / 'vision_libero_object', 'libero_object'),
                (XVLA_ROLLOUTS_DIR / 'xvla_libero' / 'experiments' / 'vision_libero_spatial', 'libero_spatial'),
                (XVLA_ROLLOUTS_DIR / 'xvla_libero' / 'experiments' / 'vision_libero_10', 'libero_10'),
            ],
        }
        scan_entries = vp_scan_dirs.get(model, [])
        matching = []
        import re as _re_vp
        for scan_dir, dir_suite in scan_entries:
            if not scan_dir.exists():
                continue
            if suite and dir_suite.lower() != suite.lower():
                continue
            for vp_file in sorted(scan_dir.rglob("*.mp4")):
                if len(matching) >= limit:
                    break
                filename = vp_file.name.lower()
                if '_combined' in filename:
                    continue
                matches_filename = any(term in filename for term in search_terms)
                if matches_filename:
                    name_no_ext = filename.replace('.mp4', '')
                    display_subtype = ''
                    for term in search_terms:
                        if term in name_no_ext:
                            idx = name_no_ext.find(term)
                            display_subtype = name_no_ext[idx:]
                            break
                    # Extract task number from path
                    task_match = _re_vp.search(r'task[_]?(\d+)', str(vp_file))
                    task_num = task_match.group(1) if task_match else ''
                    # Compute relative path from the base directory used by the video serving fallback map
                    base_dir_map = {
                        'groot': GROOT_ROLLOUTS_DIR,
                        'smolvla': SMOLVLA_ROLLOUTS_DIR,
                        'xvla': XVLA_ROLLOUTS_DIR,
                    }
                    vp_base = base_dir_map.get(model, scan_dir.parent)
                    rel_path = str(vp_file.relative_to(vp_base))
                    matching.append({
                        'path': f"/videos/{model}/{rel_path}",
                        'suite': dir_suite,
                        'subtype': display_subtype or perturbation_type,
                        'task': task_num,
                        'seed': '',
                    })
            if len(matching) >= limit:
                break
        return jsonify({
            'perturbation_type': perturbation_type,
            'videos': matching,
            'total': len(matching),
            'model': model,
        })

    with open(index_path) as f:
        index_data = json.load(f)

    matching = []
    for video in index_data.get('videos', []):
        subtype = video.get('subtype', '').lower()
        exp_type = video.get('experiment_type', '').lower()
        video_suite = video.get('suite', '').lower()
        # Also check the filename for perturbation type (handles openvla empty subtype
        # and pi05 broad subtype cases)
        filename = video.get('path', '').split('/')[-1].lower()

        if exp_type != 'vision_perturbation':
            continue
        if suite and video_suite != suite.lower():
            continue
        # Skip combined videos (pi05 has _combined versions)
        if '_combined' in filename:
            continue

        # Match against both subtype field AND filename
        matches_subtype = any(term in subtype for term in search_terms)
        matches_filename = any(term in filename for term in search_terms)
        if matches_subtype or matches_filename:
            # Extract a meaningful subtype label from the filename if subtype is empty
            display_subtype = subtype
            if not display_subtype or display_subtype == 'baseline':
                # Parse from filename: e.g. task0_blur_heavy.mp4 -> blur_heavy
                name_no_ext = filename.replace('.mp4', '')
                # Remove task prefix: task0_ or libero_goal_task0_s42_
                for term in search_terms:
                    if term in name_no_ext:
                        # Find the perturbation part
                        idx = name_no_ext.find(term)
                        display_subtype = name_no_ext[idx:]
                        break

            matching.append({
                'path': f"/videos/{model}/{video['path']}",
                'suite': video.get('suite', ''),
                'subtype': display_subtype or perturbation_type,
                'task': video.get('task', ''),
                'seed': video.get('seed', ''),
            })

        if len(matching) >= limit:
            break

    return jsonify({
        'perturbation_type': perturbation_type,
        'videos': matching,
        'total': len(matching),
        'model': model
    })


@perturbation_bp.route('/api/vla/vp_comparison', methods=['GET'])
def get_vp_comparison():
    """
    Get vision perturbation baseline/perturbation video pairs for comparison.
    
    Query params:
        model: Model name (pi05, openvla, xvla, smolvla, groot)
        suite: Suite name (libero_goal, etc.)
        task: Task number (0-9)
    
    Returns:
        baseline_video: Path to baseline video
        perturbations: List of {type, path} for each available perturbation
    """
    model = request.args.get('model', 'pi05')
    suite = request.args.get('suite', 'libero_goal')
    task = request.args.get('task', '0')
    
    # Load from baked index
    index_map = {
        'pi05': 'pi05_ablation_index.json',
        'openvla': 'oft_ablation_index.json',
        'xvla': 'xvla_ablation_index.json',
        'smolvla': 'smolvla_ablation_index.json',
        'groot': 'groot_ablation_index.json',
    }
    
    idx_file = Path(__file__).parent / "data" / index_map.get(model, '')

    # Cache the index
    cache_key = f'vp_index_{model}'
    if cache_key in _ablation_summary_cache:
        all_vids = _ablation_summary_cache[cache_key]
    elif idx_file.exists():
        with open(idx_file) as f:
            data = json.load(f)
        all_vids = data.get('videos', data if isinstance(data, list) else [])
        _ablation_summary_cache[cache_key] = all_vids
    else:
        all_vids = []

    # For Pi0.5/OFT: also check for VP videos from video indexes or Tigris paths
    if not any(v.get('experiment_type') == 'vision_perturbation' for v in all_vids[:100]):
        # Load from main video index
        video_index = load_video_index(model)
        if video_index:
            vp_from_main = [v for v in video_index.get('videos', []) if v.get('experiment_type') == 'vision_perturbation']
            all_vids = all_vids + vp_from_main
    
    # Filter VP videos for this suite and task
    # Include both vision_perturbation AND baseline videos that are in VP folders
    vp_vids = [v for v in all_vids
               if (v.get('experiment_type') == 'vision_perturbation'
                   or (v.get('experiment_type') == 'baseline' and 'perturbation' in v.get('path', '')))
               and (v.get('suite', '') == suite or suite in v.get('path', ''))]

    # Get available tasks
    task_nums = sorted(set(v.get('task', 0) for v in vp_vids if v.get('task') is not None))
    
    # Filter for requested task
    task_int = int(task)
    task_vids = [v for v in vp_vids if v.get('task') == task_int]
    
    # If no task field, try matching from path
    if not task_vids:
        task_vids = [v for v in vp_vids if f'task_{task}' in v.get('path', '') or f'task{task}' in v.get('path', '')]
    
    # Separate baseline from perturbations
    baseline = None
    perturbations = []
    
    for v in task_vids:
        subtype = v.get('subtype', '')
        fn = v.get('filename', '')
        
        if subtype == 'baseline' or 'baseline' in fn.lower():
            if baseline is None:  # Take first baseline
                baseline = v
        elif subtype and subtype not in ('ep00', 'ep01', 'ep02'):
            perturbations.append({
                'type': subtype,
                'path': v.get('path', ''),
                'filename': fn,
                'success': v.get('success'),
            })
    
    # Sort perturbations alphabetically
    perturbations.sort(key=lambda p: p['type'])
    
    return jsonify({
        'model': model,
        'suite': suite,
        'task': task_int,
        'baseline': baseline,
        'perturbations': perturbations,
        'available_tasks': task_nums[:20],
        'total_perturbation_types': len(perturbations),
    })


# Pre-computed feature embeddings for semantic search
_feature_embeddings_cache = {}

