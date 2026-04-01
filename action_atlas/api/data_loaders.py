"""Data loading utilities for Action Atlas API."""
from .helpers import *

def parse_success_from_path(video_path: str, video_data: Optional[Dict] = None) -> Optional[bool]:
    """
    Parse success/failure status from video path or video metadata.

    Patterns checked:
    - _s followed by digits in filename = success (e.g., _s42 means success with seed 42)
    - _f followed by digits in filename = failure
    - success=True or success=False in filename
    - 'success' field in video_data dict

    Args:
        video_path: Path to the video file
        video_data: Optional dict with video metadata including 'success' field

    Returns:
        True for success, False for failure, None if unknown
    """
    # First check if video_data has explicit success field
    if video_data is not None:
        success_val = video_data.get('success')
        if success_val is not None:
            return bool(success_val)

    # Get just the filename for pattern matching
    filename = Path(video_path).stem.lower()

    # Check for explicit success=True/False in filename
    if 'success=true' in filename or 'success_true' in filename:
        return True
    if 'success=false' in filename or 'success_false' in filename or 'failure' in filename:
        return False

    # Check for _success or _fail patterns
    if '_success' in filename:
        return True
    if '_fail' in filename:
        return False

    # Check for _s\d pattern (seed with success) vs _f\d pattern (seed with failure)
    # Note: In many datasets, _s followed by digits indicates success with a seed number
    # and _f followed by digits indicates failure with a seed number
    # However, we need to be careful as some datasets use _s just for seed
    # We'll treat _s\d as success only if there's no other indicator

    # For now, return None if we can't determine
    return None


def load_ablation_results(model: str) -> List[Dict]:
    """
    Load all ablation result files for the specified model.

    Returns list of ablation result dicts with baseline/intervention success rates.
    """
    results = []

    # Look for ablation results in data/libero_10/ablation_results/
    ablation_dir = Path(__file__).parent / "data" / "libero_10" / "ablation_results"

    if not ablation_dir.exists():
        return results

    for result_file in ablation_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                data['filename'] = result_file.name
                results.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    return results


def find_results_json_for_video(video_path: str) -> Optional[Dict]:
    """
    Find and parse results.json for a given video path.

    Searches for results.json in the video's directory or parent directories.
    Handles various experiment directory structures:
    - cross_task: batch_N/pair_X_Y/inject_X_into_Y/video.mp4 with results.json in batch_N/
    - counterfactual: Various subdirectories with metadata.jsonl
    - vision_perturbation: Video filenames encode results

    Args:
        video_path: Path to the video file (relative or absolute)

    Returns:
        Dict with results data or None if not found
    """
    video_path = Path(video_path)

    # Handle relative paths - resolve against known video directories
    if not video_path.is_absolute():
        possible_bases = [
            VLA_VIDEO_DIR,
            VLA_VIDEO_DIR / "pi05",
            PI05_ROLLOUTS_DIR,
        ]
        for base in possible_bases:
            candidate = base / video_path
            if candidate.exists():
                video_path = candidate
                break
        else:
            # Try to resolve symlinks
            for base in possible_bases:
                try:
                    candidate = (base / video_path).resolve()
                    if candidate.exists():
                        video_path = candidate
                        break
                except Exception:
                    continue

    if not video_path.exists():
        return None

    video_path = video_path.resolve()

    # Search for results.json in parent directories (up to 5 levels)
    current_dir = video_path.parent
    for _ in range(5):
        results_json = current_dir / "results.json"
        if results_json.exists():
            try:
                with open(results_json) as f:
                    return {
                        'results_path': str(results_json),
                        'video_path': str(video_path),
                        'data': json.load(f)
                    }
            except (json.JSONDecodeError, IOError):
                pass

        # Also check for metadata.jsonl (used in counterfactual experiments)
        metadata_jsonl = current_dir / "metadata.jsonl"
        if metadata_jsonl.exists():
            try:
                entries = []
                with open(metadata_jsonl) as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))
                return {
                    'results_path': str(metadata_jsonl),
                    'video_path': str(video_path),
                    'data': {'entries': entries, 'type': 'metadata_jsonl'}
                }
            except (json.JSONDecodeError, IOError):
                pass

        # Move up one directory
        parent = current_dir.parent
        if parent == current_dir:  # Reached root
            break
        current_dir = parent

    return None


def extract_video_results(video_path: str, results_data: Optional[Dict]) -> Dict:
    """
    Extract experiment results specific to a video file.

    Parses the video filename and path to find corresponding results
    in the results.json data.

    Args:
        video_path: Path to the video file
        results_data: Loaded results.json data

    Returns:
        Dict with extracted results for this video
    """
    video_path = Path(video_path)
    filename = video_path.stem

    result = {
        'video_path': str(video_path),
        'filename': filename,
        'success': None,
        'steps': None,
        'baseline_success': None,
        'baseline_steps': None,
        'experiment_type': None,
        'action_stats': None,
        'metadata': {}
    }

    if results_data is None:
        return result

    data = results_data.get('data', {})

    # Handle cross_task experiment results
    if 'results' in data and 'suite' in data:
        result['experiment_type'] = 'cross_task'
        result['metadata']['suite'] = data.get('suite')

        # Parse path to find pair and injection type
        # Expected: .../pair_X_Y/inject_X_into_Y/experiment_name.mp4
        path_parts = video_path.parts

        pair_name = None
        inject_name = None
        for i, part in enumerate(path_parts):
            if part.startswith('pair_'):
                pair_name = part
            elif part.startswith('inject_'):
                inject_name = part

        if pair_name and pair_name in data['results']:
            pair_data = data['results'][pair_name]
            result['metadata']['task_a'] = pair_data.get('task_a')
            result['metadata']['task_b'] = pair_data.get('task_b')
            result['metadata']['prompt_a'] = pair_data.get('prompt_a')
            result['metadata']['prompt_b'] = pair_data.get('prompt_b')

            # Get baseline results
            baseline_0 = pair_data.get('baseline_task_0', {})
            baseline_1 = pair_data.get('baseline_task_1', {})

            # Determine which baseline is relevant based on injection direction
            if inject_name:
                # inject_0_into_1 means task 0's activations injected into task 1's environment
                if inject_name == 'inject_0_into_1':
                    result['baseline_success'] = baseline_0.get('success')
                    result['baseline_steps'] = baseline_0.get('steps')
                elif inject_name == 'inject_1_into_0':
                    result['baseline_success'] = baseline_1.get('success')
                    result['baseline_steps'] = baseline_1.get('steps')

                # Find specific experiment results
                inject_data = pair_data.get(inject_name, {})
                exp_name = filename  # e.g., "cross_prompt_expert_L16"

                if exp_name in inject_data:
                    exp_results = inject_data[exp_name]
                    result['success'] = exp_results.get('success')
                    result['steps'] = exp_results.get('steps')
                    result['metadata']['prompt'] = exp_results.get('prompt')
                    result['metadata']['inject_pathway'] = exp_results.get('inject_pathway')

                    # Extract action stats if available
                    if 'cos_to_src_baseline' in exp_results:
                        result['action_stats'] = {
                            'cos_to_src_baseline': exp_results.get('cos_to_src_baseline'),
                            'cos_to_dst_baseline': exp_results.get('cos_to_dst_baseline'),
                            'xyz_to_src_baseline': exp_results.get('xyz_to_src_baseline'),
                            'xyz_to_dst_baseline': exp_results.get('xyz_to_dst_baseline'),
                        }

    # Handle metadata.jsonl (counterfactual experiments)
    elif data.get('type') == 'metadata_jsonl':
        result['experiment_type'] = 'counterfactual'
        entries = data.get('entries', [])

        # Try to match by key or filename pattern
        for entry in entries:
            entry_key = entry.get('key', '')
            if entry_key in filename or filename in entry_key:
                result['metadata'] = entry
                result['steps'] = entry.get('n_steps')
                # Success is not always recorded in metadata.jsonl
                break

    # Handle vision perturbation experiments (results encoded in filename)
    # Pattern: libero_goal_task0_s42_blur_0.5.mp4
    if 'perturbation' in str(video_path) or any(p in filename for p in ['blur', 'noise', 'flip', 'rotate', 'crop']):
        result['experiment_type'] = 'vision_perturbation'

        # Parse perturbation from filename
        perturbations = ['blur', 'noise', 'h_flip', 'v_flip', 'rotate', 'crop_top', 'crop_bottom', 'crop_left', 'crop_right']
        for pert in perturbations:
            if pert in filename:
                result['metadata']['perturbation'] = pert
                break

        # Check if "baseline" in filename
        if 'baseline' in filename:
            result['metadata']['is_baseline'] = True

    return result


# Results JSON cache to avoid repeated file reads
_results_json_cache: Dict[str, Dict] = {}

# Bulk success lookup: maps video_path_key -> success (bool)
# Built lazily on first use per model
_bulk_success_map: Dict[str, Dict[str, Optional[bool]]] = {}
_bulk_success_map_built: Dict[str, bool] = {}

# Pi0.5 concept ablation success map: (suite, layer, concept_type, concept_name, task) -> success
_pi05_ablation_success_map: Optional[Dict] = None

# OFT concept ablation success map: (suite, layer, concept_type, concept_name, task, episode) -> success
_oft_ablation_success_map: Optional[Dict] = None