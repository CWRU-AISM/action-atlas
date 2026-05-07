# Action Atlas - Ablation Routes
# Provides API endpoints for feature ablation experiments

import json
from pathlib import Path
from flask import Blueprint, request, jsonify

from ..config import VLA_CONFIGS, DEFAULT_VLA_MODEL

ablation_bp = Blueprint('ablation', __name__)


def get_ablation_results(vla_model: str):
    # Load ablation experiment results from multiple directories
    config = VLA_CONFIGS.get(vla_model)
    if config is None:
        return None

    results = {}

    # List of directories to check for ablation results
    ablation_dirs = [
        Path("outputs/concept_ablation_eval"),
        Path("outputs/temporal_ablation"),
        Path("outputs/single_step_ablation"),
        Path("outputs/fine_ablation"),
        Path("outputs/steering_only"),
        Path("outputs/cross_suite_steering"),
        Path("outputs/percentage_ablation"),
    ]

    for ablation_dir in ablation_dirs:
        if not ablation_dir.exists():
            continue

        # Load latest JSON from each directory
        for results_file in sorted(ablation_dir.glob("*.json"), reverse=True):
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    key = f"{ablation_dir.name}/{results_file.stem}"
                    results[key] = data
            except (json.JSONDecodeError, IOError):
                continue
            break  # Just get latest from each dir

    return results


@ablation_bp.route('/api/vla/ablation/results', methods=['GET'])
def get_ablation_experiment_results():
    vla_model = request.args.get('model', DEFAULT_VLA_MODEL)

    results = get_ablation_results(vla_model)
    if not results:
        return jsonify({
            "status": 404,
            "error": {"code": "NO_RESULTS", "message": "No ablation results found"}
        }), 404

    return jsonify({
        "status": 200,
        "data": {
            "vla_model": vla_model,
            "experiments": results
        }
    })


@ablation_bp.route('/api/vla/ablation/summary', methods=['GET'])
def get_ablation_summary():
    # Get summary of ablation effects for each concept
    vla_model = request.args.get('model', DEFAULT_VLA_MODEL)

    results = get_ablation_results(vla_model)
    if not results:
        return jsonify({
            "status": 404,
            "error": {"code": "NO_RESULTS", "message": "No ablation results found"}
        }), 404

    # Extract summary from latest experiment
    summary = []
    for exp_name, exp_data in results.items():
        ablation_data = exp_data.get("ablation", {})
        for concept, concept_results in ablation_data.items():
            if "summary" in concept_results:
                s = concept_results["summary"]
                summary.append({
                    "concept": concept,
                    "n_features": s.get("n_features_ablated", 0),
                    "concept_tasks_delta": s.get("avg_delta_concept_tasks", 0),
                    "other_tasks_delta": s.get("avg_delta_other_tasks", 0),
                    "selectivity": s.get("selectivity", 0),
                })

    return jsonify({
        "status": 200,
        "data": {
            "vla_model": vla_model,
            "summary": sorted(summary, key=lambda x: x["concept_tasks_delta"])
        }
    })


@ablation_bp.route('/api/vla/ablation/videos', methods=['GET'])
def get_ablation_videos():
    """
    Get available ablation videos.

    For Pi0.5:
        Video filename formats:
        - baseline_task{N}_{success/fail}.mp4
        - ablated_{concept}_task{N}_{success/fail}.mp4

    For OpenVLA-OFT:
        Videos at results/experiment_results/oft_concept_ablation/videos/{suite}/*.mp4
        Filename format: ablation_L{layer}_{concept_type}_{concept}_task{N}_ep{M}.mp4
    """
    vla_model = request.args.get('model', DEFAULT_VLA_MODEL)
    concept_filter = request.args.get('concept', None)
    suite = request.args.get('suite', 'libero_goal')
    limit = int(request.args.get('limit', '50'))

    config = VLA_CONFIGS.get(vla_model)

    # OpenVLA-OFT: serve from results/experiment_results/oft_concept_ablation/videos/
    if vla_model in ('openvla_oft', 'openvla') and config and 'ablation_video_dir' in config:
        ablation_video_dir = config['ablation_video_dir']
        suite_dir = ablation_video_dir / suite
        videos = []

        if suite_dir.exists():
            for video_path in sorted(suite_dir.glob("*.mp4"))[:limit]:
                filename = video_path.stem
                parts = filename.split("_")

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

                if concept_filter:
                    full_concept = f"{concept_type}/{concept_name}" if concept_type and concept_name else filename
                    if concept_filter not in full_concept and concept_filter != concept_name:
                        continue

                videos.append({
                    "path": f"/api/vla/video/oft_ablation/{suite}/{video_path.name}",
                    "filename": video_path.name,
                    "concept": f"{concept_type}/{concept_name}" if concept_type and concept_name else filename,
                    "concept_type": concept_type,
                    "layer": layer_num,
                    "task": task_num,
                    "episode": ep_num,
                    "suite": suite,
                    "is_ablated": True,
                    "success": "success" in filename,
                })

        available_suites = []
        if ablation_video_dir.exists():
            available_suites = [d.name for d in ablation_video_dir.iterdir() if d.is_dir()]

        return jsonify({
            "status": 200,
            "data": {
                "vla_model": vla_model,
                "videos": videos,
                "total_available": 11892,
                "available_suites": available_suites,
            }
        })

    # Pi0.5: serve from outputs/ablation_videos
    video_dir = Path(__file__).resolve().parents[4] / "ablation_videos"
    videos = []

    if video_dir.exists():
        for video_path in sorted(video_dir.glob("**/*.mp4"), reverse=True):
            filename = video_path.stem
            parts = filename.split("_")

            # Parse based on filename pattern
            is_ablated = parts[0] == "ablated"

            if is_ablated and len(parts) >= 4:
                # ablated_{concept}_task{N}_{success/fail}
                concept = parts[1]
                task_part = parts[2]  # "task0", "task1", etc.
                task_num = task_part.replace("task", "") if task_part.startswith("task") else "unknown"
                success = parts[3] == "success"
            elif parts[0] == "baseline" and len(parts) >= 3:
                # baseline_task{N}_{success/fail}
                concept = "baseline"
                task_part = parts[1]
                task_num = task_part.replace("task", "") if task_part.startswith("task") else "unknown"
                success = parts[2] == "success"
            else:
                # Unknown format
                concept = "unknown"
                task_num = "unknown"
                success = "success" in filename

            # Apply concept filter if provided
            if concept_filter and concept != concept_filter and concept != "baseline":
                continue

            videos.append({
                "path": f"/ablation_videos/{video_path.name}",  # URL path for frontend
                "filename": video_path.name,
                "concept": concept,
                "task": f"Task {task_num}",
                "is_ablated": is_ablated,
                "success": success,
            })

    return jsonify({
        "status": 200,
        "data": {
            "vla_model": vla_model,
            "videos": videos[:limit]
        }
    })
