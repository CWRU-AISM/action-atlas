# Aggregate X-VLA experiment results
from pathlib import Path
import json

def aggregate_xvla() -> dict:
    # Aggregate all X-VLA experiment results
    result = {
        "model": "xvla",
        "model_name": "X-VLA",
        "description": "X-VLA 1B with Florence-2 backbone, 24 TransformerBlocks, flow-matching",
        "architecture": "single_pathway",
        "params": "1B",
        "timestamp": datetime.now().isoformat(),
        "environments": ["libero", "simplerenv"],
        "baselines": {},
        "grid_ablation": {},
        "counterfactual": {},
        "cross_task": {},
        "vision_perturbation": {},
        "concept_ablation": {},
        "concept_steering": {},
        "displacement": {},
    }

    # --- LIBERO Baselines (from grid ablation baseline condition) ---
    for suite in XVLA_LIBERO_SUITES:
        grid_path = XVLA_LIBERO_DIR / "experiments" / f"grid_ablation_{suite}" / "grid_results.json"
        data = safe_load_json(grid_path)
        if data:
            baseline_grid = data.get("grid", {}).get("baseline", {})
            tasks_summary = {}
            for task_id, per_task in baseline_grid.get("per_task", {}).items():
                tasks_summary[task_id] = {
                    "success_rate": round_val(per_task.get("success_rate", 0)),
                    "n_episodes": per_task.get("n_episodes", 0),
                    "task_description": data.get("task_prompts", {}).get(task_id, ""),
                }
            overall = (
                sum(t["success_rate"] for t in tasks_summary.values()) / max(len(tasks_summary), 1)
                if tasks_summary else 0
            )
            result["baselines"][suite] = {
                "environment": "libero",
                "suite": suite,
                "tasks": tasks_summary,
                "overall_success_rate": round_val(overall),
            }

    # --- LIBERO Grid Ablation ---
    for suite in XVLA_LIBERO_SUITES:
        grid_path = XVLA_LIBERO_DIR / "experiments" / f"grid_ablation_{suite}" / "grid_results.json"
        data = safe_load_json(grid_path)
        if data:
            result["grid_ablation"][suite] = _extract_xvla_grid(data)

    # --- LIBERO Counterfactual ---
    for suite in XVLA_LIBERO_SUITES:
        # Try v2 first, then plain
        cf_variant = f"counterfactual_{suite}_v2" if suite != "libero_10" else f"counterfactual_{suite}"
        cf_dir = XVLA_LIBERO_DIR / "experiments" / cf_variant
        if not cf_dir.is_dir():
            cf_dir = XVLA_LIBERO_DIR / "experiments" / f"counterfactual_{suite}"
        if cf_dir.is_dir():
            result["counterfactual"][suite] = _aggregate_xvla_counterfactual(cf_dir, suite)

    # --- LIBERO Cross-Task ---
    for suite in XVLA_LIBERO_SUITES:
        ct_path = XVLA_LIBERO_DIR / "experiments" / f"cross_task_{suite}" / "results.json"
        data = safe_load_json(ct_path)
        if data:
            result["cross_task"][suite] = _extract_xvla_cross_task(data)

    # --- LIBERO Vision Perturbation ---
    for suite in XVLA_LIBERO_SUITES:
        vision_dir = XVLA_LIBERO_DIR / "experiments" / f"vision_{suite}"
        if vision_dir.is_dir():
            result["vision_perturbation"][suite] = _aggregate_xvla_vision(vision_dir, suite)

    # --- Concept Ablation ---
    for suite_short in ["goal", "object", "spatial", "10"]:
        for layer in [12, 20, 23]:
            abl_path = XVLA_CONCEPT_ABLATION_DIR / f"ablation_L{layer}_{suite_short}.json"
            data = safe_load_json(abl_path)
            if data:
                key = f"L{layer}_{suite_short}"
                result["concept_ablation"][key] = _extract_xvla_concept_ablation(data)

    # --- Concept Steering ---
    for json_file in sorted(XVLA_CONCEPT_STEERING_DIR.glob("*.json")) if XVLA_CONCEPT_STEERING_DIR.is_dir() else []:
        data = safe_load_json(json_file)
        if data:
            key = json_file.stem
            result["concept_steering"][key] = _extract_xvla_concept_steering(data)

    # --- Displacement Analysis ---
    disp_path = Path("/data/batch_1/xvla_displacement_analysis.json")
    data = safe_load_json(disp_path)
    if data:
        result["displacement"] = _extract_xvla_displacement(data)

    # --- SimplerEnv (check for experiments) ---
    if XVLA_SIMPLERENV_DIR.is_dir():
        se_experiments = XVLA_SIMPLERENV_DIR / "experiments"
        if se_experiments.is_dir():
            for exp_dir in sorted(se_experiments.iterdir()):
                if exp_dir.is_dir():
                    results_path = exp_dir / "results.json"
                    if results_path.exists():
                        se_data = safe_load_json(results_path)
                        if se_data and "simplerenv" not in result:
                            result["simplerenv"] = {}
                        if se_data:
                            result["simplerenv"][exp_dir.name] = {
                                "n_episodes": se_data.get("n_episodes", 0),
                                "status": "available",
                            }

    return result


def _extract_xvla_grid(data: dict) -> dict:
    grid = data.get("grid", {})
    n_blocks = data.get("n_blocks", 24)

    summary = {
        "suite": data.get("suite", ""),
        "ablation_mode": data.get("ablation_mode", "zero"),
        "n_episodes": data.get("n_episodes", 0),
        "n_blocks": n_blocks,
        "baseline": {},
        "per_layer": {},
    }

    # Baseline
    baseline = grid.get("baseline", {})
    per_task_baseline = baseline.get("per_task", {})
    for task_id, tdata in per_task_baseline.items():
        summary["baseline"][task_id] = round_val(tdata.get("success_rate", 0))
    if per_task_baseline:
        summary["baseline_overall"] = round_val(
            sum(t.get("success_rate", 0) for t in per_task_baseline.values())
            / max(len(per_task_baseline), 1)
        )

    # Per-layer ablation
    for layer_key, layer_data in grid.items():
        if layer_key == "baseline":
            continue
        per_task = layer_data.get("per_task", {})
        task_srs = {}
        for task_id, tdata in per_task.items():
            task_srs[task_id] = round_val(tdata.get("success_rate", 0))
        if task_srs:
            overall = sum(task_srs.values()) / max(len(task_srs), 1)
            summary["per_layer"][layer_key] = {
                "overall_success_rate": round_val(overall),
                "per_task": task_srs,
            }

    return summary


def _aggregate_xvla_counterfactual(cf_dir: Path, suite: str) -> dict:
    # Aggregate X-VLA counterfactual results from per-task directories
    summary = {
        "environment": "libero",
        "suite": suite,
        "per_task": {},
        "categories": defaultdict(lambda: {"n_episodes": 0, "successes": 0}),
    }

    for task_dir in sorted(cf_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        results_path = task_dir / "results.json"
        data = safe_load_json(results_path)
        if not data:
            continue

        task_id = str(data.get("task_id", task_dir.name.replace("task_", "")))
        task_summary = {
            "task_description": data.get("task_description", ""),
            "conditions": {},
        }

        for cond_name, cond_data in data.get("conditions", {}).items():
            cond_summary = {
                "prompt": cond_data.get("prompt", ""),
                "success_rate": round_val(cond_data.get("success_rate", 0)),
                "n_episodes": cond_data.get("n_episodes", 0),
                "avg_steps": round_val(cond_data.get("avg_steps", 0)),
            }
            # Include cosine similarity if available
            if "cosine_to_baseline" in cond_data:
                cond_summary["cosine_to_baseline"] = round_val(cond_data["cosine_to_baseline"])
            if "xyz_diff_to_baseline" in cond_data:
                cond_summary["xyz_diff_to_baseline"] = round_val(cond_data["xyz_diff_to_baseline"])
            task_summary["conditions"][cond_name] = cond_summary

            # Aggregate into category-level stats
            cat = _categorize_condition(cond_name)
            summary["categories"][cat]["n_episodes"] += cond_data.get("n_episodes", 0)
            summary["categories"][cat]["successes"] += int(
                cond_data.get("success_rate", 0) * cond_data.get("n_episodes", 0)
            )

        summary["per_task"][task_id] = task_summary

    # Finalize category stats
    cat_final = {}
    for cat, cdata in summary["categories"].items():
        n = cdata["n_episodes"]
        cat_final[cat] = {
            "n_episodes": n,
            "success_rate": round_val(cdata["successes"] / max(n, 1)),
        }
    summary["categories"] = cat_final

    return summary


def _categorize_condition(cond_name: str) -> str:
    # Map a condition name to a category
    if cond_name == "baseline":
        return "baseline"
    elif cond_name == "null":
        return "null"
    elif cond_name.startswith("neg_") or cond_name.startswith("negate"):
        return "negation"
    elif cond_name.startswith("motor_"):
        return "motor"
    elif cond_name.startswith("cross") or cond_name.startswith("swap"):
        return "cross_prompt"
    return "other"


def _extract_xvla_cross_task(data: dict) -> dict:
    # Extract X-VLA cross-task injection summary
    task_pairs = data.get("task_pairs", [])
    pairs_data = data.get("pairs", {})

    pair_summaries = {}
    for pair_key, pair_data in pairs_data.items():
        if not isinstance(pair_data, dict):
            continue
        # Extract success info, skip raw trajectories
        pair_info = {}
        for cond_name, cond_data in pair_data.items():
            if isinstance(cond_data, dict) and "success" in cond_data:
                pair_info[cond_name] = {
                    "success": cond_data.get("success"),
                    "n_steps": cond_data.get("n_steps"),
                }
                if "cosine_to_source" in cond_data:
                    pair_info[cond_name]["cosine_to_source"] = round_val(cond_data["cosine_to_source"])
                if "cosine_to_target" in cond_data:
                    pair_info[cond_name]["cosine_to_target"] = round_val(cond_data["cosine_to_target"])
        if pair_info:
            pair_summaries[pair_key] = pair_info

    return {
        "model": data.get("model", "xvla"),
        "suite": data.get("suite", ""),
        "n_task_pairs": len(task_pairs),
        "n_processed_pairs": len(pair_summaries),
        "pairs": pair_summaries,
    }


def _aggregate_xvla_vision(vision_dir: Path, suite: str) -> dict:
    # Aggregate X-VLA vision perturbation from per-task directories
    per_perturbation = defaultdict(lambda: {"successes": 0, "total": 0, "per_task": {}})
    all_perturbations = set()

    for task_dir in sorted(vision_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        results_path = task_dir / "results.json"
        data = safe_load_json(results_path)
        if not data:
            continue

        task_id = str(data.get("task", task_dir.name.replace("task_", "")))
        for result_entry in data.get("results", []):
            pert = result_entry.get("perturbation", "unknown")
            all_perturbations.add(pert)
            success = result_entry.get("success", False)
            per_perturbation[pert]["total"] += 1
            if success:
                per_perturbation[pert]["successes"] += 1
            per_perturbation[pert]["per_task"][task_id] = success

    summary = {
        "environment": "libero",
        "suite": suite,
        "perturbations": sorted(all_perturbations),
        "per_perturbation": {},
    }
    for pert, pdata in per_perturbation.items():
        summary["per_perturbation"][pert] = {
            "success_rate": round_val(pdata["successes"] / max(pdata["total"], 1)),
            "n_episodes": pdata["total"],
            "per_task": pdata["per_task"],
        }

    return summary


def _extract_xvla_concept_ablation(data: dict) -> dict:
    """
    Extract X-VLA concept ablation summary (skip raw feature lists).

    Data structure:
        tasks: {
            concept_name: {
                features: [...],
                scores: [...],
                tasks: {
                    task_id: { success_rate, delta, successes, steps }
                }
            }
        }
    """
    concepts = data.get("tasks", {})
    concept_summaries = {}

    for concept_name, concept_data in concepts.items():
        if not isinstance(concept_data, dict):
            continue
        # Per-task ablation results are under "tasks" sub-key
        per_task_results = concept_data.get("tasks", {})
        results_summary = {}
        for task_idx, task_result in per_task_results.items():
            if isinstance(task_result, dict):
                results_summary[task_idx] = {
                    "success_rate": round_val(task_result.get("success_rate", 0)),
                    "delta": round_val(task_result.get("delta", 0)),
                }
        concept_summaries[concept_name] = {
            "n_features": len(concept_data.get("features", [])),
            "per_task": results_summary,
        }

    return {
        "layer": data.get("layer"),
        "suite": data.get("suite"),
        "mode": data.get("mode"),
        "n_episodes": data.get("n_episodes", 0),
        "top_n_features": data.get("top_n_features", 0),
        "concepts": concept_summaries,
    }


def _extract_xvla_concept_steering(data: dict) -> dict:
    """
    Extract X-VLA concept steering summary.

    Data structure:
        tasks: {
            concept_name: {
                features: [...],
                scores: [...],
                strengths: {
                    strength_val: {
                        task_id: { success_rate, delta, successes }
                    }
                }
            }
        }
    """
    concepts = data.get("tasks", {})
    concept_summaries = {}

    for concept_name, concept_data in concepts.items():
        if not isinstance(concept_data, dict):
            continue
        per_strength = {}
        concept_strengths = concept_data.get("strengths", {})
        for strength_val, s_data in concept_strengths.items():
            if not isinstance(s_data, dict):
                continue
            # s_data = { task_id: { success_rate, delta } }
            task_results = {}
            total_sr = 0
            n_tasks = 0
            for task_id, tdata in s_data.items():
                if isinstance(tdata, dict):
                    sr = tdata.get("success_rate", 0)
                    task_results[task_id] = {
                        "success_rate": round_val(sr),
                        "delta": round_val(tdata.get("delta", 0)),
                    }
                    total_sr += sr
                    n_tasks += 1
            per_strength[str(strength_val)] = {
                "overall_success_rate": round_val(total_sr / max(n_tasks, 1)),
                "n_tasks": n_tasks,
                "per_task": task_results,
            }
        concept_summaries[concept_name] = {
            "n_features": len(concept_data.get("features", [])),
            "per_strength": per_strength,
        }

    return {
        "layer": data.get("layer"),
        "suite": data.get("suite"),
        "mode": data.get("mode"),
        "strengths": data.get("strengths", []),
        "n_episodes": data.get("n_episodes", 0),
        "concepts": concept_summaries,
    }


def _extract_xvla_displacement(data: dict) -> dict:
    summary = {}
    for suite_key, suite_data in data.items():
        if not isinstance(suite_data, dict):
            continue
        per_condition = {}
        for cond_name, cond_data in suite_data.get("per_condition", {}).items():
            per_condition[cond_name] = {
                "total": cond_data.get("total", 0),
                "source_behavior": cond_data.get("source_behavior", 0),
                "destination_behavior": cond_data.get("destination_behavior", 0),
                "ambiguous": cond_data.get("ambiguous", 0),
                "successes": cond_data.get("successes", 0),
                "mean_cos_to_src": round_val(cond_data.get("mean_cos_to_src", 0)),
                "mean_cos_to_dst": round_val(cond_data.get("mean_cos_to_dst", 0)),
                "override_rate": round_val(
                    cond_data.get("source_behavior", 0) / max(cond_data.get("total", 1), 1)
                ),
            }
        summary[suite_key] = {
            "total_episodes": suite_data.get("total_episodes", 0),
            "injection_episodes": suite_data.get("injection_episodes", 0),
            "per_condition": per_condition,
        }
    return summary
# GR00T aggregation
GROOT_MAIN_DIR = Path("/data/groot_rollouts")
GROOT_BATCH2_DIR = Path("/data/groot_rollouts_batch2")
GROOT_SUITES = ["libero_goal", "libero_object", "libero_long"]
GROOT_DIT_LAYERS = [f"dit_L{i:02d}" for i in range(16)]
GROOT_EAGLE_LAYERS = [f"eagle_lm_L{i:02d}" for i in range(12)]
GROOT_VLSA_LAYERS = [f"vl_sa_L{i:02d}" for i in range(4)]
GROOT_ALL_LAYERS = GROOT_DIT_LAYERS + GROOT_EAGLE_LAYERS + GROOT_VLSA_LAYERS


