"""Aggregate GR00T experiment results."""
from pathlib import Path
import json

def aggregate_groot() -> dict:
    """Aggregate all GR00T experiment results."""
    result = {
        "model": "groot",
        "model_name": "GR00T N1.5",
        "description": "GR00T N1.5 3B with DiT + Eagle LM + VL-SA triple-pathway",
        "architecture": "triple_pathway",
        "params": "3B",
        "timestamp": datetime.now().isoformat(),
        "environments": ["libero"],
        "baselines": {},
        "grid_ablation": {},
        "counterfactual": {},
        "cross_task": {},
        "vision_perturbation": {},
        "fraction_to_failure": {},
        "steering": {},
        "temporal_ablation": {},
        "cross_suite_ablation": {},
    }

    for suite in GROOT_SUITES:
        suite_dir = GROOT_MAIN_DIR / suite

        # --- Baselines ---
        baseline_path = suite_dir / "baseline" / "results.json"
        data = safe_load_json(baseline_path)
        if data:
            tasks_summary = {}
            for task_key, task_data in data.get("tasks", {}).items():
                episodes = task_data.get("episodes", [])
                tasks_summary[task_key] = {
                    "task_description": task_data.get("task_description", ""),
                    "success_rate": round_val(compute_success_rate(episodes)),
                    "n_episodes": len(episodes),
                    "avg_steps": round_val(
                        sum(ep.get("steps", 0) for ep in episodes) / max(len(episodes), 1)
                    ),
                }
            result["baselines"][suite] = {
                "environment": "libero",
                "suite": suite,
                "n_episodes": data.get("n_episodes", 0),
                "tasks": tasks_summary,
                "overall_success_rate": round_val(
                    sum(t["success_rate"] for t in tasks_summary.values()) / max(len(tasks_summary), 1)
                ),
            }

        # --- Grid Ablation ---
        grid_path = suite_dir / "grid_ablation" / "grid_results.json"
        data = safe_load_json(grid_path)
        if data:
            result["grid_ablation"][suite] = _extract_groot_grid(data)

        # --- Counterfactual ---
        cf_dir = suite_dir / "counterfactual"
        if cf_dir.is_dir():
            result["counterfactual"][suite] = _aggregate_groot_counterfactual(cf_dir, suite)

        # --- Cross-Task ---
        ct_path = suite_dir / "cross_task" / "results.json"
        data = safe_load_json(ct_path)
        if data:
            result["cross_task"][suite] = _extract_groot_cross_task(data)

        # --- Visual Perturbation ---
        vp_dir = suite_dir / "visual_perturbation"
        if vp_dir.is_dir():
            result["vision_perturbation"][suite] = _aggregate_groot_vision(vp_dir, suite)

    # --- Fraction to Failure (batch2) ---
    ftf_dir = GROOT_BATCH2_DIR / "sae_fraction_to_failure"
    if ftf_dir.is_dir():
        for suite in GROOT_SUITES:
            suite_ftf_dir = ftf_dir / suite
            if suite_ftf_dir.is_dir():
                result["fraction_to_failure"][suite] = _aggregate_groot_ftf(suite_ftf_dir, suite)

    # --- Steering (batch2) ---
    steer_dir = GROOT_BATCH2_DIR / "sae_steering"
    if steer_dir.is_dir():
        for suite in GROOT_SUITES:
            suite_steer_dir = steer_dir / suite
            if suite_steer_dir.is_dir():
                result["steering"][suite] = _aggregate_groot_steering(suite_steer_dir, suite)

    # --- Temporal Ablation (batch2) ---
    temp_dir = GROOT_BATCH2_DIR / "sae_temporal_ablation"
    if temp_dir.is_dir():
        for suite in GROOT_SUITES:
            suite_temp_dir = temp_dir / suite
            if suite_temp_dir.is_dir():
                result["temporal_ablation"][suite] = _aggregate_groot_temporal(suite_temp_dir, suite)

    # --- Cross-Suite Ablation (batch2) ---
    cs_dir = GROOT_BATCH2_DIR / "sae_cross_suite_ablation"
    if cs_dir.is_dir():
        for json_file in sorted(cs_dir.glob("*.json")):
            data = safe_load_json(json_file)
            if data:
                result["cross_suite_ablation"][json_file.stem] = data

    return result


def _extract_groot_grid(data: dict) -> dict:
    grid = data.get("grid", {})
    n_blocks = data.get("n_blocks", 16)

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


def _aggregate_groot_counterfactual(cf_dir: Path, suite: str) -> dict:
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

        task_id = str(data.get("task_idx", task_dir.name.replace("task_", "")))
        task_summary = {
            "task_description": data.get("task_description", ""),
            "conditions": {},
        }

        for cond_name, cond_data in data.get("conditions", {}).items():
            if not isinstance(cond_data, dict):
                continue
            episodes = cond_data.get("episodes", [])
            n_ep = len(episodes)
            n_success = sum(1 for ep in episodes if ep.get("success", False))
            sr = n_success / max(n_ep, 1)

            cond_summary = {
                "prompt": cond_data.get("prompt", ""),
                "success_rate": round_val(sr),
                "n_episodes": n_ep,
            }
            task_summary["conditions"][cond_name] = cond_summary

            cat = _categorize_condition(cond_name)
            summary["categories"][cat]["n_episodes"] += n_ep
            summary["categories"][cat]["successes"] += n_success

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


def _extract_groot_cross_task(data: dict) -> dict:
    """Extract GR00T cross-task injection summary."""
    task_pairs = data.get("task_pairs", [])
    pairs_data = data.get("pairs", {})

    pair_summaries = {}
    for pair_key, pair_data in pairs_data.items():
        if not isinstance(pair_data, dict):
            continue
        pair_info = {}
        for cond_name, cond_data in pair_data.items():
            if isinstance(cond_data, dict) and "success" in cond_data:
                pair_info[cond_name] = {
                    "success": cond_data.get("success"),
                    "n_steps": cond_data.get("n_steps"),
                }
                if "cosine_to_source" in cond_data:
                    pair_info[cond_name]["cosine_to_source"] = round_val(cond_data["cosine_to_source"])
        if pair_info:
            pair_summaries[pair_key] = pair_info

    return {
        "model": data.get("model", "groot"),
        "suite": data.get("suite", ""),
        "n_task_pairs": len(task_pairs),
        "n_processed_pairs": len(pair_summaries),
        "pairs": pair_summaries,
    }


def _aggregate_groot_vision(vision_dir: Path, suite: str) -> dict:
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


def _aggregate_groot_ftf(suite_dir: Path, suite: str) -> dict:
    """Aggregate GR00T fraction-to-failure results across layers."""
    per_layer = {}
    for layer_dir in sorted(suite_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        layer_name = layer_dir.name
        layer_results = {}
        for category in ["frequent", "random", "universal"]:
            json_path = layer_dir / f"fraction_{category}.json"
            data = safe_load_json(json_path)
            if data:
                titration = data.get("titration", [])
                titration_summary = []
                for entry in titration:
                    titration_summary.append({
                        "n_features": entry.get("n_features", 0),
                        "success_rate": round_val(entry.get("success_rate", 0)),
                        "successes": entry.get("successes", 0),
                        "total": entry.get("total", 0),
                    })
                layer_results[category] = {
                    "n_available_features": data.get("n_available_features", 0),
                    "baseline_success_rate": round_val(
                        data.get("baseline", {}).get("success_rate", 1.0)
                    ),
                    "titration": titration_summary,
                }
        if layer_results:
            per_layer[layer_name] = layer_results

    return {
        "suite": suite,
        "n_layers": len(per_layer),
        "per_layer": per_layer,
    }


def _aggregate_groot_steering(suite_dir: Path, suite: str) -> dict:
    """
    Aggregate GR00T steering results across layers.

    Data structure is:
        baseline: { task_id: { task_desc, success_rate, successes } }
        groups: {
            group_name: {
                features: [...],
                n_features: N,
                strengths: {
                    strength: {
                        tasks: { task_id: { success_rate, delta, successes } }
                    }
                }
            }
        }
    """
    per_layer = {}
    for layer_dir in sorted(suite_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        layer_name = layer_dir.name
        json_path = layer_dir / "steering_results.json"
        data = safe_load_json(json_path)
        if not data:
            continue

        strengths = data.get("strengths", [])
        baseline = data.get("baseline", {})
        groups = data.get("groups", {})

        # Summarize baseline
        baseline_summary = {}
        for task_id, tdata in baseline.items():
            if isinstance(tdata, dict):
                baseline_summary[task_id] = {
                    "task_desc": tdata.get("task_desc", ""),
                    "success_rate": round_val(tdata.get("success_rate", 0)),
                }

        # Summarize steering per group and strength
        group_summary = {}
        for group_name, group_data in groups.items():
            if not isinstance(group_data, dict):
                continue
            per_strength = {}
            group_strengths = group_data.get("strengths", {})
            for strength_val, s_data in group_strengths.items():
                if not isinstance(s_data, dict):
                    continue
                tasks = s_data.get("tasks", {})
                task_srs = {}
                total_sr = 0
                n_tasks = 0
                for task_id, tdata in tasks.items():
                    if isinstance(tdata, dict):
                        sr = tdata.get("success_rate", 0)
                        task_srs[task_id] = {
                            "success_rate": round_val(sr),
                            "delta": round_val(tdata.get("delta", 0)),
                        }
                        total_sr += sr
                        n_tasks += 1
                per_strength[str(strength_val)] = {
                    "overall_success_rate": round_val(total_sr / max(n_tasks, 1)),
                    "n_tasks": n_tasks,
                    "per_task": task_srs,
                }
            group_summary[group_name] = {
                "n_features": group_data.get("n_features", 0),
                "per_strength": per_strength,
            }

        per_layer[layer_name] = {
            "strengths": strengths,
            "baseline": baseline_summary,
            "groups": group_summary,
        }

    return {
        "suite": suite,
        "n_layers": len(per_layer),
        "per_layer": per_layer,
    }


def _aggregate_groot_temporal(suite_dir: Path, suite: str) -> dict:
    """Aggregate GR00T temporal ablation results across layers."""
    per_layer = {}
    for layer_dir in sorted(suite_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        layer_name = layer_dir.name
        layer_results = {}
        for category in ["frequent", "universal"]:
            json_path = layer_dir / f"temporal_{category}.json"
            data = safe_load_json(json_path)
            if data:
                windows = data.get("windows", {})
                windows_summary = {}
                for window_name, w_data in windows.items():
                    if isinstance(w_data, dict):
                        windows_summary[window_name] = {
                            "start": w_data.get("start", 0),
                            "end": w_data.get("end", 0),
                            "success_rate": round_val(w_data.get("success_rate", 0)),
                            "successes": w_data.get("successes", 0),
                            "total": w_data.get("total", 0),
                        }
                layer_results[category] = {
                    "n_features": data.get("n_features", 0),
                    "baseline_success_rate": round_val(
                        data.get("baseline", {}).get("success_rate", 1.0)
                    ),
                    "windows": windows_summary,
                }
        if layer_results:
            per_layer[layer_name] = layer_results

    return {
        "suite": suite,
        "n_layers": len(per_layer),
        "per_layer": per_layer,
    }
# Main
AGGREGATORS = {
    "smolvla": aggregate_smolvla,
    "xvla": aggregate_xvla,
    "groot": aggregate_groot,
}


def main():
    parser = argparse.ArgumentParser(description="Aggregate VLA experiment results")
    parser.add_argument(
        "--models", nargs="+", default=list(AGGREGATORS.keys()),
        help="Models to aggregate (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent.parent / "data"),
        help="Output directory for aggregated JSON files"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        if model not in AGGREGATORS:
            print(f"WARNING: Unknown model '{model}', skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Aggregating {model}...")
        print(f"{'='*60}")

        try:
            data = AGGREGATORS[model]()
        except Exception as e:
            print(f"ERROR aggregating {model}: {e}")
            import traceback
            traceback.print_exc()
            continue

        output_path = output_dir / f"experiment_results_{model}.json"
        indent = 2 if args.pretty else None
        with open(output_path, "w") as f:
            json.dump(data, f, indent=indent, default=str)

        # Print summary stats
        file_size = output_path.stat().st_size
        _print_summary(model, data, file_size)

    print(f"\nDone. Output files in: {output_dir}")


def _print_summary(model: str, data: dict, file_size: int):
    """Print a summary of what was aggregated."""
    print(f"\n  Model: {data.get('model_name', model)}")
    print(f"  File size: {file_size / 1024:.1f} KB")

    sections = [
        "baselines", "grid_ablation", "counterfactual", "cross_task",
        "vision_perturbation", "displacement", "concept_ablation",
        "concept_steering", "fraction_to_failure", "steering",
        "temporal_ablation", "cross_suite_ablation",
    ]
    for section in sections:
        sdata = data.get(section, {})
        if sdata:
            if isinstance(sdata, dict):
                n_keys = len(sdata)
                print(f"  {section}: {n_keys} entries")
            else:
                print(f"  {section}: present")


if __name__ == "__main__":
    main()
