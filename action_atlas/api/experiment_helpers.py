# Experiment data loading helpers
from .helpers import *
from .data_loaders import *

def _count_files(directory: Path, pattern: str) -> int:
    # Count files matching a glob pattern, returning 0 if directory is missing
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def _rcount_files(directory: Path, pattern: str) -> int:
    # Recursively count files matching a glob pattern, returning 0 if missing
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob(pattern))


def _dir_experiment_entry(directory: Path, pattern: str, description: str,
                          category: str, recursive: bool = False) -> Optional[dict]:
    # Build an experiment entry by scanning a directory, or None if empty
    count = _rcount_files(directory, pattern) if recursive else _count_files(directory, pattern)
    if count == 0:
        return None
    return {'count': count, 'description': description, 'category': category}


def _load_experiment_results(model: str) -> Optional[Dict]:
    # Load pre-aggregated experiment results for a model
    if model in _experiment_results_cache:
        return _experiment_results_cache[model]
    results_path = _API_DATA_DIR / f"experiment_results_{model}.json"
    data = load_json_cached(results_path, f"experiment_results_{model}")
    if data is not None:
        _experiment_results_cache[model] = data
    return data


def _resolve_file_model(model: str) -> str:
    # Map a model param name to the file suffix for experiment_results_<name>.json
    return MODEL_FILE_MAP.get(model, model)


def _load_known_stats() -> Dict:
    """
    Load verified episode counts from experiment_stats.json.

    Flattens the 'sections' nesting for backward compatibility.
    Returns empty dict if the file is missing.
    """
    stats_path = _API_DATA_DIR / 'experiment_stats.json'
    raw = load_json_cached(stats_path, "experiment_stats")
    if not raw:
        return {}

    result = {}
    for model_key, model_val in raw.items():
        if not isinstance(model_val, dict) or 'total' not in model_val:
            continue
        flat = {
            'total': model_val['total'],
            'label': model_val.get('label', str(model_val['total'])),
            'overall_success_rate': model_val.get('overall_success_rate'),
            'baseline_success_rate': model_val.get('baseline_success_rate'),
        }
        for sec_key, sec_val in model_val.get('sections', {}).items():
            flat[sec_key] = sec_val
        result[model_key] = flat

    if 'openvla' not in result and 'oft' in result:
        result['openvla'] = result['oft']
    return result


_OFT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "oft_manifest.json"
_OFT_MANIFEST: Optional[Dict[str, Dict[str, str]]] = None


def _load_oft_manifest() -> Dict[str, Dict[str, str]]:
    """
    Read the baked OFT manifest produced by ``scripts/build_oft_manifest.py``.

    The manifest maps suite -> experiment_type -> canonical run directory name.
    Loaded once and cached.
    """
    global _OFT_MANIFEST
    if _OFT_MANIFEST is None:
        payload = load_json_cached(_OFT_MANIFEST_PATH, "oft_manifest") or {}
        _OFT_MANIFEST = payload.get("models", {}).get("openvla", {})
    return _OFT_MANIFEST


def _find_latest_oft_result(suite: str, experiment_type: str) -> Optional[Path]:
    """
    Resolve the canonical OFT results.json path for ``(suite, experiment_type)``.

    Returns ``None`` when the manifest has no entry for the request, which is the
    one legitimate "not found" case. The caller surfaces a clean 404.
    """
    run_name = _load_oft_manifest().get(suite, {}).get(experiment_type)
    if run_name is None:
        return None
    return OFT_DATA_DIR / suite / run_name / experiment_type / "results.json"


def _scan_dirs_to_experiment_types(experiment_types: dict, specs: list) -> None:
    """
    Scan directories and populate experiment_types dict.

    Each spec is (directory, pattern, recursive, key, suites, description).
    """
    for directory, pattern, recursive, key, suites, description in specs:
        count = _rcount_files(directory, pattern) if recursive else _count_files(directory, pattern)
        if count > 0:
            experiment_types[key] = {'count': count, 'suites': suites, 'description': description}


def _parse_ablation_video_filename(stem: str) -> dict:
    """
    Parse ablation video filename stem into components.

    Format: ablation_L{layer}_{concept_type}_{concept}_task{N}_ep{M}
    """
    parts = stem.split("_")
    result = {'layer_num': None, 'concept_type': None, 'concept_name': None,
              'task_num': None, 'ep_num': None}

    if len(parts) < 6 or parts[0] != 'ablation':
        return result

    if parts[1].startswith('L'):
        try:
            result['layer_num'] = int(parts[1][1:])
        except ValueError:
            pass

    result['concept_type'] = parts[2]

    task_idx = None
    for i, p in enumerate(parts):
        if p.startswith('task'):
            task_idx = i
            try:
                result['task_num'] = int(p.replace('task', ''))
            except ValueError:
                pass
        elif p.startswith('ep'):
            try:
                result['ep_num'] = int(p.replace('ep', ''))
            except ValueError:
                pass

    if task_idx and task_idx > 3:
        result['concept_name'] = '_'.join(parts[3:task_idx])
    elif task_idx:
        result['concept_name'] = parts[3] if len(parts) > 3 else None

    return result
# Layer Metrics
def _build_layer_connections_from_config(model: str, suite: str) -> dict:
    # Build layer connection data from model config and concept_id files
    config = get_vla_config(model)
    layers = []
    layer_concept_counts = _load_concept_counts_for_model(model, suite, config)

    for i, layer_name in enumerate(config['layers']):
        counts = layer_concept_counts.get(layer_name, {'motion': 0, 'object': 0, 'spatial': 0, 'top_concepts': []})
        total = counts['motion'] + counts['object'] + counts['spatial']
        if total > 0:
            dominant = max([('motion', counts['motion']), ('object', counts['object']), ('spatial', counts['spatial'])], key=lambda x: x[1])[0]
        else:
            dominant = 'none'
        layers.append({
            'id': f'{layer_name}-{suite}', 'type': 'RES', 'layer': i, 'layer_name': layer_name,
            'total_motion': {'value': counts['motion'], 'rank': 1},
            'total_object': {'value': counts['object'], 'rank': 1},
            'total_spatial': {'value': counts['spatial'], 'rank': 1},
            'dominant_type': dominant,
            'top_concepts': counts.get('top_concepts', [])[:5],
        })

    for cat in ['total_motion', 'total_object', 'total_spatial']:
        sorted_indices = sorted(range(len(layers)), key=lambda idx: layers[idx][cat]['value'], reverse=True)
        for rank, idx in enumerate(sorted_indices, 1):
            layers[idx][cat]['rank'] = rank

    return {'layers': layers, 'model': model, 'suite': suite}
# OFT Ablation Videos
def _get_groot_temporal_ablation():
    # Load GR00T temporal ablation from experiment_results_groot.json
    data = _load_experiment_results('groot')
    if data is None or 'temporal_ablation' not in data:
        return jsonify({'status': 404, 'error': {'code': 'NO_DATA', 'message': 'No temporal ablation data found for GR00T.'}}), 404

    ta = data['temporal_ablation']
    windows = ['all', 'early', 'mid', 'late', 'first_quarter', 'last_quarter']
    suites_out = {}

    for suite_key, suite_data in ta.items():
        if not isinstance(suite_data, dict) or 'per_layer' not in suite_data:
            continue
        per_layer_in = suite_data['per_layer']
        per_layer_out = {}

        for layer_name, layer_data in per_layer_in.items():
            baseline_vals = []
            window_rates = {w: [] for w in windows}
            for feat_type, feat_data in layer_data.items():
                if not isinstance(feat_data, dict) or 'windows' not in feat_data:
                    continue
                baseline_vals.append(feat_data.get('baseline_success_rate', 1.0))
                for w in windows:
                    sr = feat_data['windows'].get(w, {}).get('success_rate')
                    if sr is not None:
                        window_rates[w].append(sr)

            baseline = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 1.0
            windows_out = {}
            for w in windows:
                rates = window_rates[w]
                if rates:
                    avg_sr = sum(rates) / len(rates)
                    windows_out[w] = {'success_rate': round(avg_sr, 4), 'delta': round((avg_sr - baseline) * 100, 1)}
                else:
                    windows_out[w] = {'success_rate': None, 'delta': None}
            per_layer_out[layer_name] = {'baseline': round(baseline, 4), 'windows': windows_out}

        suites_out[suite_key] = {'layers': list(per_layer_in.keys()), 'per_layer': per_layer_out}

    return jsonify({'status': 200, 'data': {'model': 'groot', 'model_name': 'GR00T N1.5', 'available': True, 'windows': windows, 'suites': suites_out}})


def _get_smolvla_temporal_ablation():
    # Aggregate SmolVLA temporal ablation from concept ablation result files
    baked_path = _API_DATA_DIR / 'smolvla_temporal_ablation.json'
    baked = load_json_cached(baked_path, "smolvla_temporal_ablation")
    if baked is not None:
        return jsonify({'status': 200, 'data': baked})

    results_dir = Path("/data/smolvla_rollouts/smolvla/concept_ablation/results")
    if not results_dir.exists():
        return jsonify({'status': 404, 'error': {'code': 'NO_DATA', 'message': 'SmolVLA concept ablation results directory not found.'}}), 404

    windows = ['early', 'mid', 'late', 'full']
    concept_types = ['motion', 'object', 'spatial']
    suites_out = {}

    for fpath in sorted(results_dir.glob("*.json")):
        try:
            with open(fpath) as f:
                d = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if 'temporal' not in d:
            continue

        component = d.get('component', 'expert')
        suite = d.get('suite', 'unknown')
        layer_raw = d.get('layer', 'unknown')
        layer_name = f"{component}_L{layer_raw:02d}" if isinstance(layer_raw, int) else f"{component}_{layer_raw}"
        baseline_overall = d.get('baseline_overall', 1.0)

        if suite not in suites_out:
            suites_out[suite] = {'layers': [], 'per_layer': {}}
        if layer_name not in suites_out[suite]['per_layer']:
            suites_out[suite]['layers'].append(layer_name)

        temporal = d['temporal']
        window_rates = {w: [] for w in windows}
        for ctype in concept_types:
            cdata = temporal.get(ctype, {})
            w_data = cdata.get('windows', cdata)
            for w in windows:
                if w in w_data:
                    w_info = w_data[w]
                    if isinstance(w_info, dict):
                        task_results = w_info.get('task_results', {})
                        if task_results:
                            rates = [t.get('rate', 0) for t in task_results.values()]
                            window_rates[w].append(sum(rates) / len(rates) if rates else 0)
                        elif 'success_rate' in w_info:
                            window_rates[w].append(w_info['success_rate'])
                        elif 'mean_delta' in w_info:
                            window_rates[w].append(baseline_overall + w_info['mean_delta'])

        windows_out = {}
        for w in windows:
            rates = window_rates[w]
            if rates:
                avg_sr = sum(rates) / len(rates)
                windows_out[w] = {'success_rate': round(avg_sr, 4), 'delta': round((avg_sr - baseline_overall) * 100, 1)}
            else:
                windows_out[w] = {'success_rate': None, 'delta': None}
        suites_out[suite]['per_layer'][layer_name] = {'baseline': round(baseline_overall, 4), 'windows': windows_out}

    if not suites_out:
        return jsonify({'status': 404, 'error': {'code': 'NO_DATA', 'message': 'No temporal data found in SmolVLA concept ablation results.'}}), 404

    return jsonify({'status': 200, 'data': {'model': 'smolvla', 'model_name': 'SmolVLA', 'available': True, 'windows': windows, 'suites': suites_out}})
# Experiment Summary
