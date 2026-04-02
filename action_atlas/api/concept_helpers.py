"""Concept data loading helpers."""
from .helpers import *
from .data_loaders import *

def _load_experiment_results(model: str) -> Optional[Dict]:
    """Load pre-aggregated experiment results for a model (late import to avoid circular)."""
    from .experiments import _load_experiment_results as _load
    return _load(model)


def _collect_concepts_from_keys(concept_keys: list, all_concepts: dict) -> None:
    """Classify concept keys into category sets using parse_concept_name."""
    for concept_name in concept_keys:
        category, subconcept = parse_concept_name(concept_name)
        if category in all_concepts:
            all_concepts[category].add(subconcept)


def _build_response_data(all_concepts: dict) -> dict:
    """Build sorted response data from concept sets, omitting empty categories."""
    return {ct: sorted(concepts) for ct, concepts in all_concepts.items() if concepts}


def _get_openvla_concept_list():
    """Load concept list from OFT concept ID JSON files."""
    config = get_vla_config('openvla')
    concept_id_dir = config.get('concept_id_dir')
    all_concepts = {'motion': set(), 'object': set(), 'spatial': set()}

    if concept_id_dir and concept_id_dir.exists():
        for json_file in concept_id_dir.glob('oft_concept_id_layer*_*.json'):
            data = load_json_cached(json_file)
            if data is None:
                continue
            for concept_name in data.get('concepts', {}).keys():
                category, subconcept = parse_concept_name(concept_name)
                if category in all_concepts:
                    all_concepts[category].add(subconcept)

    return jsonify({
        'status': 200,
        'data': _build_response_data(all_concepts),
        'metadata': {
            'concept_method': 'contrastive',
            'scoring_formula': 'score_f = cohens_d_f * freq_f',
            'source': 'oft_concept_id',
            'n_layers': 32,
        }
    })


def _extract_concept_counts_from_layer(layer_data: dict) -> dict:
    """Extract concept feature counts from a single layer entry in concept_features.json."""
    motion_count = sum(
        v.get('concept_features', 0)
        for v in layer_data.get('motion', {}).values()
    )
    object_count = sum(
        v.get('concept_features', 0)
        for v in layer_data.get('object', {}).values()
    )
    spatial_count = sum(
        v.get('concept_features', 0)
        for v in layer_data.get('spatial', {}).values()
    )
    total = motion_count + object_count + spatial_count

    counts_map = {'motion': motion_count, 'object': object_count, 'spatial': spatial_count}
    dominant = max(counts_map, key=counts_map.get) if total > 0 else 'none'

    all_concepts = []
    for cat_name, cat_data in layer_data.items():
        if cat_name.startswith('_') or not isinstance(cat_data, dict):
            continue
        for sub_name, sub_data in cat_data.items():
            if not isinstance(sub_data, dict):
                continue
            n = sub_data.get('concept_features', 0)
            if n > 0:
                all_concepts.append({'name': f'{cat_name}/{sub_name}', 'type': cat_name, 'count': n})
    all_concepts.sort(key=lambda x: x['count'], reverse=True)

    return {
        'feature_count': total,
        'motion_features': motion_count,
        'object_features': object_count,
        'spatial_features': spatial_count,
        'dominant_type': dominant,
        'top_concepts': all_concepts[:10],
    }


def _load_concept_counts_from_descriptions(model: str, suite: str, config: dict) -> dict:
    """Fallback: extract concept counts from bundled description files."""
    import re as _re
    result = {}

    desc_dir_map = {
        'smolvla': 'smolvla',
        'xvla': 'xvla',
        'groot': 'groot',
        'openvla': 'oft_single',
    }
    desc_dir_name = desc_dir_map.get(model)
    if not desc_dir_name:
        return result

    base_dir = Path(__file__).parent / 'data' / 'descriptions' / 'contrastive' / desc_dir_name
    if not base_dir.exists():
        return result

    bundled_concept_dirs = []
    if model == 'smolvla':
        mw_dir = config.get('metaworld_concept_id_dir')
        if mw_dir and Path(mw_dir).exists():
            bundled_concept_dirs.append(('metaworld', Path(mw_dir)))

    s_long = normalize_suite(suite)
    s_short = suite_short(suite) if suite.startswith('libero_') else suite

    for layer_name in config.get('layers', []):
        desc_file = None
        n_features = 0

        if model == 'smolvla':
            m = _re.match(r'(expert|vlm)_layer_(\d+)', layer_name)
            if m:
                pathway_name = m.group(1)
                layer_num = m.group(2).zfill(2)
                for s in (s_long, s_short):
                    candidate = base_dir / f"descriptions_smolvla_{pathway_name}_layer{layer_num}_{s}.json"
                    if candidate.exists():
                        desc_file = candidate
                        break

        elif model == 'xvla':
            m = _re.search(r'layer_(\d+)', layer_name)
            if m:
                layer_num = m.group(1).zfill(2)
                for s in (s_long, s_short):
                    candidate = base_dir / f"descriptions_xvla_single_layer{layer_num}_{s}.json"
                    if candidate.exists():
                        desc_file = candidate
                        break

        elif model == 'groot':
            m = _re.match(r'(dit|eagle|vlsa)_layer_(\d+)', layer_name)
            if m:
                pathway_name = m.group(1)
                layer_num = m.group(2).zfill(2)
                for s in (s_long, s_short):
                    candidate = base_dir / f"descriptions_groot_{pathway_name}_{layer_num}_{s}.json"
                    if candidate.exists():
                        desc_file = candidate
                        break

        if desc_file:
            data = load_json_cached(desc_file)
            if data:
                descs = data.get('descriptions', {})
                n_features = len(descs) if isinstance(descs, dict) else 0

        if n_features > 0:
            third = n_features // 3
            remainder = n_features % 3
            result[layer_name] = {
                'motion': third + (1 if remainder > 0 else 0),
                'object': third + (1 if remainder > 1 else 0),
                'spatial': third,
                'top_concepts': [{'name': f'{n_features} features described', 'n_features': n_features}],
            }

    return result


def _load_concept_counts_for_model(model: str, suite: str, config: dict) -> dict:
    """
    Load concept counts per layer from concept_id JSON files.

    Falls back to description files when concept_id directories are unavailable.
    """
    import re as _re
    result = {}

    concept_id_dir = config.get('concept_id_dir')
    if not concept_id_dir or not Path(concept_id_dir).exists():
        return _load_concept_counts_from_descriptions(model, suite, config)

    s_long = normalize_suite(suite)
    s_short = suite_short(suite) if suite.startswith('libero_') else suite

    for layer_name in config.get('layers', []):
        concept_file = None

        if model == 'xvla':
            m = _re.search(r'layer_(\d+)', layer_name)
            if m:
                layer_num = m.group(1).zfill(2)
                for s in (s_long, s_short):
                    candidate = Path(concept_id_dir) / f"xvla_concept_id_layer{layer_num}_{s}.json"
                    if candidate.exists():
                        concept_file = candidate
                        break

        elif model == 'smolvla':
            m = _re.match(r'(expert|vlm)_layer_(\d+)', layer_name)
            if m:
                pathway_name = m.group(1)
                layer_num = m.group(2).zfill(2)
                for s in (s_long, s_short):
                    candidate = Path(concept_id_dir) / f"smolvla_{pathway_name}_concept_id_L{layer_num}_{s}_mean.json"
                    if candidate.exists():
                        concept_file = candidate
                        break
                if not concept_file and suite.startswith('metaworld'):
                    mw_dir = config.get('metaworld_concept_id_dir')
                    if mw_dir and Path(mw_dir).exists():
                        candidate = Path(mw_dir) / f"smolvla_{pathway_name}_L{layer_num}_metaworld_concept_id.json"
                        if candidate.exists():
                            concept_file = candidate

        elif model == 'groot':
            m = _re.match(r'(dit|eagle|vlsa)_layer_(\d+)', layer_name)
            if m:
                pathway_name = m.group(1)
                file_pathway = 'eagle_lm' if pathway_name == 'eagle' else pathway_name
                layer_num = m.group(2).zfill(2)
                for s in (s_long, s_short):
                    candidate = Path(concept_id_dir) / f"groot_concept_id_{file_pathway}_L{layer_num}_{s}.json"
                    if candidate.exists():
                        concept_file = candidate
                        break

        if not concept_file:
            continue

        data = load_json_cached(concept_file)
        if data is None:
            continue

        try:
            concepts = data.get('concepts', None)
            if concepts is None:
                metadata_keys = {'layer', 'suite', 'model', 'n_tasks', 'total_samples',
                                 'task_ids', 'n_features_total', 'n_alive_features', 'sae_config',
                                 'hook_point', 'pathway'}
                concepts = {k: v for k, v in data.items()
                            if isinstance(v, dict) and k not in metadata_keys}

            motion_count = 0
            object_count = 0
            spatial_count = 0
            top_concepts = []

            for concept_name, concept_data in concepts.items():
                category, _ = parse_concept_name(concept_name)
                if category == 'motion':
                    motion_count += 1
                elif category == 'object':
                    object_count += 1
                elif category == 'spatial':
                    spatial_count += 1
                else:
                    # Extended classification by prefix
                    if category in ('grasp', 'reach', 'push', 'pull', 'lift', 'place'):
                        motion_count += 1
                    elif category in ('tool', 'container'):
                        object_count += 1
                    elif category in ('position', 'location', 'direction'):
                        spatial_count += 1

                n_features = 0
                if isinstance(concept_data, dict):
                    n_features = concept_data.get('n_features', len(concept_data.get('top_features', [])))
                top_concepts.append({'name': concept_name, 'n_features': n_features})

            top_concepts.sort(key=lambda x: x['n_features'], reverse=True)

            result[layer_name] = {
                'motion': motion_count,
                'object': object_count,
                'spatial': spatial_count,
                'top_concepts': top_concepts[:5],
            }

        except KeyError:
            pass

    return result


def _load_concept_ablation_scene(model: str, suite: str, component: str) -> Optional[dict]:
    cache_key = f"{model}_{suite}_{component}"
    if cache_key in _concept_ablation_scene_cache:
        return _concept_ablation_scene_cache[cache_key]

    dir_name = MODEL_SCENE_STATE_DIRS.get(model)
    if not dir_name:
        return None

    file_suite = normalize_suite(suite)

    data_dir = Path(__file__).parent / 'data' / dir_name
    path = data_dir / f'{file_suite}_{component}_concept_ablation.json'
    if not path.exists():
        path = data_dir / f'{file_suite}_{component}_concept_ablation_compact.json'
    if not path.exists():
        path = data_dir / f'{file_suite}_concept_ablation.json'
    if not path.exists():
        return None

    data = load_json_cached(path, f'concept_ablation_scene_{cache_key}')
    if data is not None:
        _concept_ablation_scene_cache[cache_key] = data
    return data


def _load_oft_ablation(layer: str, suite: str) -> Optional[dict]:
    """Load OFT concept ablation data for a layer/suite."""
    cache_key = f"{layer}_{suite}"
    if cache_key in _oft_ablation_cache:
        return _oft_ablation_cache[cache_key]

    filepath = OFT_ABLATION_DIR / f"ablation_{layer}_{suite}.json"
    data = load_json_cached(filepath, f'oft_ablation_{cache_key}')
    if data is not None:
        _oft_ablation_cache[cache_key] = data
    return data


