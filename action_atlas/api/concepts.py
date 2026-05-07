# Action Atlas API: concept routes
import json
from pathlib import Path
from typing import Dict, Optional
from flask import Blueprint, request, jsonify
from .helpers import (
    OFT_ABLATION_DIR, OFT_ABLATION_VIDEO_DIR, OFT_CONCEPT_ID_DIR,
    ABLATION_INDEX_FILES, MODEL_SCENE_STATE_DIRS,
    get_vla_config, load_concept_features, load_json_cached,
    normalize_suite, suite_short, parse_concept_name, parse_ablation_filename,
)
from .success_tracking import _build_oft_ablation_success_map, _build_pi05_ablation_success_map

concepts_bp = Blueprint("concepts", __name__)

# Caches local to this module
_ablation_summary_cache: Dict[str, dict] = {}
_oft_ablation_cache: Dict[str, dict] = {}
_concept_ablation_scene_cache: Dict[str, dict] = {}



from .concept_helpers import *

@concepts_bp.route('/api/concepts/list', methods=['GET'])
def get_concept_list():
    # Get available concepts organized by type (reads from actual data)
    model = request.args.get('model', 'pi05')

    if model in ('openvla', 'openvla_oft'):
        return _get_openvla_concept_list()

    if model in ('xvla', 'smolvla', 'groot'):
        config = get_vla_config(model)
        concept_id_dir = config.get('concept_id_dir')
        all_concepts = {'motion': set(), 'object': set(), 'spatial': set()}

        if concept_id_dir and Path(concept_id_dir).exists():
            for json_file in Path(concept_id_dir).glob('*.json'):
                cdata = load_json_cached(json_file)
                if cdata is None:
                    continue
                concept_keys = list(cdata.get('concepts', {}).keys())
                if not concept_keys:
                    concept_keys = [k for k in cdata.keys() if '/' in k or '_' in k]
                _collect_concepts_from_keys(concept_keys, all_concepts)

        # Fallback: extract from experiment_results
        if not any(all_concepts.values()):
            exp_data = _load_experiment_results(model)
            if exp_data:
                for section_key in ('concept_ablation', 'concept_steering', 'steering'):
                    section = exp_data.get(section_key, {})
                    if not isinstance(section, dict):
                        continue
                    for entry_val in section.values():
                        if not isinstance(entry_val, dict):
                            continue
                        concepts = entry_val.get('concepts', entry_val.get('tasks', {}))
                        if isinstance(concepts, dict):
                            _collect_concepts_from_keys(list(concepts.keys()), all_concepts)

        # Fallback: load from baked concept list files
        if not any(all_concepts.values()):
            baked_list_path = Path(__file__).parent / 'data' / f'{model}_concept_list.json'
            baked = load_json_cached(baked_list_path)
            if baked:
                for ct in ('motion', 'object', 'spatial'):
                    if ct in baked and isinstance(baked[ct], list):
                        all_concepts[ct].update(baked[ct])

        # Fallback: extract from concept_features.json
        if not any(all_concepts.values()):
            cf_data = load_concept_features()
            if cf_data:
                model_prefixes = {
                    'smolvla': ('vlm_layer_', 'expert_layer_'),
                    'groot': ('dit_layer_', 'eagle_layer_', 'vlsa_layer_'),
                    'xvla': ('layer_',),
                }
                prefixes = model_prefixes.get(model, ())
                for layer_name, layer_data in cf_data.items():
                    if layer_name.startswith('_'):
                        continue
                    if prefixes and not any(layer_name.startswith(p) for p in prefixes):
                        continue
                    for concept_type in ('motion', 'object', 'spatial'):
                        if concept_type in layer_data and isinstance(layer_data[concept_type], dict):
                            for concept in layer_data[concept_type].keys():
                                all_concepts[concept_type].add(concept)

        return jsonify({
            'status': 200,
            'data': _build_response_data(all_concepts),
            'metadata': {'concept_method': 'contrastive', 'model': model},
        })

    # Pi0.5: read from concept_features.json
    data = load_concept_features()
    if data is None:
        return jsonify({
            'status': 200,
            'data': {
                'motion': ['put', 'open', 'push', 'interact'],
                'object': ['bowl', 'plate', 'stove', 'cabinet', 'wine_bottle', 'rack', 'cream_cheese', 'drawer'],
                'spatial': ['on', 'in', 'top', 'front', 'middle'],
            }
        })

    pi05_prefixes = ('action_expert_layer_', 'paligemma_layer_', 'action_in_proj', 'action_out_proj')
    concept_types = ('motion', 'object', 'spatial')
    all_concepts = {ct: set() for ct in concept_types}
    for layer_name, layer_data in data.items():
        if layer_name.startswith('_'):
            continue
        if not any(layer_name.startswith(p) for p in pi05_prefixes):
            continue
        for concept_type in concept_types:
            if concept_type in layer_data:
                for concept in layer_data[concept_type].keys():
                    all_concepts[concept_type].add(concept)

    metadata = data.get('_metadata', {})
    return jsonify({
        'status': 200,
        'data': _build_response_data(all_concepts),
        'metadata': {
            'concept_method': metadata.get('concept_method', 'unknown'),
            'scoring_formula': metadata.get('scoring_formula', ''),
        }
    })


@concepts_bp.route('/api/concepts/summary', methods=['GET'])
def get_concepts_summary():
    # Get summary of all concepts across all layers
    data = load_concept_features()
    if data is None:
        return jsonify({
            'status': 404,
            'error': {'code': 'DATA_NOT_FOUND', 'message': 'Concept features data not found'}
        }), 404

    summary = {'motion': {}, 'object': {}, 'spatial': {}, 'action_phase': {}}

    for layer_name, layer_data in data.items():
        if layer_name.startswith('_'):
            continue
        for concept_type in ('motion', 'object', 'spatial', 'action_phase'):
            if concept_type not in layer_data:
                continue
            for concept, concept_data in layer_data[concept_type].items():
                if concept not in summary[concept_type]:
                    summary[concept_type][concept] = {
                        'total_features': 0,
                        'best_layer': None,
                        'best_layer_count': 0,
                        'max_cohens_d': 0.0,
                        'best_layer_cohens_d': None,
                    }

                count = concept_data.get('concept_features', 0)
                summary[concept_type][concept]['total_features'] += count

                if count > summary[concept_type][concept]['best_layer_count']:
                    summary[concept_type][concept]['best_layer'] = layer_name
                    summary[concept_type][concept]['best_layer_count'] = count

                max_d = concept_data.get('max_cohens_d', 0.0)
                if max_d > summary[concept_type][concept]['max_cohens_d']:
                    summary[concept_type][concept]['max_cohens_d'] = round(max_d, 4)
                    summary[concept_type][concept]['best_layer_cohens_d'] = layer_name

    return jsonify({'status': 200, 'data': summary})


@concepts_bp.route('/api/ablation/summary', methods=['GET'])
def ablation_summary_alias():
    """
    Alias for /api/vla/ablation/summary to support frontend calls.

    Query params:
        model: pi05 or openvla (default: pi05)
    """
    model = request.args.get('model', 'pi05')

    if model in ('openvla', 'openvla_oft'):
        summary = []
        oft_summary = {
            'total_task_concept_pairs': 1810,
            'zero_effect_pct': 91.6,
            'significant_effect_pct': 8.4,
            'significant_threshold_pp': 10,
            'ablation_jsons': 18,
            'ablation_videos': 11892,
            'sae_validation': '119/120 (99.2%)',
            'interpretation': 'Width (4096-dim) implies resilience: concepts are distributed redundantly across the hidden space. Contrasts with Pi0.5 (1024-dim) where ablation is severe.',
        }

        if OFT_ABLATION_VIDEO_DIR.parent.exists():
            for aj in sorted(OFT_ABLATION_VIDEO_DIR.parent.glob("ablation_*.json")):
                data = load_json_cached(aj)
                if data is None:
                    continue
                summary.append({
                    'filename': aj.name,
                    'layer': data.get('layer'),
                    'suite': data.get('suite'),
                    'n_concepts': len(data.get('results', {})),
                    'baseline_success': data.get('baseline_success_rate'),
                })

        concept_summary = []
        config = get_vla_config('openvla')
        oft_cid_dir = config.get('concept_id_dir')
        if oft_cid_dir and Path(oft_cid_dir).exists():
            all_concepts_set: Dict[str, set] = {}
            for json_file in Path(oft_cid_dir).glob('oft_concept_id_layer*_*.json'):
                cid_data = load_json_cached(json_file)
                if cid_data is None:
                    continue
                for cname in cid_data.get('concepts', {}).keys():
                    all_concepts_set.setdefault(cname, set()).add(json_file.stem)
            for cname in sorted(all_concepts_set.keys()):
                concept_summary.append({
                    'concept': cname,
                    'n_files': len(all_concepts_set[cname]),
                })
            oft_summary['n_concepts'] = len(all_concepts_set)

        return jsonify({
            'status': 200,
            'data': {
                'vla_model': 'openvla',
                'overview': oft_summary,
                'ablation_files': summary,
                'summary': concept_summary,
            }
        })

    # X-VLA, SmolVLA, GR00T: load from baked ablation index
    if model in ('xvla', 'smolvla', 'groot'):
        fname = ABLATION_INDEX_FILES.get(model)
        if not fname:
            return jsonify({
                'status': 200,
                'data': {'vla_model': model, 'overview': {}, 'note': f'No ablation index found for {model}'}
            })

        baked_index = Path(__file__).parent / "data" / fname

        if baked_index.exists():
            cache_key = f'ablation_summary_{model}'
            if cache_key in _ablation_summary_cache:
                return jsonify(_ablation_summary_cache[cache_key])

            idx_data = load_json_cached(baked_index, f'ablation_index_raw_{model}')
            if idx_data is None:
                idx_data = {}

            if isinstance(idx_data, list):
                all_items = idx_data
                overview = {}
            elif isinstance(idx_data, dict):
                all_items = idx_data.get('videos', idx_data.get('entries', []))
                overview = {k: v for k, v in idx_data.items() if k not in ('videos', 'entries')}
            else:
                all_items = []
                overview = {}

            available_suites = sorted(set(
                v.get('suite', '') for v in all_items if v.get('suite')
            ))

            ablation_summary = []
            first_item = all_items[0] if all_items else {}

            if 'file' in first_item:
                # SmolVLA/X-VLA: entries are ablation result files
                for entry in all_items:
                    ablation_summary.append({
                        'layer': entry.get('layer'),
                        'suite': entry.get('suite'),
                        'component': entry.get('component', ''),
                        'mode': entry.get('mode', ''),
                        'n_concepts': entry.get('n_concepts', entry.get('n_results', 0)),
                        'baseline_overall': entry.get('baseline_overall'),
                        'batch': entry.get('batch'),
                    })
                available_layers = sorted(set(e.get('layer') for e in all_items if e.get('layer') is not None))
                overview['available_layers'] = available_layers
                overview['total_entries'] = len(all_items)

            elif 'path' in first_item:
                # GR00T: entries are video records
                overview['total_videos'] = len(all_items)
                exp_types: Dict[str, int] = {}
                layer_set: set = set()
                suite_layer_exp: Dict[str, Dict[str, set]] = {}
                for v in all_items:
                    et = v.get('experiment_type', 'unknown')
                    exp_types[et] = exp_types.get(et, 0) + 1
                    layer_val = v.get('layer', '')
                    if layer_val:
                        layer_set.add(layer_val)
                    v_suite = v.get('suite', '')
                    if v_suite and layer_val:
                        suite_layer_exp.setdefault(v_suite, {}).setdefault(layer_val, set()).add(et)
                overview['experiment_types'] = exp_types
                overview['available_layers'] = sorted(layer_set)
                for v_suite, layers_map in sorted(suite_layer_exp.items()):
                    for layer_val, etypes in sorted(layers_map.items()):
                        ablation_summary.append({
                            'layer': layer_val,
                            'suite': v_suite,
                            'component': layer_val.split('_')[0] if '_' in layer_val else '',
                            'experiment_types': sorted(etypes),
                        })

            result = {
                'status': 200,
                'data': {
                    'vla_model': model,
                    'overview': overview,
                    'total_items': len(all_items),
                    'available_suites': available_suites,
                    'ablation_summary': ablation_summary if ablation_summary else None,
                }
            }
            _ablation_summary_cache[cache_key] = result
            return jsonify(result)

        return jsonify({
            'status': 200,
            'data': {'vla_model': model, 'overview': {}, 'note': f'No ablation index found for {model}'}
        })

    # Pi0.5 ablation summary
    concepts_data = {
        'motion/put': {'concept_tasks_delta': -100, 'other_tasks_delta': -93.3, 'selectivity': -6.7, 'n_features': 30},
        'motion/open': {'concept_tasks_delta': -100, 'other_tasks_delta': -90, 'selectivity': -10, 'n_features': 30},
        'motion/push': {'concept_tasks_delta': -80, 'other_tasks_delta': -88.9, 'selectivity': 8.9, 'n_features': 30},
        'motion/interact': {'concept_tasks_delta': -100, 'other_tasks_delta': -88.9, 'selectivity': -11.1, 'n_features': 30},
        'motion/pick': {'concept_tasks_delta': -100, 'other_tasks_delta': -47, 'selectivity': -53, 'n_features': 30},
        'object/bowl': {'concept_tasks_delta': -67, 'other_tasks_delta': -40, 'selectivity': -27, 'n_features': 30},
    }

    summary = [
        {
            'concept': concept,
            'n_features': data['n_features'],
            'concept_tasks_delta': data['concept_tasks_delta'],
            'other_tasks_delta': data['other_tasks_delta'],
            'selectivity': data['selectivity'],
        }
        for concept, data in concepts_data.items()
    ]

    return jsonify({
        'status': 200,
        'data': {
            'vla_model': 'pi05',
            'note': 'Pi0.5 ablation is severe (-60 to -100pp) due to 1024-dim hidden space concentrating critical information.',
            'summary': summary
        }
    })


@concepts_bp.route('/api/ablation/videos', methods=['GET'])
def ablation_videos_alias():
    """
    Alias for /api/vla/ablation/videos to support frontend calls.

    Query params:
        model: pi05 or openvla (default: pi05)
        suite: LIBERO suite (for openvla, default: libero_goal)
        limit: max videos (default: 100)
    """
    model = request.args.get('model', 'pi05')
    limit = int(request.args.get('limit', '100'))

    if model in ('openvla_DISABLED',):  # Disabled: OFT now uses baked index below
        suite = request.args.get('suite', 'libero_goal')
        suite_dir = OFT_ABLATION_VIDEO_DIR / suite
        videos = []
        oft_success_map = _build_oft_ablation_success_map()

        if suite_dir.exists():
            for video_path in sorted(suite_dir.glob("*.mp4"))[:limit]:
                parsed = parse_ablation_filename(video_path.name)
                concept_type = parsed['concept_type']
                concept_name = parsed['concept']
                full_concept = f"{concept_type}/{concept_name}" if concept_type and concept_name else video_path.stem

                success_key = (suite, parsed['layer'], concept_type, concept_name, parsed['task'], parsed['episode'])
                videos.append({
                    "path": f"oft_ablation/{suite}/{video_path.name}",
                    "filename": video_path.name,
                    "concept": full_concept,
                    "concept_type": concept_type,
                    "layer": parsed['layer'],
                    "task": parsed['task'],
                    "episode": parsed['episode'],
                    "suite": suite,
                    "is_ablated": True,
                    "success": oft_success_map.get(success_key),
                })
        else:
            baked_index = Path(__file__).parent / "data" / "oft_ablation_index.json"
            idx_data = load_json_cached(baked_index)
            if idx_data:
                all_vids = idx_data.get('videos', [])
                videos = [v for v in all_vids if v.get('suite') == suite][:limit]

        if OFT_ABLATION_VIDEO_DIR.exists():
            available_suites = [d.name for d in OFT_ABLATION_VIDEO_DIR.iterdir() if d.is_dir()]
        else:
            available_suites = ['libero_goal', 'libero_10', 'libero_object', 'libero_spatial']

        return jsonify({
            'status': 200,
            'data': {
                'vla_model': 'openvla',
                'videos': videos,
                'total_available': 11892,
                'available_suites': available_suites,
            }
        })

    # For OFT, X-VLA, SmolVLA, GR00T: load from baked ablation index
    suite = request.args.get('suite', '')
    if model in ('openvla', 'openvla_oft', 'xvla', 'smolvla', 'groot'):
        fname = ABLATION_INDEX_FILES.get(model, ABLATION_INDEX_FILES.get('openvla', ''))
        vid_cache_key = f'ablation_videos_index_{model}'

        if vid_cache_key in _ablation_summary_cache:
            all_vids = _ablation_summary_cache[vid_cache_key]
        else:
            baked_index = Path(__file__).parent / "data" / fname
            all_vids = []
            idx_data = load_json_cached(baked_index, f'ablation_vid_index_{model}')
            if idx_data is not None:
                if isinstance(idx_data, dict):
                    all_vids = idx_data.get('videos', idx_data)
                else:
                    all_vids = idx_data
            _ablation_summary_cache[vid_cache_key] = all_vids

        videos = all_vids
        if suite:
            videos = [v for v in videos if v.get('suite') == suite or v.get('suite', '').startswith(suite)]
        if limit:
            videos = videos[:limit]

        concepts = sorted(set(v.get('concept', '') for v in all_vids if v.get('concept')))
        exp_types = sorted(set(v.get('experiment_type', '') for v in all_vids if v.get('experiment_type')))
        suites_avail = sorted(set(v.get('suite', '') for v in all_vids if v.get('suite')))

        return jsonify({
            'status': 200,
            'data': {
                'vla_model': model,
                'videos': videos,
                'total': len(all_vids),
                'concepts': concepts,
                'experiment_types': exp_types,
                'available_suites': suites_avail,
            }
        })

    # Pi0.5 concept ablation videos
    from .features import PI05_ABLATION_VIDEO_DIR
    videos = []
    pi05_success_map = _build_pi05_ablation_success_map()

    if PI05_ABLATION_VIDEO_DIR.exists():
        suite_dirs = [PI05_ABLATION_VIDEO_DIR / suite] if suite else sorted(PI05_ABLATION_VIDEO_DIR.iterdir())
        for suite_path in suite_dirs:
            if not suite_path.is_dir():
                continue
            suite_name = suite_path.name
            for video_path in sorted(suite_path.glob("*.mp4")):
                parsed = parse_ablation_filename(video_path.name)
                concept_type = parsed['concept_type']
                concept_name = parsed['concept']
                full_concept = f"{concept_type}/{concept_name}" if concept_type and concept_name else video_path.stem

                success_key = (suite_name, parsed['layer'], concept_type, concept_name, parsed['task'])
                videos.append({
                    "path": f"pi05_ablation/{suite_name}/{video_path.name}",
                    "filename": video_path.name,
                    "concept": full_concept,
                    "concept_type": concept_type,
                    "layer": parsed['layer'],
                    "task": parsed['task'],
                    "episode": parsed['episode'],
                    "suite": suite_name,
                    "is_ablated": True,
                    "success": pi05_success_map.get(success_key),
                })
    else:
        baked_index = Path(__file__).parent / "data" / "pi05_ablation_index.json"
        idx_data = load_json_cached(baked_index)
        if idx_data:
            all_vids = idx_data.get('videos', [])
            videos = [v for v in all_vids if v.get('suite') == suite] if suite else all_vids

    if PI05_ABLATION_VIDEO_DIR.exists():
        available_suites = [d.name for d in PI05_ABLATION_VIDEO_DIR.iterdir() if d.is_dir()]
    else:
        available_suites = ['libero_goal', 'libero_object', 'libero_spatial']

    return jsonify({
        'status': 200,
        'data': {
            'vla_model': model if model in ('xvla', 'smolvla', 'groot') else 'pi05',
            'videos': videos[:limit],
            'total_available': len(videos),
            'available_suites': available_suites,
        }
    })


# Concept Counts and Layer Analysis


@concepts_bp.route('/api/vla/concept_id', methods=['GET'])
def get_concept_id():
    """
    Get concept identification results for all VLA models.

    Query params:
        model: openvla, xvla, smolvla, groot (default: openvla)
        suite: libero_goal, libero_object, libero_spatial, libero_10, libero_long
        layer: layer number (0-31 for OFT, 0-17 for Pi0.5) or 'all'
    """
    model = request.args.get('model', 'openvla')
    suite = request.args.get('suite', 'libero_goal')
    layer = request.args.get('layer', 'all')

    if model in ('openvla', 'openvla_oft'):
        concept_id_dir = OFT_CONCEPT_ID_DIR
        layer_format = 'oft_concept_id_layer{:02d}_{}.json'
        n_layers = 32
    elif model in ('xvla', 'smolvla', 'groot', 'act'):
        config = get_vla_config(model)
        if model == 'smolvla' and suite == 'metaworld':
            concept_id_dir = config.get('metaworld_concept_id_dir')
            if not concept_id_dir:
                concept_id_dir = config.get('concept_id_dir')
        else:
            concept_id_dir = config.get('concept_id_dir')
        if not concept_id_dir:
            return jsonify({
                'status': 404,
                'error': {'code': 'DATA_NOT_FOUND', 'message': f'No concept_id_dir configured for {model}'}
            }), 404
        concept_id_dir = Path(concept_id_dir)
        layer_format = None
        n_layers = None
    else:
        return jsonify({
            'status': 404,
            'error': {'code': 'NOT_IMPLEMENTED', 'message': f'Concept ID not available for {model}'}
        }), 404

    if not concept_id_dir.exists():
        config = get_vla_config(model)
        return jsonify({
            'status': 200,
            'data': {
                'model': model,
                'suite': suite,
                'n_layers': 0,
                'layers': {},
                'concept_summary': {},
                'available_suites': config.get('suites', []),
                'empty': True,
                'message': f'No concept ID data directory found for {model}',
            }
        })

    def _extract_concepts(data):
        # Extract concept dict from either nested or flat format
        concepts = data.get('concepts', {})
        if concepts:
            return concepts
        return {k: v for k, v in data.items() if '/' in k and isinstance(v, dict)}

    if layer == 'all':
        results = {}

        if layer_format is not None:
            for layer_idx in range(n_layers):
                filename = layer_format.format(layer_idx, suite)
                filepath = concept_id_dir / filename
                data = load_json_cached(filepath)
                if data is not None:
                    results[f'layer_{layer_idx}'] = data
        else:
            for json_file in sorted(concept_id_dir.glob(f'*_{suite}*.json')):
                data = load_json_cached(json_file)
                if data is not None:
                    results[json_file.stem] = data

        # Compute cross-layer summary
        concept_summary = {}
        for layer_key, layer_data in results.items():
            concepts = _extract_concepts(layer_data)
            for concept_key, concept_info in concepts.items():
                if not isinstance(concept_info, dict):
                    continue
                if concept_key not in concept_summary:
                    concept_summary[concept_key] = {
                        'max_cohens_d': 0,
                        'best_layer': None,
                        'n_features_d_gt_1': 0,
                        'layers_with_signal': 0,
                    }
                max_d = concept_info.get('max_cohens_d', 0)
                if max_d > concept_summary[concept_key]['max_cohens_d']:
                    concept_summary[concept_key]['max_cohens_d'] = max_d
                    concept_summary[concept_key]['best_layer'] = layer_key
                n_gt1 = concept_info.get('n_features_d_gt_1', 0)
                concept_summary[concept_key]['n_features_d_gt_1'] += n_gt1
                if n_gt1 > 0:
                    concept_summary[concept_key]['layers_with_signal'] += 1

        config = get_vla_config(model)
        return jsonify({
            'status': 200,
            'data': {
                'model': model,
                'suite': suite,
                'n_layers': len(results),
                'layers': results,
                'concept_summary': concept_summary,
                'available_suites': config.get('suites', ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10']),
            }
        })

    # Single layer
    try:
        layer_idx = int(layer)
    except ValueError:
        return jsonify({
            'status': 400,
            'error': {'code': 'INVALID_LAYER', 'message': 'layer must be an integer or "all"'}
        }), 400

    if layer_format is not None:
        filename = layer_format.format(layer_idx, suite)
        filepath = concept_id_dir / filename
        layer_data = load_json_cached(filepath)
        if layer_data is None:
            return jsonify({
                'status': 404,
                'error': {'code': 'DATA_NOT_FOUND', 'message': f'No data for layer {layer_idx} suite {suite}'}
            }), 404
    else:
        pattern = f'*L{layer_idx:02d}*{suite}*.json'
        matches = list(concept_id_dir.glob(pattern))
        if not matches:
            return jsonify({
                'status': 200,
                'data': {
                    'model': model,
                    'suite': suite,
                    'layer': layer_idx,
                    'concepts': {},
                    'empty': True,
                    'message': f'No concept ID data yet for layer {layer_idx} suite {suite}',
                }
            })
        layer_data = load_json_cached(matches[0])
        if layer_data is None:
            layer_data = {}

    return jsonify({
        'status': 200,
        'data': {
            'model': model,
            'suite': suite,
            'layer': layer_idx,
            **layer_data
        }
    })


@concepts_bp.route('/api/vla/scene_state/concept_ablation', methods=['GET'])
def get_concept_ablation_scene_state():
    # Get concept ablation scene state data (tasks, conditions, trials with EEF trajectories)
    model = request.args.get('model', 'smolvla')
    suite = request.args.get('suite', 'libero_spatial')
    component = request.args.get('component', 'expert')
    task_id = request.args.get('task_id', type=int)
    condition = request.args.get('condition')

    data = _load_concept_ablation_scene(model, suite, component)
    if data is None:
        return jsonify({
            'model': model,
            'suite': suite,
            'component': component,
            'type': 'concept_ablation',
            'n_tasks': 0,
            'total_trials': 0,
            'tasks': [],
            'empty': True,
            'message': f'No concept ablation scene data yet for {model}/{suite}/{component}',
        })

    tasks = data.get('tasks', [])

    if task_id is not None:
        tasks = [t for t in tasks if t.get('task_id') == task_id]
        if not tasks:
            return jsonify({'error': f'Task {task_id} not found'}), 404

    if condition:
        for task in tasks:
            task['conditions'] = [c for c in task.get('conditions', [])
                                  if c.get('condition') == condition or c.get('concept') == condition]

    total_trials = sum(t.get('total_trials', 0) for t in tasks)
    return jsonify({
        'model': model,
        'suite': suite,
        'component': component,
        'type': 'concept_ablation',
        'n_tasks': len(tasks),
        'total_trials': total_trials,
        'tasks': tasks,
    })


@concepts_bp.route('/api/vla/scene_state/concept_ablation/summary', methods=['GET'])
def get_concept_ablation_scene_summary():
    # List available concept ablation scene state files
    model = request.args.get('model', 'smolvla')
    dir_name = MODEL_SCENE_STATE_DIRS.get(model, f'{model}_scene_state')
    data_dir = Path(__file__).parent / 'data' / dir_name

    if not data_dir.exists():
        return jsonify({'files': [], 'model': model}), 404

    files = []
    for f in sorted(data_dir.glob('*_concept_ablation.json')):
        parts = f.stem.replace('_concept_ablation', '').rsplit('_', 1)
        if len(parts) == 2:
            file_suite, component = parts
        else:
            file_suite, component = parts[0], 'unknown'
        size_mb = f.stat().st_size / (1024 * 1024)
        files.append({
            'filename': f.name,
            'suite': file_suite,
            'component': component,
            'size_mb': round(size_mb, 1),
        })

    return jsonify({'model': model, 'files': files, 'total': len(files)})


@concepts_bp.route('/api/vla/concept_ablation_results', methods=['GET'])
def get_concept_ablation_results():
    """
    Get per-concept ablation results (success rates, deltas) for pen testing display.

    Query params:
        model: Model name
        layer: Layer name (e.g., L12, expert_L00)
        suite: Suite name (e.g., libero_goal, metaworld)
    """
    model = request.args.get('model', 'pi05')
    layer = request.args.get('layer', '')
    suite_param = request.args.get('suite', 'libero_goal')

    # Try baked concept ablation results first
    baked_path = Path(__file__).parent / 'data' / f'{model}_concept_ablation_baked.json'
    baked = load_json_cached(baked_path)
    if baked:
        s_short = suite_short(suite_param)
        best_key = None
        for key in baked:
            if layer and s_short:
                if layer in key and s_short in key:
                    best_key = key
                    break
            elif s_short and s_short in key:
                if best_key is None:
                    best_key = key
        if not best_key and baked:
            best_key = next(iter(baked))
        if best_key:
            entry = baked[best_key]
            return jsonify({
                'concepts': entry.get('concepts', []),
                'baseline': {'success_rate': entry.get('baseline_sr', 1.0)},
                'baseline_sr': entry.get('baseline_sr', 1.0),
                'layer': entry.get('layer', layer),
                'suite': entry.get('suite', suite_param),
                'n_episodes': entry.get('n_episodes', 0),
                'model': model,
                'entry_key': best_key,
                'available_entries': list(baked.keys()),
            })

    data = _load_experiment_results(model if model != 'openvla' else 'oft')
    if not data:
        return jsonify({'concepts': [], 'baseline': {}, 'layer': layer, 'suite': suite_param})

    ca = data.get('concept_ablation', {})
    if not ca:
        return jsonify({'concepts': [], 'baseline': {}, 'layer': layer, 'suite': suite_param})

    s_short = suite_short(suite_param)

    best_key = None
    for key in ca:
        if not isinstance(ca[key], dict):
            continue
        if layer and s_short:
            if layer in key and s_short in key:
                best_key = key
                break
        elif s_short and s_short in key:
            if best_key is None:
                best_key = key

    if not best_key:
        available = [k for k in ca if isinstance(ca[k], dict) and ('concepts' in ca[k] or 'tasks' in ca[k])]
        return jsonify({
            'concepts': [],
            'baseline': {},
            'available_entries': available[:20],
            'layer': layer,
            'suite': suite_param,
        })

    entry = ca[best_key]
    baseline = entry.get('baseline', {})

    concepts = []
    tasks_or_concepts = entry.get('tasks', entry.get('concepts', {}))
    if isinstance(tasks_or_concepts, dict):
        for concept_name, cdata in tasks_or_concepts.items():
            if not isinstance(cdata, dict) or 'tasks' not in cdata:
                continue
            task_results = cdata.get('tasks', {})
            total_success = 0
            total_episodes = 0
            per_task = []
            for task_id, tdata in sorted(task_results.items()):
                if isinstance(tdata, dict):
                    sr = tdata.get('success_rate', 0)
                    delta = tdata.get('delta', 0)
                    n = len(tdata.get('successes', []))
                    total_success += sr * n
                    total_episodes += n
                    per_task.append({
                        'task': int(task_id) if task_id.isdigit() else task_id,
                        'success_rate': sr,
                        'delta': delta,
                    })

            overall_sr = total_success / total_episodes if total_episodes > 0 else 0
            bl_sr = None
            if isinstance(baseline, dict):
                bl_srs = [v.get('success_rate', 0) for v in baseline.values() if isinstance(v, dict)]
                bl_sr = sum(bl_srs) / len(bl_srs) if bl_srs else None

            # Format concept name using parse_concept_name
            category, subconcept = parse_concept_name(concept_name)
            if category != 'unknown':
                concept_formatted = f'{category}/{subconcept}'
            else:
                concept_formatted = concept_name

            concepts.append({
                'concept': concept_formatted,
                'success_rate': round(overall_sr, 4),
                'baseline_rate': round(bl_sr, 4) if bl_sr is not None else None,
                'delta': round(overall_sr - (bl_sr or 0), 4) if bl_sr is not None else None,
                'n_episodes': total_episodes,
                'n_features': len(cdata.get('features', [])),
                'per_task': per_task,
            })

    concepts.sort(key=lambda c: c.get('delta', 0))

    return jsonify({
        'concepts': concepts,
        'baseline': baseline,
        'layer': entry.get('layer', layer),
        'suite': entry.get('suite', suite_param),
        'n_episodes': entry.get('n_episodes', 0),
        'model': model,
        'entry_key': best_key,
        'available_entries': [k for k in ca if isinstance(ca[k], dict)],
    })
