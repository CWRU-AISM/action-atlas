# Action Atlas API - feature search and semantic search
from flask import Blueprint, request, jsonify
from .helpers import *
from .data_loaders import *
import numpy as np
import re
from pathlib import Path

search_bp = Blueprint("search", __name__)
PI05_CONCEPT_ABLATION_DIR = Path(__file__).parent.parent / "results" / "experiment_results" / "pi05_concept_ablation"
PI05_ABLATION_VIDEO_DIR = PI05_CONCEPT_ABLATION_DIR / "videos"
PI05_PROBES_DIR = Path(__file__).parent.parent / "results" / "valid" / "probes"

XVLA_STEERING_DIR = Path("/data/batch_1/xvla_concept_steering")
SMOLVLA_CONCEPT_ID_DIR = Path("/data/smolvla_rollouts/concept_id")
SMOLVLA_ABLATION_DIR = Path("/data/openvla_rollouts/smolvla/concept_ablation")


def _load_feature_embeddings(model_key):
    # Load pre-computed feature embeddings from NPZ file
    if model_key in _feature_embeddings_cache:
        return _feature_embeddings_cache[model_key]

    emb_path = Path(__file__).parent / 'data' / 'feature_embeddings' / f'{model_key}_embeddings.npz'
    if not emb_path.exists():
        return None

    try:
        data = np.load(emb_path, allow_pickle=True)
        result = {
            'embeddings': data['embeddings'],
            'feature_ids': data['feature_ids'],
            'descriptions': data['descriptions'],
            'layers': data['layers'],
            'suites': data['suites'],
        }
        _feature_embeddings_cache[model_key] = result
        return result
    except Exception as e:
        print(f"Error loading embeddings {emb_path}: {e}")
        return None


def _load_common_queries():
    if 'common_queries' in _feature_embeddings_cache:
        return _feature_embeddings_cache['common_queries']

    path = Path(__file__).parent / 'data' / 'feature_embeddings' / 'common_queries.npz'
    if not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=True)
        result = {
            'queries': data['queries'].tolist(),
            'embeddings': data['embeddings'],
        }
        _feature_embeddings_cache['common_queries'] = result
        return result
    except Exception:
        return None


@search_bp.route('/api/vla/semantic_search', methods=['GET'])
def semantic_search():
    """
    Semantic search over SAE feature descriptions using pre-computed embeddings.

    Query params:
        q: Search query text
        model: Model name (pi05, openvla, xvla, smolvla, groot)
        layer: Optional layer filter
        suite: Optional suite filter
        limit: Max results (default: 20)

    Returns semantically similar features ranked by cosine similarity.
    Uses pre-computed query embeddings for common terms, falls back to text matching.
    """
    query = request.args.get('q', '').strip()
    model = request.args.get('model', 'pi05')
    layer_filter = request.args.get('layer', '')
    suite_filter = request.args.get('suite', '')
    limit = int(request.args.get('limit', '20'))

    if not query:
        return jsonify({'results': [], 'query': '', 'method': 'none'})

    model_keys = {
        'pi05': ['pi05_expert', 'pi05_paligemma'],
        'openvla': ['oft_single'],
        'xvla': ['xvla'],
        'smolvla': ['smolvla'],
        'groot': ['groot'],
    }

    keys = model_keys.get(model, [])
    if not keys:
        return jsonify({'results': [], 'query': query, 'method': 'unknown_model'})

    common = _load_common_queries()
    query_embedding = None

    if common is not None:
        queries_list = common['queries']
        query_lower = query.lower()

        if query_lower in queries_list:
            idx = queries_list.index(query_lower)
            query_embedding = common['embeddings'][idx:idx+1]
        else:
            matches = [(i, q) for i, q in enumerate(queries_list) if query_lower in q or q in query_lower]
            if matches:
                idx = matches[0][0]
                query_embedding = common['embeddings'][idx:idx+1]

    results = []
    method = 'semantic' if query_embedding is not None else 'text'

    for key in keys:
        emb_data = _load_feature_embeddings(key)
        if emb_data is None:
            continue

        embeddings = emb_data['embeddings']
        descriptions = emb_data['descriptions']
        feature_ids = emb_data['feature_ids']
        layers = emb_data['layers']
        suites = emb_data['suites']

        if query_embedding is not None:
            similarities = np.dot(embeddings, query_embedding.T).flatten()

            mask = np.ones(len(similarities), dtype=bool)
            if layer_filter:
                mask &= np.array([layer_filter in str(l) for l in layers])
            if suite_filter:
                mask &= np.array([suite_filter in str(s) for s in suites])

            similarities[~mask] = -1

            top_indices = np.argsort(similarities)[::-1][:limit * 2]

            for idx in top_indices:
                sim = float(similarities[idx])
                if sim < 0.1:
                    break
                results.append({
                    'feature_id': int(feature_ids[idx]) if isinstance(feature_ids[idx], (int, np.integer)) else str(feature_ids[idx]),
                    'description': str(descriptions[idx]),
                    'layer': str(layers[idx]),
                    'suite': str(suites[idx]),
                    'similarity': round(sim, 4),
                    'pathway': key,
                })
        else:
            query_lower = query.lower()
            for i, desc in enumerate(descriptions):
                desc_str = str(desc).lower()
                if query_lower not in desc_str:
                    continue
                if layer_filter and layer_filter not in str(layers[i]):
                    continue
                if suite_filter and suite_filter not in str(suites[i]):
                    continue

                pos = desc_str.index(query_lower)
                score = 1.0 - (pos / max(len(desc_str), 1)) * 0.5

                results.append({
                    'feature_id': int(feature_ids[i]) if isinstance(feature_ids[i], (int, np.integer)) else str(feature_ids[i]),
                    'description': str(descriptions[i]),
                    'layer': str(layers[i]),
                    'suite': str(suites[i]),
                    'similarity': round(score, 4),
                    'pathway': key,
                })

    results.sort(key=lambda r: r['similarity'], reverse=True)
    results = results[:limit]

    autocomplete = []
    if common is not None:
        query_lower = query.lower()
        autocomplete = [q for q in common['queries'] if query_lower in q][:10]

    return jsonify({
        'results': results,
        'query': query,
        'method': method,
        'total': len(results),
        'autocomplete': autocomplete,
    })
