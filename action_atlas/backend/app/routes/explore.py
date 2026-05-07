# Feature exploration API routes for Action Atlas

import json
import numpy as np
from pathlib import Path
from flask import Blueprint, request, jsonify

from ..config import VLA_CONFIGS, DEFAULT_VLA_MODEL, MOTION_CONCEPTS, OBJECT_CONCEPTS, SPATIAL_CONCEPTS

explore_bp = Blueprint('explore', __name__)


def get_vla_config(vla_model: str):
    # Get configuration for a VLA model
    if vla_model not in VLA_CONFIGS:
        return None
    return VLA_CONFIGS[vla_model]


def get_description(data, feature_id):
    # Get feature description, returning a placeholder if descriptions aren't available
    if 'descriptions' in data.files:
        try:
            return str(data['descriptions'][feature_id])
        except (IndexError, KeyError):
            pass
    return f"Feature {feature_id}"


def load_viz_data(vla_model: str, layer_name: str):
    # Load preprocessed Action Atlas data for a layer
    config = get_vla_config(vla_model)
    if config is None:
        return None

    layer_num = layer_name.split("_")[-1]

    # Try different file naming conventions
    possible_paths = [
        config["viz_data"] / f"hierarchical_clustering_layer_{layer_num}_goal.npz",
        config["viz_data"] / f"hierarchical_clustering_layer_{layer_num}_concepts.npz",
        config["viz_data"] / f"hierarchical_clustering_{layer_num}_vla_colored.npz",
        config["viz_data"] / f"hierarchical_clustering_{layer_num}.npz",
    ]

    for npz_path in possible_paths:
        if npz_path.exists():
            return np.load(str(npz_path), allow_pickle=True)

    return None


def load_analysis_results(vla_model: str):
    config = get_vla_config(vla_model)
    if config is None:
        return None

    analysis_files = sorted(Path(config["analysis_dir"]).glob("all_layers_analysis_*.json"), reverse=True)
    if not analysis_files:
        return None

    with open(analysis_files[0]) as f:
        return json.load(f)


def search_features_by_concept(query: str, vla_model: str, layer_name: str, data):
    # Search for features matching a concept query
    query_lower = query.lower()

    # Check if query matches known concepts
    matching_concept = None
    concept_type = None

    for name, info in MOTION_CONCEPTS.items():
        if any(kw in query_lower for kw in info["keywords"]):
            matching_concept = name
            concept_type = "motion"
            break

    if not matching_concept:
        for name, info in OBJECT_CONCEPTS.items():
            if any(kw in query_lower for kw in info["keywords"]):
                matching_concept = name
                concept_type = "object"
                break

    if not matching_concept:
        for name, info in SPATIAL_CONCEPTS.items():
            if any(kw in query_lower for kw in info["keywords"]):
                matching_concept = name
                concept_type = "spatial"
                break

    # Load analysis results to find relevant features
    analysis = load_analysis_results(vla_model)
    if analysis is None:
        return {"text": query, "nearest_features": []}

    layer_data = analysis.get(layer_name, {})
    feature_indices = []

    if matching_concept and concept_type:
        concept_data = layer_data.get(concept_type, {}).get(matching_concept, {})
        feature_indices = concept_data.get("feature_indices", [])[:100]

    # Build response
    nearest_features = []
    for i, feat_idx in enumerate(feature_indices):
        if feat_idx < len(data['coords']):
            nearest_features.append({
                "feature_id": str(feat_idx),
                "similarity": 1.0 - (i / len(feature_indices)) if feature_indices else 0,
                "description": get_description(data, feat_idx),
                "coordinates": data['coords'][feat_idx].tolist(),
                "concept": matching_concept,
                "concept_type": concept_type,
            })

    return {
        "text": query,
        "matched_concept": matching_concept,
        "concept_type": concept_type,
        "n_features": len(feature_indices),
        "nearest_features": nearest_features
    }


@explore_bp.route('/api/vla/concepts', methods=['GET'])
def get_concepts():
    # Get all defined concepts and their statistics
    vla_model = request.args.get('model', DEFAULT_VLA_MODEL)
    layer_name = request.args.get('layer', 'action_expert_layer_10')

    analysis = load_analysis_results(vla_model)
    if analysis is None:
        return jsonify({
            "status": 404,
            "error": {"code": "ANALYSIS_NOT_FOUND", "message": "No analysis results"}
        }), 404

    layer_data = analysis.get(layer_name, {})

    concepts = {
        "motion": {},
        "object": {},
        "spatial": {},
        "action_phase": {}
    }

    for concept_type in concepts.keys():
        type_data = layer_data.get(concept_type, {})
        for concept_name, concept_data in type_data.items():
            if isinstance(concept_data, dict):
                concepts[concept_type][concept_name] = {
                    "n_features": len(concept_data.get("feature_indices", [])),
                    "strong_features": concept_data.get("strong_features", 0),
                    "very_strong_features": concept_data.get("very_strong_features", 0),
                    "score": concept_data.get("score", 0),
                    "top_features": concept_data.get("feature_indices", [])[:10],
                }

    return jsonify({
        "status": 200,
        "data": {
            "vla_model": vla_model,
            "layer": layer_name,
            "concepts": concepts
        }
    })


@explore_bp.route('/api/vla/tasks', methods=['GET'])
def get_tasks():
    # Get task definitions for a VLA model
    vla_model = request.args.get('model', DEFAULT_VLA_MODEL)
    config = get_vla_config(vla_model)

    if config is None:
        return jsonify({
            "status": 404,
            "error": {"code": "MODEL_NOT_FOUND", "message": f"Model {vla_model} not found"}
        }), 404

    return jsonify({
        "status": 200,
        "data": {
            "vla_model": vla_model,
            "tasks": config["tasks"]
        }
    })


@explore_bp.route('/api/vla/feature/<int:feature_id>', methods=['GET'])
def get_feature_detail(feature_id: int):
    # Get details for a specific feature
    vla_model = request.args.get('model', DEFAULT_VLA_MODEL)
    layer_name = request.args.get('layer', 'action_expert_layer_10')

    data = load_viz_data(vla_model, layer_name)
    if data is None:
        return jsonify({
            "status": 404,
            "error": {"code": "DATA_NOT_FOUND", "message": f"No data for {layer_name}"}
        }), 404

    if feature_id >= len(data['indices']):
        return jsonify({
            "status": 404,
            "error": {"code": "FEATURE_NOT_FOUND", "message": f"Feature {feature_id} not found"}
        }), 404

    # Get concept associations for this feature
    analysis = load_analysis_results(vla_model)
    associated_concepts = []
    if analysis:
        layer_data = analysis.get(layer_name, {})
        for concept_type in ["motion", "object", "spatial", "action_phase"]:
            for concept_name, concept_data in layer_data.get(concept_type, {}).items():
                if isinstance(concept_data, dict):
                    if feature_id in concept_data.get("feature_indices", []):
                        associated_concepts.append({
                            "type": concept_type,
                            "name": concept_name,
                            "rank": concept_data["feature_indices"].index(feature_id)
                        })

    return jsonify({
        "status": 200,
        "data": {
            "feature_id": feature_id,
            "layer": layer_name,
            "vla_model": vla_model,
            "description": get_description(data, feature_id),
            "coordinates": data['coords'][feature_id].tolist(),
            "associated_concepts": associated_concepts,
            "cluster_10": int(data['cluster_labels_10'][feature_id]) if 'cluster_labels_10' in data.files else None,
            "cluster_30": int(data['cluster_labels_30'][feature_id]) if 'cluster_labels_30' in data.files else None,
            "cluster_90": int(data['cluster_labels_90'][feature_id]) if 'cluster_labels_90' in data.files else None,
        }
    })
