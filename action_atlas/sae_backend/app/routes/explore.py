# explore.py
import json
import os
import random
from typing import List, Tuple

import numpy as np
import requests
from flask import Blueprint, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

from ..config import (
    EXPLANATIONS_EMBEDDING_PATH, VECTOR_DB_PATH, SIMILARITIES_PATH, CLUSTERING_PATH,
    EXPLANATIONS_EMBEDDING_PATH_GPT, VECTOR_DB_PATH_GPT, SIMILARITIES_PATH_GPT, CLUSTERING_PATH_GPT
)
from ..utils.db_manager import DBManager
from ..utils.global_state import GlobalState
from ..utils.openai_service import OpenAIService
from concurrent.futures import ThreadPoolExecutor

explore_bp = Blueprint('explore', __name__)


def get_llm_config(llm_model):
    """Get the corresponding configuration according to the LLM model"""
    if llm_model == 'gpt2-small':
        return {
            'clustering_path': CLUSTERING_PATH_GPT,
            'explanations_embedding_path': EXPLANATIONS_EMBEDDING_PATH_GPT,
            'vector_db_path': VECTOR_DB_PATH_GPT,
            'similarities_path': SIMILARITIES_PATH_GPT,
            'clustering_file_prefix': 'hierarchical_clustering_',
            'clustering_file_suffix': '_gpt_colored.npz',
            'embedding_file_prefix': 'gpt2-small_',
            'embedding_file_suffix': '-embedding.npz',
            'api_base_url': 'https://www.neuronpedia.org/api/feature/gpt2-small',
            'api_url_format': '{base_url}/{layer}-res_post_32k-oai/{index}'
        }
    else:  # 默认 gemma_2_2b
        return {
            'clustering_path': CLUSTERING_PATH,
            'explanations_embedding_path': EXPLANATIONS_EMBEDDING_PATH,
            'vector_db_path': VECTOR_DB_PATH,
            'similarities_path': SIMILARITIES_PATH,
            'clustering_file_prefix': 'hierarchical_clustering_',
            'clustering_file_suffix': '_colored.npz',
            'embedding_file_prefix': 'gemma-2-2b_',
            'embedding_file_suffix': '-embedding.npz',
            'api_base_url': 'https://www.neuronpedia.org/api/feature/gemma-2-2b',
            'api_url_format': '{base_url}/{layer}-gemmascope-{type_}-16k/{index}'
        }


def get_similar_features(layer, type_, feature_id, llm_model='gemma_2_2b'):
    """获取特征的相似特征列表"""
    config = get_llm_config(llm_model)
    
    if config['similarities_path'] is None:  # The GPT2 model currently has no similarity data
        return []
    
    filename = f"layer_{layer}_{type_}_similarities.json"
    file_path = os.path.join(config['similarities_path'], filename)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        layer_key = f"layer_{layer}_{type_}"
        layer_data = data.get(layer_key, {})

        similar_features = layer_data.get(str(feature_id), [])

        similar_list = [
            {"index": int(pair[0]), "similarity": float(pair[1])}
            for pair in similar_features
        ]

        return similar_list

    except Exception as e:
        print(f"Error loading similarity data: {str(e)}")
        return []


def convert_sae_id(sae_id, llm_model='gemma_2_2b'):
    try:
        if llm_model == 'gpt2-small':
            # GPT2: "1-res_post_32k-oai" -> "0"
            parts = sae_id.split('-')
            layer = parts[0]
            return layer
        else:
            # Gemma: "1-gemmascope-att-16k" -> "1_att"
            parts = sae_id.split('-')
            layer = parts[0]
            type_ = parts[2]  # att/mlp/res
            return f"{layer}_{type_}"
    except Exception:
        raise ValueError(f"Invalid sae_id format: {sae_id}")


@explore_bp.route('/api/sae/scatter', methods=['GET'])
def get_scatter_plot():
    """Get the two-dimensional scatter plot data of SAE characteristics"""
    sae_id = request.args.get('sae_id')
    query = request.args.get('query')
    llm_model = request.args.get('llm', 'gemma_2_2b')

    if not sae_id:
        return jsonify({
            'status': 400,
            'error': {
                'code': 'MISSING_SAE_ID',
                'message': 'MISSING_SAE_ID'
            }
        }), 400

    if llm_model not in ['gemma_2_2b', 'gpt2-small']:
        return jsonify({
            'status': 400,
            'error': {
                'code': 'INVALID_LLM',
                'message': f'INVALID_LLM: {llm_model}'
            }
        }), 400

    print(f"SAE ID: {sae_id}, LLM: {llm_model}")

    try:
        config = get_llm_config(llm_model)
        
        simplified_sae_id = convert_sae_id(sae_id, llm_model)
        print(f"Simplified SAE ID: {simplified_sae_id}")
        
        clustering_filename = f"{config['clustering_file_prefix']}{simplified_sae_id}{config['clustering_file_suffix']}"
        clustering_file = os.path.join(config['clustering_path'], clustering_filename)
        
        data = np.load(clustering_file, allow_pickle=True)

        response_data = {
            'status': 200,
            'data': {
                'llm_model': llm_model, 
                'coordinates': data['coords'].tolist(),
                'indices': data['indices'].tolist(),
                'descriptions': data['descriptions'].tolist(),
                'hierarchical_clusters': {}
            }
        }

        cluster_levels = [10, 30, 90]
        for level in cluster_levels:
            level_key = f'cluster_labels_{level}'
            color_key = f'cluster_colors_{level}'
            center_key = f'cluster_centers_{level}'

            if level_key in data.files and color_key in data.files:
                labels = data[level_key].tolist()
                unique_labels = np.unique(data[level_key])
                
                cluster_color_map = {}
                for label in unique_labels:
                    indices = np.where(data[level_key] == label)[0]
                    if len(indices) > 0:
                        representative_color = data[color_key][indices[0]].tolist()
                        cluster_color_map[str(label)] = representative_color
                
                response_data['data']['hierarchical_clusters'][level] = {
                    'labels': labels,
                    'colors': data[color_key].tolist(),
                    'centers': data[center_key].tolist(),
                    'topics': data[f'topic_words_{level}'].item(),
                    'topic_scores': data[f'topic_word_scores_{level}'].item(),
                    'cluster_colors': cluster_color_map
                }

        query_vector = None
        current_query = GlobalState.get_instance().get_current_query()

        if query:
            openai_service = OpenAIService.get_instance()
            query_vector = openai_service.get_embedding(query)
        elif current_query:
            query_vector = current_query['query_vector']
            query = current_query['query_text']

        if query_vector is not None:
            print("Using Query:", query)
            
            db = DBManager.get_instance().get_vector_db(
                config['vector_db_path'], 
                config['explanations_embedding_path']
            )

            embedding_filename = f"{config['embedding_file_prefix']}{sae_id}{config['embedding_file_suffix']}"
            target_file_path = os.path.join(config['explanations_embedding_path'], embedding_filename)

            if not os.path.exists(target_file_path):
                print(f"Error: File does not exist: {target_file_path}")
                raise FileNotFoundError(f"未找到文件: {target_file_path}")

            results = db.search_within_file(
                query_vector,
                target_file_path,
                k=150
            )

            top_10_results = results[:10]
            nearest_coords = np.array([data['coords'][result['local_index']] for result in top_10_results])
            distances = np.array([result['distance'] for result in top_10_results])

            similarities = np.exp(-distances)
            epsilon = 1e-10
            similarities = similarities + epsilon
            weights = similarities / similarities.sum()
            query_coords = np.average(nearest_coords, weights=weights, axis=0)
            
            response_data['data']['query'] = {
                'text': query,
                'sae_id': sae_id,
                'coordinates': query_coords.tolist(),
                'nearest_features': [
                    {
                        'feature_id': str(result['index']),
                        'similarity': 1 - (result['distance'] / 2),
                        'description': result['description'],
                        'coordinates': data['coords'][result['local_index']].tolist()
                    } for result in results
                ]
            }

        return jsonify(response_data)

    except FileNotFoundError:
        return jsonify({
            'status': 404,
            'error': {
                'code': 'DATA_NOT_FOUND',
                'message': f'No preprocessed data found for SAE {sae_id} (LLM: {llm_model})'
            }
        }), 404
    except Exception as e:
        return jsonify({
            'status': 500,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': str(e)
            }
        }), 500


def tokens_to_sentence(tokens: List[str]) -> Tuple[str, List[str]]:
    """Concatenate tokens into a sentence and return a list of processed tokens"""
    processed_tokens = [token.replace("▁", " ") for token in tokens]
    sentence = "".join(processed_tokens).strip()
    return sentence, processed_tokens


@explore_bp.route('/api/feature/detail', methods=['GET'])
def get_feature_detail():
    """Get the details of a specific feature"""
    feature_id = request.args.get('feature_id')
    sae_id = request.args.get('sae_id')
    llm_model = request.args.get('llm', 'gemma_2_2b') 

    if not feature_id or not sae_id:
        return jsonify({
            'status': 400,
            'error': {
                'code': 'MISSING_PARAMETERS',
                'message': '缺少必要参数'
            }
        }), 400

    if llm_model not in ['gemma_2_2b', 'gpt2-small']:
        return jsonify({
            'status': 400,
            'error': {
                'code': 'INVALID_LLM',
                'message': f'不支持的LLM模型: {llm_model}'
            }
        }), 400

    try:
        config = get_llm_config(llm_model)
        
        parts = sae_id.split('-')
        layer = int(parts[0])
        index = int(feature_id)
        
        if llm_model == 'gpt2-small':
            type_ = 'res_post' 
            url = config['api_url_format'].format(
                base_url=config['api_base_url'],
                layer=layer,
                index=index
            )
        else:
            type_ = parts[2]
            url = config['api_url_format'].format(
                base_url=config['api_base_url'],
                layer=layer,
                type_=type_,
                index=index
            )

        headers = {"X-Api-Key": "sk-np-8iipt8jLRWUzHtGmDW38zQzzkyCBVNuYggnJfd70l880"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        raw_data = response.json()

        # Get the OpenAI service instance
        openai_service = OpenAIService().get_instance()

        # Filter for gpt-4o-mini explanations
        explanations = raw_data.get("explanations", [])
        gpt_4o_mini_explanations = [
            exp for exp in explanations if exp["explanationModelName"] == "gpt-4o-mini"
        ]
        explanation = random.choice(gpt_4o_mini_explanations)["description"] if gpt_4o_mini_explanations else "No gpt-4o-mini explanation found"

        processed_data = {
            "feature_info": {
                "feature_id": feature_id,
                "sae_id": sae_id,
                "layer": layer,
                "type": type_,
                "index": index,
                "llm_model": llm_model 
            },
            "activation_data": [],
            "explanation": explanation,
            "raw_stats": {
                "neg_tokens": {
                    "tokens": raw_data["neg_str"],
                    "values": raw_data["neg_values"]
                },
                "pos_tokens": {
                    "tokens": raw_data["pos_str"],
                    "values": raw_data["pos_values"]
                },
                "freq_histogram": {
                    "heights": raw_data["freq_hist_data_bar_heights"],
                    "values": raw_data["freq_hist_data_bar_values"]
                },
                "logits_histogram": {
                    "heights": raw_data["logits_hist_data_bar_heights"],
                    "values": raw_data["logits_hist_data_bar_values"]
                },
                "similar_features": {
                    "feature_ids": raw_data["topkCosSimIndices"],
                    "values": raw_data["topkCosSimValues"]
                }
            }
        }

        explanation_embedding = openai_service.get_embedding(text=explanation)

        Intervals = {}
        
        def process_activation(activation):
            tokens = activation["tokens"]
            values = activation["values"]
            sentence, processed_tokens = tokens_to_sentence(tokens)
            sentence_embedding = openai_service.get_embedding(text=sentence)

            similarity = cosine_similarity(
                np.array(sentence_embedding).reshape(1, -1),
                np.array(explanation_embedding).reshape(1, -1)
            )[0][0]

            activation_data = {
                "sentence": sentence,
                "similarity": similarity,
                "max_value": activation["maxValue"],
                "max_value_token": tokens[activation["maxValueTokenIndex"]],
                "maxValueTokenIndex": activation['maxValueTokenIndex'],
                "token_value_pairs": [{"token": t, "value": v} for t, v in zip(processed_tokens, values)],
                "Interval_Bin": (activation["binMin"], activation["binMax"]),
                "Interval_contains": activation["binContains"]
            }
            
            interval_key = (activation["binMin"], activation["binMax"])
            return activation_data, interval_key, activation["binContains"]
          
        result_datas = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for activation in raw_data["activations"]:
                futures.append(executor.submit(process_activation, activation))
                
            futures_results = []
            for future in futures:
                activation_data, interval_key, bin_contains = future.result()
                futures_results.append((activation_data, interval_key, bin_contains))
                Intervals[interval_key] = bin_contains
                
            unique_results = {}
            for activation_data, interval_key, bin_contains in futures_results:
                similarity = activation_data["similarity"]
                if similarity not in unique_results:
                    unique_results[similarity] = (activation_data, interval_key, bin_contains)
            
            deduplicated_results = [item for item in unique_results.values()]
                
            similarity_sorted_results = sorted(deduplicated_results, key=lambda x: x[0]["similarity"], reverse=True)
            for similarity_rank, (activation_data, _, _) in enumerate(similarity_sorted_results, 1):
                activation_data["similarity_rank"] = similarity_rank
            
            max_value_sorted_results = sorted(deduplicated_results, key=lambda x: x[0]["max_value"], reverse=True)
            for max_value_rank, (activation_data, _, _) in enumerate(max_value_sorted_results, 1):
                activation_data["max_value_rank"] = max_value_rank
            
            result_datas = [item[0] for item in deduplicated_results]
            
            bins_count = {}
            for _, interval_key, _ in deduplicated_results:
                bin_key = str(interval_key)
                bins_count[bin_key] = bins_count.get(bin_key, 0) + 1
            
            processed_data["bins_statistics"] = {
                "bins_data": [
                    {
                        "bin_range": eval(bin_key),
                        "sample_count": count,
                        "bin_contains": Intervals[eval(bin_key)]
                    }
                    for bin_key, count in bins_count.items()
                ],
                "total_samples": len(result_datas)
            }

        processed_data["activation_data"] = result_datas

        return jsonify({
            'status': 200,
            'data': processed_data
        }), 200

    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 500,
            'error': {
                'code': 'API_REQUEST_ERROR',
                'message': f'API_REQUEST_ERROR: {str(e)}'
            }
        }), 500

    except Exception as e:
        return jsonify({
            'status': 500,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': f'PROCESSING_ERROR: {str(e)}'
            }
        }), 500