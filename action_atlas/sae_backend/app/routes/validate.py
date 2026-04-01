from flask import Blueprint, request, jsonify, Flask
import requests
import numpy as np
import os
import joblib
from ..utils.openai_service import OpenAIService
from ..config import (
    EXPLANATIONS_EMBEDDING_PATH, 
    EXPLANATIONS_EMBEDDING_PATH_GPT, NEURONPEDIA_API_KEY
)
from openai import OpenAI
import time
import json
from concurrent.futures import ThreadPoolExecutor
import random


validate_bp = Blueprint('validate', __name__)
app = Flask(__name__)

def get_llm_config(llm_model):
    if llm_model == 'gpt2-small':
        return {
            'explanations_embedding_path': EXPLANATIONS_EMBEDDING_PATH_GPT,
            'embedding_file_prefix': 'gpt2-small_',
            'embedding_file_suffix': '-embedding.npz',
            'api_model_id': 'gpt2-small',
            'api_base_url': 'https://www.neuronpedia.org/api/feature/gpt2-small',
            'source_set': 'res_post_32k-oai', 
            'layer_format': '{layer}-res_post_32k-oai'
        }
    else: 
        return {
            'explanations_embedding_path': EXPLANATIONS_EMBEDDING_PATH,
            'embedding_file_prefix': 'gemma-2-2b_',
            'embedding_file_suffix': '-embedding.npz',
            'api_model_id': 'gemma-2-2b',
            'api_base_url': 'https://www.neuronpedia.org/api/feature/gemma-2-2b',
            'source_set': 'gemmascope-res-16k', 
            'layer_format': '{layer}-gemmascope-res-16k'
        }


def get_explanation_embedding(sae_id: str, feature_id: str, llm_model='gemma_2_2b') -> np.ndarray: 
    try:
        config = get_llm_config(llm_model)
        
        npz_file = f"{config['embedding_file_prefix']}{sae_id}{config['embedding_file_suffix']}"
        npz_path = os.path.join(config['explanations_embedding_path'], npz_file)
        
        data = np.load(npz_path, allow_pickle=True)
        matches = np.where(data['indices'] == feature_id)[0]
        if len(matches) == 0:
            raise ValueError(f"Feature ID  {feature_id} not found in indices")

        feature_idx = matches[0]
        return data['embeddings'][feature_idx]
        
    except Exception as e:
        raise Exception(f"Failed to get feature explanation embedding: {str(e)}")


def get_feature_info(sae_id: str, feature_id: str, llm_model='gemma_2_2b') -> tuple:
    """Get the embedding and original explanation of the feature"""
    try:
        config = get_llm_config(llm_model)
        
        npz_file = f"{config['embedding_file_prefix']}{sae_id}{config['embedding_file_suffix']}"
        npz_path = os.path.join(config['explanations_embedding_path'], npz_file)
        
        data = np.load(npz_path, allow_pickle=True)
        matches = np.where(data['indices'] == feature_id)[0]
        if len(matches) == 0:
            raise ValueError(f"Feature ID  {feature_id} not found in indices")

        feature_idx = matches[0]
        embedding = data['embeddings'][feature_idx]
        description = data['descriptions'][feature_idx]
        
        return embedding, description
        
    except Exception as e:
        raise Exception(f"Failed to get feature information: {str(e)}")


def get_token_related_features(prompt: str, token_index: int, sae_id: str, llm_model='gemma_2_2b', top_k: int = 30) -> list:
    """Get the list of features related to a specific token in the prompt"""
    try:
        config = get_llm_config(llm_model)
        
        if llm_model == 'gpt2-small':
            layer = sae_id.split('-')[0]
            selected_layers = [config['layer_format'].format(layer=layer)]
        else:
            selected_layers = [sae_id]
        
        search_data = {
            "modelId": config['api_model_id'],
            "sourceSet": config['source_set'],
            "text": prompt,
            "selectedLayers": selected_layers,
            "sortIndexes": [token_index],
            "ignoreBos": False,
            "densityThreshold": -1,
            "numResults": top_k
        }
        print(search_data)
        
        response = requests.post(
            "https://www.neuronpedia.org/api/search-all",
            headers={"Content-Type": "application/json",
                     "X-Api-Key": NEURONPEDIA_API_KEY},
            json=search_data
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed API call: {response.text}")

        data = response.json()
        
        features = []
        for result in data.get('result', []):
            feature_id = result.get('index')
            activation = result.get('values', [])[token_index] if len(result.get('values', [])) > token_index else 0
            features.append({
                'feature_id': feature_id,
                'activation': activation
            })
            
        return features
        
    except Exception as e:
        print(f"Failed to get token-related features: {str(e)}")
        return []


@validate_bp.route('/api/feature/steer', methods=['POST'])
def steer_feature():
    """Validate the impact of features on model output"""
    data = request.get_json()
    
    required_fields = ['feature_id', 'sae_id', 'prompt', 'feature_strengths']
    if not all(field in data for field in required_fields):
        return jsonify({
            'status': 400,
            'error': {
                'code': 'MISSING_PARAMETERS',
                'message': 'MISSING_PARAMETERS'
            }
        }), 400

    llm_model = data.get('llm', 'gemma_2_2b')
    
    if llm_model not in ['gemma_2_2b', 'gpt2-small']:
        return jsonify({
            'status': 400,
            'error': {
                'code': 'INVALID_LLM',
                'message': f'INVALID_LLM: {llm_model}'
            }
        }), 400

    try:
        feature_id = data['feature_id']
        sae_id = data['sae_id']
        prompt = data['prompt']
        feature_strengths = data['feature_strengths']

        config = get_llm_config(llm_model)
        
        openai_service = OpenAIService.get_instance()
        explanation_embedding = get_explanation_embedding(sae_id, feature_id, llm_model)
        
        def process_strength(strength):
            MAX_RETRIES = 3
            retry_count = 0
            base_delay = 2

            while retry_count < MAX_RETRIES:
                try:
                    print(f"Request API, strength={strength}, LLM={llm_model}, number of attempts={retry_count+1}/{MAX_RETRIES}")
                    
                    if llm_model == 'gpt2-small':
                        layer = sae_id.split('-')[0] 
                        api_layer = f"{layer}-res_post_32k-oai"
                    else:
                        api_layer = sae_id
                    
                    steer_response = requests.post(
                        "https://www.neuronpedia.org/api/steer",
                        headers={"Content-Type": "application/json"},
                        json={
                            "prompt": prompt,
                            "modelId": config['api_model_id'],
                            "features": [
                                {
                                    "modelId": config['api_model_id'],
                                    "layer": api_layer,
                                    "index": int(feature_id),
                                    "strength": float(strength)
                                }
                            ],
                            "temperature": 0.5,
                            "n_tokens": 48,
                            "freq_penalty": 2,
                            "seed": 16,
                            "strength_multiplier": 1,
                            "steer_special_tokens": True
                        },
                        timeout=60
                    )
                    
                    if steer_response.status_code != 200:
                        print(f"API returned non-200 status code: {steer_response.status_code}, response: {steer_response.text}")
                        if steer_response.status_code == 504:
                            raise requests.exceptions.RequestException(f"API timeout (504): {steer_response.text}")
                        else:
                            raise requests.exceptions.RequestException(f"API returned status code: {steer_response.status_code}")

                    steer_result = steer_response.json()
                    print(f"strength={strength}, API response successful")

                    modified_output = steer_result.get('STEERED', '')
                    default_output = steer_result.get('DEFAULT', '')
                    
                    if not modified_output:
                        raise Exception(f"Unable to get generated text from API response: {steer_result}")
                    
                    output_embedding = openai_service.get_embedding(modified_output)
                    default_embedding = openai_service.get_embedding(default_output)
                    
                    similarity = float(np.dot(output_embedding, explanation_embedding) / 
                                (np.linalg.norm(output_embedding) * np.linalg.norm(explanation_embedding)))
                    
                    diff_to_default = float(np.dot(output_embedding, default_embedding) / 
                                     (np.linalg.norm(output_embedding) * np.linalg.norm(default_embedding)))
                    
                    return {
                        'strength': strength,
                        'model_output': modified_output,
                        'default_output': default_output,
                        'similarity_to_explanation': similarity,
                        'similarity_to_default': diff_to_default,
                        'share_url': steer_result.get('shareUrl', ''),
                        'llm_model': llm_model 
                    }
                    
                except (requests.exceptions.RequestException, Exception) as e:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        print(f"The maximum number of retries has been reached, the request has been abandoned, strength={strength}, error: {e}")
                        raise Exception(f"After {MAX_RETRIES} retries, Steer API call still failed: {e}")

                    current_base_delay = base_delay * 2 if "504" in str(e) else base_delay
                    delay = current_base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
                    print(f"Request failed, strength={strength}, retry count={retry_count}/{MAX_RETRIES}, delay={delay:.2f} seconds, error: {e}")
                    time.sleep(delay)

        outputs = []
        with ThreadPoolExecutor(max_workers=min(5, len(feature_strengths))) as executor:
            futures = []
            for strength in feature_strengths:
                time.sleep(random.uniform(0.1, 0.5))
                futures.append(executor.submit(process_strength, strength))
                
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    outputs.append(result)
                    print(f"Processing completed {i+1}/{len(futures)}")
                except Exception as e:
                    print(f"Error occurred while processing strength value: {e}")
                    outputs.append({
                        'error': str(e),
                        'strength': feature_strengths[i] if i < len(feature_strengths) else 'unknown',
                        'llm_model': llm_model
                    })

        return jsonify({
            'status': 200,
            'data': {
                'llm_model': llm_model,
                'outputs': outputs
            }
        })

    except Exception as e:
        print(f"Overall handling error: {e}")
        return jsonify({
            'status': 500,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': str(e)
            }
        }), 500


@validate_bp.route('/api/feature/tokens-activation', methods=['POST'])
def get_tokens_activation():
    data = request.get_json()
    
    required_fields = ['feature_id', 'sae_id', 'prompts']
    if not all(field in data for field in required_fields):
        return jsonify({
            'status': 400,
            'error': {
                'code': 'MISSING_PARAMETERS',
                'message': 'MISSING_PARAMETERS'
            }
        }), 400

    llm_model = data.get('llm', 'gemma_2_2b')
    
    if llm_model not in ['gemma_2_2b', 'gpt2-small']:
        return jsonify({
            'status': 400,
            'error': {
                'code': 'INVALID_LLM',
                'message': f'INVALID_LLM: {llm_model}'
            }
        }), 400

    try:
        feature_id = data['feature_id']
        sae_id = data['sae_id']
        prompts = data['prompts']
        
        config = get_llm_config(llm_model)
        
        if llm_model == 'gpt2-small':
            layer = sae_id.split('-')[0]
            api_url = f"{config['api_base_url']}/{layer}-res_post_32k-oai/{feature_id}"
        else:
            api_url = f"{config['api_base_url']}/{sae_id}/{feature_id}"
        
        feature_response = requests.get(api_url)
        
        if feature_response.status_code != 200:
            raise Exception(f"Failed to obtain feature information: {feature_response.text}")
        
        feature_response_data = feature_response.json()
        max_act_approx = feature_response_data.get('maxActApprox', 1.0)
        
        prompt_results = []

        for prompt in prompts:
            if llm_model == 'gpt2-small':
                layer = sae_id.split('-')[0]
                feature_data = {
                    "modelId": config['api_model_id'],
                    "source": f"{layer}-res_post_32k-oai",
                    "index": str(feature_id)
                }
            else:
                feature_data = {
                    "modelId": config['api_model_id'],
                    "source": sae_id,
                    "index": str(feature_id)
                }
            
            activation_response = requests.post(
                "https://www.neuronpedia.org/api/activation/new",
                headers={"Content-Type": "application/json"},
                json={ 
                    "feature": feature_data,
                    "customText": prompt
                }
            )
            
            if activation_response.status_code != 200:
                raise Exception(f"Failed to call Activation API: {activation_response.text}")

            activation_response_data = activation_response.json()
            
            tokens = activation_response_data.get('tokens', [])
            values = activation_response_data.get('values', [])
            
            tokens_data = [
                {
                    'token': token,
                    'activation_value': float(value),
                    'relative_activation': round((value / max_act_approx) * 10) if max_act_approx != 0 else 0
                }
                for token, value in zip(tokens, values)
            ]
            
            activation_values = [token_data['activation_value'] for token_data in tokens_data]
            max_value = max(activation_values) if activation_values else 0
            min_value = min(activation_values) if activation_values else 0
            max_value_token_index = activation_values.index(max_value) if activation_values else -1
            
            prompt_results.append({
                'max_value': max_value,
                'max_value_token_index': max_value_token_index,
                'min_value': min_value,
                'prompt': prompt,
                'tokens': [
                    {
                        'activation_value': token_data['activation_value'],
                        'relative_activation': token_data['relative_activation'],
                        'token': token_data['token']
                    }
                    for token_data in tokens_data
                ]
            })

        return jsonify({
            'status': 200,
            'data': {
                'llm_model': llm_model,
                'request': {
                    'feature_id': feature_id,
                    'sae_id': sae_id,
                    'prompts': prompts
                },
                'prompt_results': prompt_results
            }
        })

    except Exception as e:
        return jsonify({
            'status': 500,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': str(e)
            }
        }), 500


@validate_bp.route('/api/feature/tokens-analysis', methods=['POST'])
def analyze_tokens():
    data = request.get_json()
    
    required_fields = ['feature_id', 'sae_id', 'selected_prompt_tokens']
    if not all(field in data for field in required_fields):
        return jsonify({
            'status': 400,
            'error': {
                'code': 'MISSING_PARAMETERS',
                'message': 'MISSING_PARAMETERS'
            }
        }), 400

    llm_model = data.get('llm', 'gemma_2_2b')
    
    if llm_model not in ['gemma_2_2b', 'gpt2-small']:
        return jsonify({
            'status': 400,
            'error': {
                'code': 'INVALID_LLM',
                'message': f'INVALID_LLM: {llm_model}'
            }
        }), 400

    try:
        feature_id = data['feature_id']
        sae_id = data['sae_id']
        selected_prompt_tokens = data['selected_prompt_tokens']
        
        _, original_explanation = get_feature_info(sae_id, feature_id, llm_model)
        
        prompt_token_features = {}
        feature_sets = []
        
        for prompt_token in selected_prompt_tokens:
            prompt = prompt_token['prompt']
            token_index = prompt_token['token_index']
            
            prompt_token_id = f"{prompt}_{token_index}"
            
            related_features = get_token_related_features(prompt, token_index, sae_id, llm_model)
            prompt_token_features[prompt_token_id] = {
                'prompt': prompt,
                'token_index': token_index,
                'features': related_features
            }
            
            feature_set = set(f['feature_id'] for f in related_features)
            feature_sets.append(feature_set)
        
        if feature_sets:
            intersection_features = set.intersection(*feature_sets)
            union_features = set.union(*feature_sets)
        else:
            intersection_features = set()
            union_features = set()
        
        response_data = {
            'status': 200,
            'data': {
                'llm_model': llm_model, 
                'original_explanation': original_explanation,
                'related_features_intersection': list(intersection_features),
                'related_features_union': list(union_features),
                'prompt_token_features': prompt_token_features
            }
        }
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'status': 500,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': str(e)
            }
        }), 500


if __name__ == "__main__":
    app.run(debug=True)