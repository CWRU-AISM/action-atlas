# app/routes/input.py

from flask import g, Blueprint, request, jsonify
from openai import OpenAI
from datetime import datetime
import numpy as np
from collections import Counter
from ..utils.errors import APIError
from ..utils.vector_db import VectorDB
from ..config import (
    EXPLANATIONS_EMBEDDING_PATH, VECTOR_DB_PATH,
    EXPLANATIONS_EMBEDDING_PATH_GPT, VECTOR_DB_PATH_GPT
)
from ..utils.db_manager import DBManager
from ..utils.openai_service import OpenAIService

input_bp = Blueprint('input', __name__, url_prefix='/api')

def get_llm_config(llm_model):
    if llm_model == 'gpt2-small':
        return {
            'vector_db_path': VECTOR_DB_PATH_GPT,
            'explanations_embedding_path': EXPLANATIONS_EMBEDDING_PATH_GPT
        }
    else:
        return {
            'vector_db_path': VECTOR_DB_PATH,
            'explanations_embedding_path': EXPLANATIONS_EMBEDDING_PATH
        }

def calculate_sae_distribution(results, top_n):
    """Calculate the SAE distribution of the top_n features"""
    sae_counts = Counter()
    total = min(top_n, len(results))
    
    for result in results[:total]:
        sae_id = result['file_path'].split('/')[-1].replace('.npz', '').split('_', 1)[1].replace('-embedding', '')
        sae_counts[sae_id] += 1
    
    distribution = {
        sae_id: {
            'count': count,
            'percentage': (count / total) * 100
        }
        for sae_id, count in sae_counts.items()
    }
    
    return {
        'total_features': total,
        'distribution': distribution
    }

@input_bp.route('/query/search', methods=['POST'])
def search_query():
    try:
        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k', 2000)
        llm_model = data.get('llm', 'gemma_2_2b') 

        if not query:
            raise APIError('MISSING_QUERY', 'The query text cannot be empty')

        if llm_model not in ['gemma_2_2b', 'gpt2-small']:
            raise APIError('INVALID_LLM', f'INVALID_LLM: {llm_model}')

        llm_config = get_llm_config(llm_model)

        openai_service = OpenAIService.get_instance()
        query_vector = openai_service.get_embedding(query)
        if query_vector is None:
            raise APIError('EMBEDDING_ERROR', 'Unable to generate query vector')

        # Get the vector database instance corresponding to LLM
        db = DBManager.get_instance().get_vector_db(
            llm_config['vector_db_path'], 
            llm_config['explanations_embedding_path']
        )

        results, search_time = db.search(query_vector, k=top_k)

        distances = [result['distance'] for result in results]
        similaritys = [1 - (distance / 2) for distance in distances]
        

        optimized_query = openai_service.generate_optimized_query(query)
        
        optimized_query_vector = openai_service.get_embedding(optimized_query)
        
        optimized_results, optimized_search_time = db.search(optimized_query_vector, k=top_k)
        
        optimized_distances = [result['distance'] for result in optimized_results]
        optimized_similaritys = [1 - (distance / 2) for distance in optimized_distances]
        
        min_similarity = min(min(similaritys), min(optimized_similaritys))
        max_similarity = max(max(similaritys), max(optimized_similaritys))
        
        common_bins = np.linspace(min_similarity, max_similarity, 50)
        
        hist, bin_edges = np.histogram(similaritys, bins=common_bins)
        optimized_hist, _ = np.histogram(optimized_similaritys, bins=common_bins)
        
        sae_distributions = {
            'top_10': calculate_sae_distribution(results, 10),
            'top_100': calculate_sae_distribution(results, 100),
            'top_1000': calculate_sae_distribution(results, 1000)
        }
        
        response = {
            'status': 200,
            'data': {
                'llm_model': llm_model, 
                'similarity_distribution': {
                    'bins': bin_edges.tolist(),
                    'counts': hist.tolist()
                },
                'similarity_distribution_optimized': {
                    'bins': bin_edges.tolist(),
                    'counts': optimized_hist.tolist(),
                    'query': optimized_query
                },
                'features': [
                    {
                        'feature_id': str(result['local_index']),
                        'sae_id': result['file_path'].split('/')[-1].replace('.npz', '').split('_')[1].replace('-embedding', ''),
                        'similarity': 1 - (result['distance'] / 2),
                        'description': result['description']
                    } for result in results
                ],
                'sae_distributions': sae_distributions
            }
        }
        return jsonify(response)

    except APIError as e:
        raise e
    except Exception as e:
        raise APIError('SEARCH_ERROR', str(e))