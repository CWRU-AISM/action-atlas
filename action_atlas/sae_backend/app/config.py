# app/config.py

import os

# OpenAI config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "your-openai-base-url-here"  

# Neuronpedia config
NEURONPEDIA_API_KEY = "your-neuronpedia-api-key-here"

# Data config
EXPLANATIONS_EMBEDDING_PATH = "/root/autodl-tmp/learning/sae_backend/data/explanations_embedding_new13-17/explanations_embedding"
VECTOR_DB_PATH = "/root/autodl-tmp/learning/sae_backend/data/vector_db_new"
CLUSTERING_PATH = "/root/autodl-tmp/batch_hierarchical_clustering_results_colored_new"
SIMILARITIES_PATH = None

# Data config GPT 2 SMALL
EXPLANATIONS_EMBEDDING_PATH_GPT = "/root/autodl-tmp/gpt"
VECTOR_DB_PATH_GPT = "/root/autodl-tmp/learning/sae_backend/data/vector_db_gpt"
CLUSTERING_PATH_GPT = "/root/autodl-tmp/batch_hierarchical_clustering_results_gpt_colored"
SIMILARITIES_PATH_GPT = None

# Other Configuration
MAX_RETRIES = 3  # OpenAI API call retry count
BATCH_SIZE = 5   # Vector database batch size