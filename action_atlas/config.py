# Configuration for Action Atlas
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
ACTION_ATLAS_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = ACTION_ATLAS_ROOT / "data"
DESCRIPTIONS_DIR = DATA_DIR / "descriptions"
PROCESSED_DIR = DATA_DIR / "processed"
CLUSTERING_DIR = PROCESSED_DIR / "clustering"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
VECTOR_DB_DIR = PROCESSED_DIR / "vector_db"

# SAE paths
SAE_BASE_DIR = PROJECT_ROOT / "outputs" / "pi05_saes"
OPENVLA_SAE_BASE_DIR = PROJECT_ROOT / "outputs" / "sae_openvla"

# Supported models
SUPPORTED_MODELS = ["pi05", "openvla"]

# Action dimensions
ACTION_DIMS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
ACTION_DESCRIPTIONS = {
    'x': 'forward/backward movement',
    'y': 'left/right movement',
    'z': 'up/down movement',
    'roll': 'wrist rotation around forward axis',
    'pitch': 'wrist tilt up/down',
    'yaw': 'wrist rotation left/right',
    'gripper': 'gripper open/close'
}

# Layer configurations for Pi05
PI05_LAYERS = {
    'action_expert_layer_0': {'dim': 1024, 'type': 'expert'},
    'action_expert_layer_6': {'dim': 1024, 'type': 'expert'},
    'action_expert_layer_12': {'dim': 1024, 'type': 'expert'},
    'action_expert_layer_13': {'dim': 1024, 'type': 'expert'},
    'action_expert_layer_14': {'dim': 1024, 'type': 'expert'},
    'action_expert_layer_15': {'dim': 1024, 'type': 'expert'},
    'action_expert_layer_16': {'dim': 1024, 'type': 'expert'},
    'action_expert_layer_17': {'dim': 1024, 'type': 'expert'},
    'action_in_proj': {'dim': 1024, 'type': 'projection'},
    'action_out_proj_input': {'dim': 1024, 'type': 'projection'},
}

# Layer configurations for OpenVLA
OPENVLA_LAYERS = {
    f'llm_layer_{i}': {'dim': 4096, 'type': 'llm'} 
    for i in range(32)  # Layers 0-31
}

# Task suites
TASK_SUITES = ['goal', 'spatial', 'object', 'libero_10', 'libero_90']
