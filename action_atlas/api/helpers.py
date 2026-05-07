# Shared helpers for Action Atlas API
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from flask import Blueprint, request, jsonify, send_file, abort, make_response, redirect
from PIL import Image

TIGRIS_BUCKET = os.environ.get("BUCKET_NAME", "")
TIGRIS_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL_S3", "")
TIGRIS_PUBLIC_URL = f"https://{TIGRIS_BUCKET}.fly.storage.tigris.dev" if TIGRIS_BUCKET else ""

VLA_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
VLA_VIDEO_DIR = Path(__file__).parent.parent / "data" / "videos"

PI05_ROLLOUTS_DIR = Path(os.environ.get("PI05_ROLLOUTS_DIR", "/data/robotsteering/pi05_rollouts"))
OPENVLA_ROLLOUTS_DIR = Path(os.environ.get("OPENVLA_ROLLOUTS_DIR", "/data/openvla_rollouts"))
ALOHA_ROLLOUTS_DIR = Path(os.environ.get("ALOHA_ROLLOUTS_DIR", "/data/robotsteering/aloha_rollouts"))
PI05_BASELINE_DIR = Path(os.environ.get("PI05_BASELINE_DIR", "/data/robotsteering/pi05_pertoken_baseline"))
XVLA_ROLLOUTS_DIR = Path(os.environ.get("XVLA_ROLLOUTS_DIR", "/data/batch_1"))
SMOLVLA_ROLLOUTS_DIR = Path(os.environ.get("SMOLVLA_ROLLOUTS_DIR", "/data/smolvla_rollouts"))
SMOLVLA_LIBERO_DIR = Path(os.environ.get("SMOLVLA_LIBERO_DIR", "/data/smolvla_rollouts/smolvla"))
GROOT_ROLLOUTS_DIR = Path(os.environ.get("GROOT_ROLLOUTS_DIR", "/data/groot_rollouts"))
GROOT_ROLLOUTS_DIR_BATCH2 = Path(os.environ.get("GROOT_ROLLOUTS_DIR_BATCH2", "/data/groot_rollouts_batch2"))

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
OFT_ABLATION_VIDEO_DIR = RESULTS_DIR / "experiment_results" / "oft_concept_ablation" / "videos"
OFT_CONCEPT_ID_DIR = RESULTS_DIR / "experiment_results" / "oft_concept_id"
OFT_PROBING_DIR = RESULTS_DIR / "experiment_results" / "oft_probing"
OFT_ABLATION_DIR = RESULTS_DIR / "experiment_results" / "oft_concept_ablation"
OFT_DATA_DIR = OPENVLA_ROLLOUTS_DIR / "openvla_oft"
ACT_RESULTS_DIR = RESULTS_DIR / "act_aloha_interp"
ACT_GRID_ABLATION_DIR = ACT_RESULTS_DIR / "grid_ablation"
ACT_INJECTION_DIR = ACT_RESULTS_DIR / "injection"
ALOHA_DATA_DIR = ALOHA_ROLLOUTS_DIR / "act_aloha_interp"
XVLA_CONCEPT_ID_DIR = XVLA_ROLLOUTS_DIR / "xvla_concept_id"
XVLA_ABLATION_DIR = XVLA_ROLLOUTS_DIR / "xvla_concept_ablation"

VALID_MODELS = ["pi05", "openvla", "xvla", "smolvla", "groot", "act"]

SUITE_MAP = {
    "goal": "libero_goal", "object": "libero_object",
    "spatial": "libero_spatial", "10": "libero_10",
    "long": "libero_long", "libero_goal": "libero_goal",
    "libero_object": "libero_object", "libero_spatial": "libero_spatial",
    "libero_10": "libero_10", "libero_long": "libero_long",
}

MODEL_SCENE_STATE_DIRS = {
    "xvla": "xvla_scene_state", "smolvla": "smolvla_scene_state",
    "groot": "groot_scene_state", "pi05": "pi05_scene_state",
    "openvla": "oft_scene_state",
}

ABLATION_INDEX_FILES = {
    "xvla": "xvla_ablation_index.json",
    "smolvla": "smolvla_ablation_index.json",
    "groot": "groot_ablation_index.json",
    "openvla": "oft_ablation_index.json",
}

MODEL_FILE_MAP = {"openvla": "oft", "openvla_oft": "oft"}

# Shared caches
_json_cache: Dict[str, dict] = {}


def normalize_suite(suite: str) -> str:
    # Normalize suite name to full form (e.g. 'goal' -> 'libero_goal')
    return SUITE_MAP.get(suite, suite)


def suite_short(suite: str) -> str:
    # Get short suite name (e.g. 'libero_goal' -> 'goal')
    return suite.replace("libero_", "") if suite.startswith("libero_") else suite


def detect_model_from_layer(layer_id: str) -> str:
    # Detect model name from layer ID prefix
    if any(x in layer_id for x in ("dit_layer_", "eagle_layer_", "vlsa_layer_")):
        return "groot"
    if any(x in layer_id for x in ("vlm_layer_", "expert_layer_")):
        return "smolvla"
    return "openvla"


def load_json_cached(path: Path, cache_key: str = None) -> Optional[dict]:
    # Load JSON file with caching. Returns None if file doesn't exist
    key = cache_key or str(path)
    if key in _json_cache:
        return _json_cache[key]
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        _json_cache[key] = data
        return data
    except (json.JSONDecodeError, IOError):
        return None


def serve_video_response(path: Path, filename: str = None):
    # Serve a video file with CORS headers
    if not path.exists():
        abort(404)
    name = filename or path.name
    response = make_response(send_file(str(path), mimetype="video/mp4",
                                        as_attachment=False, download_name=name))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


def parse_ablation_filename(filename: str) -> dict:
    """
    Parse ablation video filename into components.

    Format: ablation_L{layer}_{concept_type}_{concept}_task{N}_ep{M}.mp4
    Returns: {layer, concept_type, concept, task, episode, is_baseline}
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    result = {"layer": None, "concept_type": None, "concept": None,
              "task": None, "episode": None, "is_baseline": parts[0] == "baseline"}

    if parts[0] == "baseline":
        for p in parts:
            if p.startswith("task"):
                try: result["task"] = int(p.replace("task", ""))
                except ValueError: pass
            elif p.startswith("ep"):
                try: result["episode"] = int(p.replace("ep", ""))
                except ValueError: pass
        return result

    if len(parts) >= 4 and parts[0] == "ablation" and parts[1].startswith("L"):
        try: result["layer"] = int(parts[1][1:])
        except ValueError: pass
        result["concept_type"] = parts[2]
        task_idx = None
        for i, p in enumerate(parts):
            if p.startswith("task"):
                task_idx = i
                try: result["task"] = int(p.replace("task", ""))
                except ValueError: pass
            elif p.startswith("ep"):
                try: result["episode"] = int(p.replace("ep", ""))
                except ValueError: pass
        if task_idx and task_idx > 3:
            result["concept"] = "_".join(parts[3:task_idx])

    return result


def parse_concept_name(name: str):
    # Parse 'type/name' concept format into (category, subconcept)
    if "/" in name:
        parts = name.split("/", 1)
        return parts[0], parts[1]
    for prefix in ("motion", "object", "spatial", "action_phase"):
        if name.startswith(prefix + "_"):
            return prefix, name[len(prefix) + 1:]
    return "unknown", name


def load_ablation_index(model: str) -> Optional[list]:
    # Load baked ablation index for a model
    fname = ABLATION_INDEX_FILES.get(model)
    if not fname:
        return None
    path = Path(__file__).parent.parent / "data" / fname
    return load_json_cached(path, f"ablation_index_{model}")


def get_vla_config(model: str = 'pi05'):
    # Get configuration for VLA model
    if model in ('openvla', 'openvla_oft'):
        return {
            'model': 'openvla',
            'data_dir': VLA_DATA_DIR / 'openvla',
            'suites': ['libero_goal', 'libero_spatial', 'libero_object', 'libero_10'],
            'layers': [f'layer_{i}' for i in range(32)],
            'layer_prefix': 'layer',
            'sae_width': 32768,
            'concept_id_dir': Path(__file__).parent.parent / 'results' / 'experiment_results' / 'oft_concept_id',
            'ablation_video_dir': Path(__file__).parent.parent / 'results' / 'experiment_results' / 'oft_concept_ablation' / 'videos',
        }
    elif model == 'xvla':
        return {
            'model': 'xvla',
            'data_dir': VLA_DATA_DIR / 'xvla',
            'suites': ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10'],
            'layers': [f'layer_{i}' for i in range(24)],
            'layer_prefix': 'layer',
            'sae_width': 8192,
            'hidden_dim': 1024,
            'concept_id_dir': Path("/data/batch_1/xvla_concept_id"),
            'ablation_dir': Path("/data/batch_1/xvla_concept_ablation"),
            'steering_dir': Path("/data/batch_1/xvla_concept_steering"),
            'feature_descriptions_dir': Path("/data/batch_1/xvla_feature_descriptions"),
            'oracle_probes_dir': Path("/data/batch_1/xvla_matched_oracle_probe"),
            'architecture': 'single_pathway',
        }
    elif model == 'smolvla':
        return {
            'model': 'smolvla',
            'data_dir': VLA_DATA_DIR / 'smolvla',
            'suites': ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10', 'metaworld'],
            'layers': (
                [f'vlm_layer_{i}' for i in range(32)] +
                [f'expert_layer_{i}' for i in range(32)]
            ),
            'layer_prefix': 'vlm_layer',  # primary prefix; expert_layer is secondary
            'sae_width': {'vlm': 7680, 'expert': 3840},
            'hidden_dim': {'vlm': 960, 'expert': 480},
            'concept_id_dir': Path("/data/smolvla_rollouts/concept_id"),
            'metaworld_concept_id_dir': Path(__file__).parent / "data" / "smolvla_metaworld_concept_id",
            'ablation_dir': Path("/data/smolvla_rollouts/concept_ablation"),
            'ffn_dir': Path("/data/smolvla_rollouts/ffn_contrastive"),
            'oracle_probes_dir': Path("/data/smolvla_rollouts/oracle_probes"),
            'architecture': 'dual_pathway_interleaved',
        }
    elif model == 'groot':
        return {
            'model': 'groot',
            'data_dir': VLA_DATA_DIR / 'groot',
            'suites': ['libero_object', 'libero_goal', 'libero_long'],
            'layers': (
                [f'dit_layer_{i}' for i in range(16)] +
                [f'eagle_layer_{i}' for i in range(12)] +
                [f'vlsa_layer_{i}' for i in range(4)]
            ),
            'layer_prefix': 'dit_layer',  # primary prefix; eagle_layer, vlsa_layer are secondary
            'sae_width': 16384,
            'hidden_dim': {'dit': 2048, 'eagle': 2048, 'vlsa': 2048},
            'concept_id_dir': Path(__file__).parent.parent / 'results' / 'experiment_results' / 'groot_concept_id',
            'ablation_dir': Path("/data/groot_rollouts/sae_feature_ablation"),
            'steering_dir': Path("/data/groot_rollouts_batch2/sae_steering"),
            'fraction_to_failure_dir': Path("/data/groot_rollouts_batch2/sae_fraction_to_failure"),
            'temporal_ablation_dir': Path("/data/groot_rollouts_batch2/sae_temporal_ablation"),
            'cross_suite_dir': Path("/data/groot_rollouts_batch2/sae_cross_suite_ablation"),
            'probing_dir': Path("/data/groot_rollouts/sae_probing"),
            'architecture': 'triple_pathway',
        }
    elif model == 'act_aloha' or model == 'act':
        return {
            'model': 'act',
            'data_dir': VLA_DATA_DIR,
            'suites': [],
            'layers': [],
            'layer_prefix': '',
            'sae_width': 0,
            'grid_ablation_dir': Path(__file__).parent.parent / 'results' / 'act_aloha_interp' / 'grid_ablation',
            'injection_dir': Path(__file__).parent.parent / 'results' / 'act_aloha_interp' / 'injection',
            'rollout_dir': ALOHA_DATA_DIR,
            'all_results': Path(__file__).parent.parent / 'results' / 'act_aloha_interp' / 'all_results.json',
        }
    else:  # pi05 (default)
        return {
            'model': model,
            'data_dir': VLA_DATA_DIR,
            'suites': ['concepts', 'spatial', 'libero_10', 'goal', 'object', 'libero_90'],
            'layers': [
                'action_expert_layer_0', 'action_expert_layer_1', 'action_expert_layer_2',
                'action_expert_layer_3', 'action_expert_layer_4', 'action_expert_layer_5',
                'action_expert_layer_6', 'action_expert_layer_7', 'action_expert_layer_8',
                'action_expert_layer_9', 'action_expert_layer_10', 'action_expert_layer_11',
                'action_expert_layer_12', 'action_expert_layer_13', 'action_expert_layer_14',
                'action_expert_layer_15', 'action_expert_layer_16', 'action_expert_layer_17',
                'action_in_proj', 'action_out_proj_input'
            ],
            'layer_prefix': 'action_expert_layer',
            'sae_width': 16384
        }


def load_clustering_data(suite: str, layer: str, model: str = 'pi05',
                         method: str = 'ffn', pathway: str = 'expert') -> Optional[Dict]:
    """
    Load clustering data for a specific layer.

    Args:
        suite: LIBERO suite name (e.g., 'concepts', 'goal', 'spatial', 'object', '10')
        layer: Layer identifier (e.g., 'action_expert_layer_12', 'layer_16',
               'vlm_layer_5', 'expert_layer_3', 'dit_layer_0', 'eagle_layer_2', 'vlsa_layer_1')
        model: Model type ('pi05', 'openvla', 'xvla', 'smolvla', 'groot')
        method: Concept identification method ('contrastive' or 'ffn').
                FFN is only available for Pi0.5 expert pathway.
        pathway: Pi0.5 pathway ('expert' or 'paligemma'). Ignored for other models.
    """
    config = get_vla_config(model)

    # Map layer name to file naming convention
    if model in ('openvla', 'xvla'):
        layer_short = layer  # Uses 'layer_N' directly
    elif model == 'smolvla':
        layer_short = layer  # Uses 'vlm_layer_N' or 'expert_layer_N' directly
    elif model == 'groot':
        layer_short = layer  # Uses 'dit_layer_N', 'eagle_layer_N', 'vlsa_layer_N' directly
    else:
        layer_short = layer.replace('action_expert_', '').replace('action_', '')

    # Determine the base data directory based on method and pathway
    if method == 'contrastive':
        if model == 'openvla':
            base_dir = VLA_DATA_DIR / 'contrastive' / 'oft_single'
        elif model == 'xvla':
            base_dir = VLA_DATA_DIR / 'contrastive' / 'xvla'
        elif model == 'smolvla':
            base_dir = VLA_DATA_DIR / 'contrastive' / 'smolvla'
        elif model == 'groot':
            base_dir = VLA_DATA_DIR / 'contrastive' / 'groot'
        elif pathway == 'paligemma':
            base_dir = VLA_DATA_DIR / 'contrastive' / 'pi05_paligemma'
        else:
            base_dir = VLA_DATA_DIR / 'contrastive' / 'pi05_expert'
    else:
        # FFN method - original root-level data (Pi0.5 expert only)
        if model in ('openvla', 'xvla', 'smolvla', 'groot'):
            base_dir = config['data_dir']  # Model-specific dir
        else:
            base_dir = VLA_DATA_DIR  # Root processed/ dir

    # Normalize suite name: Pi0.5 and OFT files use short names (goal, object, spatial, 10)
    # while X-VLA, SmolVLA, GR00T use libero_ prefix (libero_goal, libero_object, etc.)
    suite_short = suite.replace('libero_', '') if suite.startswith('libero_') else suite
    suite_long = f"libero_{suite}" if not suite.startswith('libero_') else suite

    # Map frontend suite names to clustering file suite names
    # SimplerEnv suites use short names in NPZ files (widowx, google_robot)
    # MetaWorld suites collapse to just 'metaworld'
    suite_file_map = {
        'simplerenv_widowx': 'widowx',
        'simplerenv_google_robot': 'google_robot',
        'metaworld_easy': 'metaworld',
        'metaworld_medium': 'metaworld',
        'metaworld_hard': 'metaworld',
        'metaworld_very_hard': 'metaworld',
    }
    suite_file = suite_file_map.get(suite, None)

    # Try multiple file naming conventions and locations
    possible_paths = []
    # If we have a mapped suite name, try that first
    if suite_file:
        possible_paths.append(base_dir / f"hierarchical_clustering_{layer_short}_{suite_file}.npz")
    possible_paths.extend([
        # Direct match with suite
        base_dir / f"hierarchical_clustering_{layer_short}_{suite}.npz",
        # Try without libero_ prefix (Pi0.5, OFT use short names)
        base_dir / f"hierarchical_clustering_{layer_short}_{suite_short}.npz",
        # Try with libero_ prefix (X-VLA, SmolVLA, GR00T use long names)
        base_dir / f"hierarchical_clustering_{layer_short}_{suite_long}.npz",
        # With _goal suffix (some files use 'goal' instead of 'concepts')
        base_dir / f"hierarchical_clustering_{layer_short}_goal.npz",
        # With suite subdirectory
        base_dir / suite / f"hierarchical_clustering_{layer_short}_{suite}.npz",
        base_dir / suite / f"hierarchical_clustering_{layer_short}_goal.npz",
        # OpenVLA format (no suite suffix)
        base_dir / f"hierarchical_clustering_{layer_short}.npz",
    ])

    filepath = None
    for path in possible_paths:
        print(f"Looking for clustering data at: {path}")
        if path.exists():
            filepath = path
            print(f"  -> Found!")
            break

    if filepath is None:
        print(f"  -> No data file found for layer {layer}, suite {suite}, method {method}, pathway {pathway}")
        return None

    data = np.load(filepath, allow_pickle=True)
    return {
        'coords': data['coords'],
        'indices': data['indices'],
        'descriptions': data['descriptions'],
        'cluster_data': {
            level: {
                'labels': data[f'cluster_labels_{level}'],
                'colors': data[f'cluster_colors_{level}'],
                'centers': data[f'cluster_centers_{level}'],
                'topics': data[f'topic_words_{level}'].item() if f'topic_words_{level}' in data else {},
                'topic_scores': data[f'topic_word_scores_{level}'].item() if f'topic_word_scores_{level}' in data else {}
            }
            for level in [10, 30] if f'cluster_labels_{level}' in data
        }
    }


def load_feature_metadata(suite: str) -> List[Dict]:
    # Load feature metadata for a suite
    config = get_vla_config()
    filepath = config['data_dir'] / suite / "feature_metadata.json"

    if not filepath.exists():
        return []

    with open(filepath) as f:
        return json.load(f)


def load_concept_features() -> Optional[Dict]:
    # Load concept-to-feature mapping data (cached)
    path = Path(__file__).parent.parent / "data" / "concept_features.json"
    return load_json_cached(path, "concept_features")



