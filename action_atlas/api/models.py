# Action Atlas API - models routes
from flask import Blueprint, request, jsonify, send_file, abort, make_response, redirect
from .helpers import *

models_bp = Blueprint("models", __name__)


@models_bp.route('/api/vla/models', methods=['GET'])
def get_vla_models():
    # Get available VLA models and their configurations
    model = request.args.get('model', 'pi05')
    config = get_vla_config(model)
    # Convert Path to string for JSON serialization
    serializable_config = {
        'model': config['model'],
        'data_dir': str(config['data_dir']),
        'suites': config['suites'],
        'layers': config['layers']
    }
    return jsonify({
        'status': 200,
        'data': {
            'models': ['pi05', 'openvla', 'xvla', 'smolvla', 'groot', 'act'],
            'current_config': serializable_config,
            'model_info': {
                'pi05': {
                    'name': 'Pi0.5',
                    'description': 'Physical Intelligence Pi0.5 VLA model (3B, flow-matching)',
                    'status': 'available',
                    'layers': 18,
                    'sae_width': 16384,
                    'architecture': 'flow-matching (50 denoising steps)',
                    'hidden_dim': 1024,
                },
                'openvla': {
                    'name': 'OpenVLA-OFT',
                    'description': 'OpenVLA 7B with OFT fine-tuning on LIBERO (L1 regression)',
                    'status': 'available',
                    'layers': 32,
                    'sae_width': 32768,
                    'architecture': 'continuous L1 regression via MLP action head',
                    'hidden_dim': 4096,
                    'suites': ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10'],
                    'concept_id_files': 128,
                    'ablation_videos': 11892,
                    'ablation_jsons': 18,
                    'concept_ablation_pairs': 1810,
                },
                'xvla': {
                    'name': 'X-VLA',
                    'description': 'X-VLA 1B with Florence-2 backbone, 24 TransformerBlocks, flow-matching (LIBERO + SimplerEnv)',
                    'status': 'available',
                    'layers': 24,
                    'sae_width': 8192,
                    'architecture': 'single_pathway (Florence-2 + flow-matching)',
                    'hidden_dim': 1024,
                    'suites': ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10'],
                    'environments': ['libero', 'simplerenv_widowx', 'simplerenv_google_robot'],
                },
                'smolvla': {
                    'name': 'SmolVLA',
                    'description': 'SmolVLA 450M with interleaved VLM (960-dim) + Expert (480-dim) pathways (LIBERO + MetaWorld)',
                    'status': 'available',
                    'layers': 64,
                    'sae_width': {'vlm': 7680, 'expert': 3840},
                    'architecture': 'dual_pathway_interleaved (SmolVLM)',
                    'hidden_dim': {'vlm': 960, 'expert': 480},
                    'suites': ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10', 'metaworld'],
                    'environments': ['libero', 'metaworld'],
                    'pathway_types': ['vlm', 'expert'],
                },
                'groot': {
                    'name': 'GR00T N1.5',
                    'description': 'GR00T N1.5 3B with DiT (16L) + Eagle LM (12L) + VL-SA (4L) triple-pathway',
                    'status': 'available',
                    'layers': 32,
                    'sae_width': 16384,
                    'architecture': 'triple_pathway (DiT + Eagle LM + VL-SA, diffusion)',
                    'hidden_dim': {'dit': 2048, 'eagle': 2048, 'vlsa': 2048},
                    'suites': ['libero_object', 'libero_goal', 'libero_long'],
                    'pathway_types': ['dit', 'eagle', 'vlsa'],
                },
                'act': {
                    'name': 'ACT-ALOHA',
                    'description': 'Action Chunking Transformer on ALOHA sim (CVAE decoder)',
                    'status': 'available',
                    'layers': 0,
                    'sae_width': 0,
                    'architecture': 'CVAE decoder (action chunking, continuous)',
                    'hidden_dim': 512,
                    'tasks': ['AlohaInsertion-v0', 'AlohaTransferCube-v0'],
                    'grid_ablation': True,
                    'injection': True,
                }
            }
        }
    })


@models_bp.route('/api/vla/suites', methods=['GET'])
def get_vla_suites():
    # Get available task suites for VLA
    model = request.args.get('model', 'pi05')
    config = get_vla_config(model)
    available_suites = []

    for suite in config['suites']:
        suite_dir = config['data_dir'] / suite
        if suite_dir.exists():
            available_suites.append(suite)
        elif model in ('smolvla', 'xvla', 'groot'):
            # For models with experiment_results JSON, suites may not have
            # dedicated directories but still have data in the aggregated JSON
            data_dir = Path(__file__).parent / "data"
            exp_path = data_dir / f"experiment_results_{model}.json"
            if exp_path.exists():
                available_suites.append(suite)

    return jsonify({
        'status': 200,
        'data': {
            'suites': available_suites
        }
    })


@models_bp.route('/api/vla/layers', methods=['GET'])
def get_vla_layers():
    # Get available layers for a given suite
    suite = request.args.get('suite', 'spatial')
    model = request.args.get('model', 'pi05')
    config = get_vla_config(model)

    available_layers = []
    suite_dir = config['data_dir'] / suite

    if suite_dir.exists():
        for layer in config['layers']:
            layer_short = layer.replace('action_expert_', '').replace('action_', '')
            filename = f"hierarchical_clustering_{layer_short}_{suite}.npz"
            if (suite_dir / filename).exists():
                available_layers.append({
                    'id': f"{layer}-{suite}",
                    'name': layer,
                    'suite': suite
                })

    return jsonify({
        'status': 200,
        'data': {
            'layers': available_layers
        }
    })


