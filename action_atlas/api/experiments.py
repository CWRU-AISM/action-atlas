# Action Atlas API - experiments routes
import json
import traceback
from pathlib import Path
from typing import Dict, Optional

from flask import Blueprint, request, jsonify

from .helpers import (
    VALID_MODELS, MODEL_FILE_MAP, XVLA_CONCEPT_ID_DIR, XVLA_ABLATION_DIR,
    OFT_ABLATION_VIDEO_DIR, OFT_CONCEPT_ID_DIR, OFT_DATA_DIR,
    ACT_GRID_ABLATION_DIR, ACT_INJECTION_DIR, ACT_RESULTS_DIR,
    PI05_ROLLOUTS_DIR, VLA_VIDEO_DIR,
    load_json_cached, load_concept_features, get_vla_config,
)
from .data_loaders import load_ablation_results, parse_success_from_path, find_results_json_for_video, extract_video_results
from .success_tracking import _load_layer_connections_openvla, _load_layer_connections_pi05
from .data_loaders import *
from .success_tracking import *
from .concept_helpers import _load_concept_counts_for_model
from .videos import load_video_index

experiments_bp = Blueprint("experiments", __name__)

# Directory constants not yet in helpers (defined in features.py)
XVLA_STEERING_DIR = Path("/data/batch_1/xvla_concept_steering")
SMOLVLA_CONCEPT_ID_DIR = Path("/data/smolvla_rollouts/concept_id")
SMOLVLA_ABLATION_DIR = Path("/data/openvla_rollouts/smolvla/concept_ablation")
GROOT_ABLATION_DIR = Path("/data/groot_rollouts/sae_feature_ablation")
GROOT_STEERING_DIR = Path("/data/groot_rollouts_batch2/sae_steering")
GROOT_PROBING_DIR = Path("/data/groot_rollouts/sae_probing")

# Module-level data directory (api/data -- legacy path resolution)
_API_DATA_DIR = Path(__file__).parent / "data"

# Cache for experiment results (loaded once on first request)
_experiment_results_cache: Dict = {}


# Shared helpers


from .experiment_helpers import *

@experiments_bp.route('/api/vla/layer_metrics', methods=['GET'])
def get_layer_metrics():
    """
    Get concept-based metrics for all layers.

    Query params:
        model: 'pi05_expert' | 'pi05_paligemma' | 'openvla_oft' | 'all' (default: 'pi05_expert')
    """
    data = load_concept_features()
    if data is None:
        return jsonify({
            'status': 404,
            'error': {'code': 'DATA_NOT_FOUND', 'message': 'Concept features data not found'}
        }), 404

    model = request.args.get('model', 'pi05_expert')

    prefix_map = {
        'pi05_expert': 'action_expert_layer_',
        'pi05_paligemma': 'paligemma_layer_',
        'openvla_oft': 'openvla_oft_layer_',
        'xvla': 'xvla_layer_',
        'smolvla_vlm': 'smolvla_vlm_layer_',
        'smolvla_expert': 'smolvla_expert_layer_',
        'groot_dit': 'groot_dit_layer_',
        'groot_eagle': 'groot_eagle_layer_',
        'groot_vlsa': 'groot_vlsa_layer_',
    }

    if model == 'all':
        prefixes = list(prefix_map.values())
    else:
        prefixes = [prefix_map.get(model, 'action_expert_layer_')]

    layer_configs = []
    for layer_key in data.keys():
        if layer_key.startswith('_'):
            continue
        for prefix in prefixes:
            if layer_key.startswith(prefix):
                try:
                    layer_num = int(layer_key[len(prefix):])
                    layer_configs.append((layer_num, layer_key))
                except ValueError:
                    pass
    layer_configs.sort(key=lambda x: (x[1].split('_layer_')[0], x[0]))

    for legacy_key in ['action_in_proj', 'action_out_proj_input']:
        if legacy_key in data and model in ('pi05_expert', 'all'):
            max_idx = max((lc[0] for lc in layer_configs), default=-1)
            layer_configs.append((max_idx + 1, legacy_key))

    all_motion = set()
    all_object = set()
    all_spatial = set()
    for _, layer_name in layer_configs:
        layer_data = data.get(layer_name, {})
        all_motion.update(layer_data.get('motion', {}).keys())
        all_object.update(layer_data.get('object', {}).keys())
        all_spatial.update(layer_data.get('spatial', {}).keys())

    layers = []
    for layer_idx, layer_name in layer_configs:
        layer_data = data.get(layer_name, {})

        motion = layer_data.get('motion', {})
        obj = layer_data.get('object', {})
        spatial = layer_data.get('spatial', {})
        action_phase = layer_data.get('action_phase', {})

        total_motion = sum(c.get('concept_features', 0) for c in motion.values())
        total_object = sum(c.get('concept_features', 0) for c in obj.values())
        total_spatial = sum(c.get('concept_features', 0) for c in spatial.values())
        total_action_phase = sum(c.get('concept_features', 0) for c in action_phase.values())

        entry = {
            'id': f'{layer_name}-concepts',
            'type': 'RES',
            'layer': layer_idx,
            'layer_name': layer_name,
            'total_motion': {'value': total_motion, 'rank': 1},
            'total_object': {'value': total_object, 'rank': 1},
            'total_spatial': {'value': total_spatial, 'rank': 1},
            'total_action_phase': {'value': total_action_phase, 'rank': 1},
            'top_10_score': {'value': min(1.0, total_motion / 500), 'rank': 1},
            'top_100_score': {'value': min(1.0, total_object / 500), 'rank': 1},
            'top_1000_score': {'value': min(1.0, total_spatial / 500), 'rank': 1},
        }

        for concept_name in sorted(all_motion):
            entry[f'{concept_name}_features'] = {'value': motion.get(concept_name, {}).get('concept_features', 0), 'rank': 1}
        for concept_name in sorted(all_object):
            entry[f'{concept_name}_features'] = {'value': obj.get(concept_name, {}).get('concept_features', 0), 'rank': 1}
        for concept_name in sorted(all_spatial):
            entry[f'{concept_name}_features'] = {'value': spatial.get(concept_name, {}).get('concept_features', 0), 'rank': 1}
        for concept_name in sorted(action_phase.keys()):
            entry[f'{concept_name}_features'] = {'value': action_phase.get(concept_name, {}).get('concept_features', 0), 'rank': 1}

        for cat_name, cat_data in [('motion', motion), ('object', obj), ('spatial', spatial)]:
            if cat_data:
                max_d = max(c.get('max_cohens_d', 0) for c in cat_data.values())
                entry[f'max_cohens_d_{cat_name}'] = {'value': round(max_d, 3), 'rank': 1}

        layers.append(entry)

    if layers:
        rank_metrics = [k for k in layers[0] if isinstance(layers[0].get(k), dict) and 'rank' in layers[0][k]]
        for metric in rank_metrics:
            sorted_layers = sorted(enumerate(layers), key=lambda x: x[1][metric]['value'], reverse=True)
            for rank, (idx, _) in enumerate(sorted_layers, 1):
                layers[idx][metric]['rank'] = rank

    metadata = data.get('_metadata', {})

    return jsonify({
        'status': 200,
        'data': layers,
        'metadata': {
            'concept_method': metadata.get('concept_method', 'unknown'),
            'model': model,
            'n_layers': len(layers),
            'available_models': list(prefix_map.keys())
        }
    })


# Prompts

@experiments_bp.route('/api/vla/prompts', methods=['GET'])
def get_vla_prompts():
    # Get prompts/task descriptions for autocomplete
    model = request.args.get('model', 'pi05')
    video_index = load_video_index(model)

    if video_index is None:
        return jsonify({
            'status': 404,
            'error': {'code': 'INDEX_NOT_FOUND', 'message': f'No video index found for model: {model}'}
        }), 404

    prompts = video_index.get('prompts_for_autocomplete', [])
    return jsonify({'prompts': prompts, 'model': model, 'total': len(prompts)})


# Experiments

@experiments_bp.route('/api/vla/experiments', methods=['GET'])
def get_vla_experiments():
    # Get list of experiment types with counts and descriptions
    model = request.args.get('model', 'pi05')

    experiment_definitions = {
        'counterfactual': {'description': 'Prompt manipulation experiments testing concept understanding', 'category': 'interpretability'},
        'ablation': {'description': 'Feature ablation studies removing specific SAE features', 'category': 'causal'},
        'steering': {'description': 'Feature steering experiments amplifying or suppressing concepts', 'category': 'control'},
        'baseline': {'description': 'Baseline runs without any intervention', 'category': 'control'},
        'reconstruction': {'description': 'SAE reconstruction quality analysis', 'category': 'validation'},
        'temporal': {'description': 'Time-based ablation during specific action phases', 'category': 'causal'},
        'fractional': {'description': 'Partial feature ablation with varying percentages', 'category': 'causal'},
    }

    experiments = {}

    if model == 'pi05':
        ablation_results_dir = _API_DATA_DIR / "libero_10" / "ablation_results"
        steering_results_dir = _API_DATA_DIR / "libero_10" / "steering_results"
        video_dir = _API_DATA_DIR / "videos" / "goal"

        for exp_key, directory, pattern in [
            ('temporal', ablation_results_dir, "temporal_ablation_*.json"),
            ('fractional', ablation_results_dir, "fractional_ablation_*.json"),
            ('steering', steering_results_dir, "steering_*.json"),
        ]:
            count = _count_files(directory, pattern)
            if count > 0:
                experiments[exp_key] = {'count': count, **experiment_definitions.get(exp_key, {})}

        for exp_key, subdir_name in [('ablation', 'ablation'), ('baseline', 'baseline'), ('reconstruction', 'reconstruction')]:
            count = _count_files(video_dir / subdir_name, "*.mp4")
            if count > 0:
                experiments[exp_key] = {'count': count, **experiment_definitions.get(exp_key, {})}

        if (_API_DATA_DIR / "concept_features.json").exists():
            experiments['counterfactual'] = {'count': 2229, **experiment_definitions.get('counterfactual', {})}

    elif model in ('openvla', 'openvla_oft'):
        experiments = {}
        entry = _dir_experiment_entry(OFT_CONCEPT_ID_DIR, "oft_concept_id_*.json",
            'Contrastive concept identification across 32 layers x 4 suites', 'interpretability')
        if entry:
            experiments['concept_identification'] = entry
        entry = _dir_experiment_entry(OFT_ABLATION_VIDEO_DIR, "*.mp4",
            'Concept ablation rollout videos across 4 LIBERO suites', 'causal', recursive=True)
        if entry:
            experiments['concept_ablation'] = entry
        experiments['counterfactual'] = {'count': 1110, 'description': 'Counterfactual prompting episodes across 4 suites', 'category': 'interpretability'}
        experiments['cross_task_injection'] = {'count': 168, 'description': 'Cross-task activation injection pairs', 'category': 'causal'}
        experiments['same_scene_injection'] = {'count': 400, 'description': 'Same-scene activation injection episodes', 'category': 'causal'}
        experiments['sae_validation'] = {'count': 120, 'description': 'SAE hook validation episodes (119/120 = 99.2% success)', 'category': 'validation'}

    elif model == 'xvla':
        experiments = {}
        entry = _dir_experiment_entry(XVLA_CONCEPT_ID_DIR, "*.json", 'Contrastive concept identification across 24 layers', 'interpretability')
        if entry:
            experiments['concept_identification'] = entry
        entry = _dir_experiment_entry(XVLA_ABLATION_DIR, "*.json", 'Concept ablation experiments (single pathway, 24 layers)', 'causal', recursive=True)
        if entry:
            experiments['concept_ablation'] = entry
        entry = _dir_experiment_entry(XVLA_STEERING_DIR, "*.json", 'Concept steering experiments', 'control', recursive=True)
        if entry:
            experiments['concept_steering'] = entry

    elif model == 'smolvla':
        experiments = {}
        entry = _dir_experiment_entry(SMOLVLA_CONCEPT_ID_DIR, "*.json", 'Contrastive concept identification across 64 layers (32 VLM + 32 expert)', 'interpretability')
        if entry:
            experiments['concept_identification'] = entry
        entry = _dir_experiment_entry(SMOLVLA_ABLATION_DIR, "*.json", 'Concept ablation experiments (dual pathway, interleaved)', 'causal', recursive=True)
        if entry:
            experiments['concept_ablation'] = entry

    elif model == 'groot':
        experiments = {}
        entry = _dir_experiment_entry(GROOT_ABLATION_DIR, "*.json", 'SAE feature ablation across DiT (16L) + Eagle (12L) + VL-SA (4L)', 'causal', recursive=True)
        if entry:
            experiments['concept_ablation'] = entry
        entry = _dir_experiment_entry(GROOT_STEERING_DIR, "*.json", 'SAE feature steering experiments', 'control', recursive=True)
        if entry:
            experiments['concept_steering'] = entry
        entry = _dir_experiment_entry(GROOT_PROBING_DIR, "*.json", 'Linear probing across triple-pathway layers', 'interpretability', recursive=True)
        if entry:
            experiments['probing'] = entry

    elif model in ('act', 'act_aloha'):
        experiments = {}
        entry = _dir_experiment_entry(ACT_GRID_ABLATION_DIR, "*.json", 'Grid ablation (4x4 region masking) across 2 tasks', 'causal')
        if entry:
            experiments['grid_ablation'] = entry
        entry = _dir_experiment_entry(ACT_INJECTION_DIR, "*.json", 'Cross-task and same-task activation injection', 'causal')
        if entry:
            experiments['injection'] = entry
        experiments['baseline'] = {'count': 40, 'description': 'Baseline episodes (20 per task)', 'category': 'control'}

    return jsonify({
        'status': 200,
        'data': {'model': model, 'experiments': experiments, 'total_experiment_types': len(experiments)}
    })


# Findings

@experiments_bp.route('/api/vla/findings', methods=['GET'])
def get_vla_findings():
    # Get key findings from the VLA interpretability research
    model = request.args.get('model', 'pi05')

    findings_file = _API_DATA_DIR / "findings.json"
    all_findings = load_json_cached(findings_file, "findings")
    if all_findings and model in all_findings:
        return jsonify({'status': 200, 'data': all_findings[model]})

    if model == 'pi05':
        findings = {
            'model': 'pi05', 'model_name': 'Pi0.5 VLA',
            'summary': '20 validated findings from 6,000+ LIBERO episodes. Key discoveries: pathway specialization (PaliGemma=WHAT, Expert=HOW), width-dependent ablation catastrophe in 1024-dim space, and phase-specific steering.',
            'key_findings': [
                {'id': 'pathway_specialization', 'title': 'Pathway Specialization: PaliGemma=WHAT, Expert=HOW', 'description': 'PaliGemma (VLM backbone) encodes WHAT: goals (76.4%), objects (100% from L1), semantic context. It does NOT encode dynamics (R^2~0) or success prediction (AUC=0.50). The Gemma Expert (action head) encodes HOW: world model (R^2=0.45), success prediction (AUC=0.93), motor primitives. Cross-task injection shows distinct failure modes: PaliGemma -> passive stalling (520 steps), Expert -> active wrong behavior (231-302 steps).', 'evidence': 'Cross-task injection, linear probing, and success prediction across 6,000+ LIBERO episodes.', 'confidence': 'high', 'category': 'architecture'},
                {'id': 'concept_emergence', 'title': 'Concept Emergence Across Layers', 'description': 'SAE features encode task-relevant concepts including objects (bowl, plate, cabinet), actions (put, open, push), spatial relations (on, in, top), and action phases (approach, grasp, lift, transport, lower, release).', 'evidence': 'Analysis of 16,384 SAE features across 18 layers reveals consistent concept encoding.', 'confidence': 'high', 'category': 'interpretability'},
                {'id': 'layer_specialization', 'title': 'Layer Specialization', 'description': 'Middle layers (10-14) show strongest concept representations, while early layers encode low-level visual features and late layers prepare action outputs.', 'evidence': 'Layer 12 shows peak performance for most concept ablations.', 'confidence': 'high', 'category': 'architecture'},
                {'id': 'width_dependent_ablation', 'title': 'Width-Dependent Ablation Catastrophe', 'description': "Ablating just 30 features in Pi0.5's 1024-dim hidden space causes severe failure (-60 to -100pp). This contrasts sharply with OpenVLA-OFT (4096-dim) where 91.6% of ablations have zero effect, demonstrating that narrower representations concentrate critical information.", 'evidence': 'Concept ablation experiments comparing Pi0.5 (1024-dim) vs OpenVLA-OFT (4096-dim).', 'confidence': 'high', 'category': 'causal'},
                {'id': 'causal_asymmetry', 'title': 'Causal Asymmetry: Ablation Tolerated, Steering Overwhelms', 'description': 'Causal asymmetry: ablating concept features is tolerated (p=0.975), but boosting at 7x+ overwhelms the model (-14% at 7x). Ablating step 0 alone causes 0% success.', 'evidence': 'Ablation and steering experiments across concept features.', 'confidence': 'high', 'category': 'causal'},
                {'id': 'phase_specific_steering', 'title': 'Phase-Specific Steering (Transport Phase)', 'description': 'Steering transport-phase features has a statistically significant effect (p=0.013), demonstrating that action phases can be selectively influenced. This is the first demonstration of targeted phase manipulation in VLA models.', 'evidence': 'Targeted steering of transport features on LIBERO-10 tasks.', 'confidence': 'medium', 'category': 'causal'},
                {'id': 'phase_sensitivity', 'title': 'Action Phase Sensitivity', 'description': 'Different action phases show varying sensitivity to feature intervention. Step 0 (initialization) is critical: ablating it alone causes total failure, while step 1+ ablation has no effect.', 'evidence': 'Step-specific ablation study: step 0 = 0% success, step 1 = 100% success.', 'confidence': 'medium', 'category': 'temporal'},
                {'id': 'steering_effectiveness', 'title': 'Feature Steering Control', 'description': 'Amplifying or suppressing specific SAE features enables controlled manipulation of robot behavior without retraining.', 'evidence': 'Steering experiments show behavioral changes proportional to steering strength.', 'confidence': 'medium', 'category': 'control'},
                {'id': 'concept_selectivity', 'title': 'Concept Selectivity', 'description': 'Some concepts show high selectivity (affecting only relevant tasks) while others have broader effects. "Push" features show 8.9% selectivity advantage.', 'evidence': 'Ablation comparison between concept-specific tasks and other tasks.', 'confidence': 'medium', 'category': 'interpretability'},
                {'id': 'vision_dominance', 'title': 'Visual Pathway Dominance', 'description': 'Null injection (removing language) still achieves 73% success for Pi0.5, indicating heavy reliance on visual pathways. Same-scene injection from different episodes further confirms vision-driven behavior.', 'evidence': 'Null injection and same-scene injection experiments across 4 LIBERO suites.', 'confidence': 'high', 'category': 'injection'},
            ],
            'metrics': {'total_episodes': 6000, 'counterfactual_episodes': 2229, 'cross_task_episodes': 360, 'same_scene_episodes': 240, 'null_injection_episodes': 900, 'vision_perturbation_episodes': 2000, 'temporal_episodes': 1500, 'total_features_analyzed': 16384, 'layers_analyzed': 18, 'concepts_identified': 75, 'task_suites': ['libero_10', 'libero_goal', 'libero_spatial', 'libero_object'], 'validated_findings': 20},
            'limitations': ['Results specific to LIBERO simulation benchmark - physical robot validation in progress', 'SAE reconstruction introduces some information loss (~5%)', 'Causal claims limited to tested intervention types'],
        }
    elif model == 'openvla':
        findings = {
            'model': 'openvla', 'model_name': 'OpenVLA-OFT',
            'summary': 'Cross-architecture validation using OpenVLA-OFT (7B, L1 regression). Key contrasts with Pi0.5: null injection is highly disruptive (14% vs 73% recovery), same-scene injection shows suite-dependent effects, and cross-task injection primarily affects libero_goal.',
            'key_findings': [
                {'id': 'null_injection_disruptive', 'title': 'Null Injection Highly Disruptive', 'description': 'Removing the language prompt causes only 14% task success (vs 73% for Pi0.5), suggesting OpenVLA-OFT relies more on language conditioning.', 'evidence': 'Null injection episodes across 4 LIBERO suites. Only 14% task success vs 73% for Pi0.5.', 'confidence': 'high', 'category': 'injection'},
                {'id': 'cross_task_suite_dependent', 'title': 'Cross-Task Effects Suite-Dependent', 'description': 'Cross-task activation injection primarily degrades libero_goal (-40pp) while other suites show near-zero effect.', 'evidence': '168 cross-task injection pairs.', 'confidence': 'high', 'category': 'injection'},
                {'id': 'sae_cross_model', 'title': 'SAE Generalizes Across Fine-Tuned Models', 'description': 'A single SAE trained on one suite achieves 99.2% task success when applied to 4 different fine-tuned OpenVLA-OFT models.', 'evidence': '119/120 validation episodes (99.2% success) across 4 fine-tuned OFT models.', 'confidence': 'high', 'category': 'sae'},
                {'id': 'language_suite_dependent', 'title': 'Language Sensitivity is Suite-Dependent', 'description': 'Language conditioning has near-zero effect on libero_object but strong effects on libero_goal, suggesting task complexity modulates language reliance.', 'evidence': '1,110 counterfactual prompting episodes.', 'confidence': 'medium', 'category': 'language'},
                {'id': 'width_resilience', 'title': 'Width-Resilience: Architectural Robustness Varies', 'description': 'Pi0.5 (1024-dim) shows severe failure under ablation, while OpenVLA-OFT (4096-dim) is resilient. This suggests that wider representations distribute concepts more redundantly.', 'evidence': 'Pi0.5 ablation causes 0% success on all tasks; OFT ablation has near-zero effect (p=0.975).', 'confidence': 'high', 'category': 'architecture'},
                {'id': 'phase_specific_steering', 'title': 'Phase-Specific Steering (Transport Phase)', 'description': 'Steering transport-phase features has a statistically significant effect (p=0.013), demonstrating that action phases can be selectively influenced.', 'evidence': 'Targeted steering of transport features on LIBERO-10.', 'confidence': 'medium', 'category': 'causal'},
                {'id': 'same_scene_negative', 'title': 'Same-Scene Injection Hurts All Suites', 'description': 'Injecting same-scene activations from a different episode degrades performance by -17pp to -57pp across all 4 LIBERO suites, indicating high sensitivity to episode-specific visual context.', 'evidence': 'OFT same-scene injection across 4 suites, 10 tasks each.', 'confidence': 'high', 'category': 'injection'},
                {'id': 'cross_model_sae', 'title': 'Cross-Architecture Validation: 0% Cross-Task', 'description': 'Both Pi0.5 and OpenVLA-OFT collapse to 0% success under cross-task injection, confirming that VLA internal representations are highly task-specific.', 'evidence': 'Cross-task injection across both architectures.', 'confidence': 'high', 'category': 'injection'},
                {'id': 'concept_id_128', 'title': 'Concept Identification Across 32 Layers', 'description': "Contrastive concept identification (Cohen's d x frequency) applied to all 32 OFT layers across 4 suites yields 128 concept ID JSON files with 15 unique concept categories, revealing which concepts are encoded at each layer.", 'evidence': '128 JSON files (32 layers x 4 suites) in results/experiment_results/oft_concept_id/', 'confidence': 'high', 'category': 'interpretability'},
                {'id': 'concept_ablation_resilience', 'title': 'Concept Ablation: Width Implies Resilience', 'description': '91.6% of 1,810 task-concept ablation pairs show zero effect (< 10pp change). Only 8.4% show >10pp change. This contrasts with Pi0.5 where ablation is devastating (-60 to -100pp). The 4096-dim OFT hidden space distributes concepts redundantly.', 'evidence': '18 ablation JSON files, 11,892 ablation videos across 4 LIBERO suites, 1,810 task-concept pairs evaluated.', 'confidence': 'high', 'category': 'causal'},
                {'id': 'world_model_distributed', 'title': 'World Model Distributed Across All 32 Layers', 'description': 'Linear probing reveals world-model information (R^2 > 0.60) is distributed across all 32 OFT layers, not concentrated in specific layers. This contrasts with Pi0.5 where 94% (17/18) of expert layers carry world-model signal.', 'evidence': 'Multi-layer R^2 probing across 32 layers for spatial (R^2=0.45), object (R^2=0.34) predictions.', 'confidence': 'medium', 'category': 'architecture'},
            ],
            'metrics': {'total_episodes': 13354, 'rollout_trials': 12354, 'activation_captures': 1000, 'concepts_identified': 80, 'concept_categories': 15, 'layers_analyzed': 32, 'sae_checkpoints': 32, 'concept_id_files': 128, 'ablation_videos': 11892, 'ablation_jsons': 18, 'concept_ablation_pairs': 1810, 'zero_effect_pct': 91.6, 'significant_effect_pct': 8.4, 'sae_validation': '119/120 (99.2%)'},
            'limitations': ['Results specific to LIBERO simulation benchmark - physical robot validation in progress', 'Single SAE layer (L16) tested in rollouts; other layers untested', 'Cross-task injection effects vary by suite'],
        }
    elif model in ('act', 'act_aloha'):
        findings = {
            'model': 'act_aloha', 'model_name': 'ACT-ALOHA',
            'summary': 'ACT-ALOHA (CVAE decoder) provides cross-architecture validation. Grid ablation reveals spatially structured vision dependence. Residual connections completely wash out injected activations.',
            'key_findings': [
                {'id': 'grid_spatial', 'title': 'Spatially Structured Vision Dependence', 'description': 'Grid ablation (4x4 masking) reveals that specific image regions are critical for task success. For TransferCube, the workspace center is essential; for Insertion, the peg region is critical.', 'evidence': '960 trajectories across 2 tasks x 16 grid cells x 20 episodes.', 'confidence': 'high', 'category': 'vision'},
                {'id': 'residual_washout', 'title': 'Residual Connections Wash Out Injections', 'description': 'ACT-ALOHA residual connections completely negate injected activations (cos_to_baseline=1.0). Cross-task injection has ZERO effect on behavior.', 'evidence': 'Cross-task and same-task injection experiments.', 'confidence': 'high', 'category': 'injection'},
                {'id': 'noise_resilience', 'title': 'Noise Resilience Varies by Task', 'description': 'TransferCube maintains ~100% success even with noise_0.3, while Insertion degrades more quickly, suggesting task complexity modulates vision robustness.', 'evidence': 'Gaussian noise perturbation at sigma=0.1 and 0.3.', 'confidence': 'medium', 'category': 'vision'},
            ],
            'metrics': {'total_episodes': 1110, 'experimental_episodes': 960, 'baseline_episodes': 150, 'grid_conditions': 32, 'injection_experiments': 150, 'tasks': 2},
            'limitations': ['No SAE analysis (ACT uses a different architecture)', 'Only 2 simulation tasks (no real robot)', 'blur_5 condition crashed due to tensor shape mismatch'],
        }
    elif model == 'xvla':
        findings = {
            'model': 'xvla', 'model_name': 'X-VLA',
            'summary': 'X-VLA (1B, Florence-2 backbone, 24 TransformerBlocks, flow-matching) provides single-pathway analysis across LIBERO and SimplerEnv. 99.8% source trajectory override under cross-task injection. Language sensitivity depends on task structure, not model design.',
            'key_findings': [
                {'id': 'source_override', 'title': 'Near-Total Source Behavior Override (99.8%)', 'description': 'Cross-task injection steers robots toward source-task positions in 99.8% of episodes (3,150 episodes, 4 suites). Injected trajectories have 0.94 mean cosine similarity to source vs 0.31 to destination, exposing spatially bound motor programs tied to scene coordinates rather than abstract task representations.', 'evidence': '3,150 cross-task injection episodes across 4 LIBERO suites (layers 12, 20, 23).', 'confidence': 'high', 'category': 'injection'},
                {'id': 'language_task_dependent', 'title': 'Language Sensitivity Depends on Task Structure', 'description': 'When visual context uniquely specifies the task, language is ignored (libero_object: 60-100% regardless of prompt). When multiple goals share a scene, language becomes essential (libero_goal: 94% to 10% under wrong prompts). This demonstrates language sensitivity depends on task structure, not model design.', 'evidence': 'Counterfactual prompting across 4 LIBERO suites with 14+ prompt conditions.', 'confidence': 'high', 'category': 'language'},
                {'id': 'all_layers_critical', 'title': 'All 24 Layers Critical (Narrow Architecture)', 'description': "Zeroing any single one of X-VLA's 24 TransformerBlocks causes complete task failure (0% success, baseline 96.7%). Despite 1024-dim hidden space (same as Pi0.5), X-VLA shows 82.3% zero-effect rate under concept ablation, behaving more like 4096-dim OFT than like Pi0.5.", 'evidence': 'Per-layer grid ablation across 4 LIBERO suites + concept ablation (2,480 pairs).', 'confidence': 'high', 'category': 'architecture'},
                {'id': 'concept_resilience', 'title': 'Concept Representation Width-Independent Resilience', 'description': '82.3% zero-effect rate under concept ablation despite 1024-dim hidden space. X-VLA distributes concepts differently than Pi0.5 (same width), suggesting representation width alone does not predict sensitivity.', 'evidence': '2,480 concept ablation pairs, 82+ manipulation concepts across 17 categories.', 'confidence': 'high', 'category': 'causal'},
                {'id': 'simplerenv_transfer', 'title': 'Cross-Benchmark Transfer (SimplerEnv)', 'description': 'X-VLA injection and counterfactual findings transfer to SimplerEnv (WidowX, Google Robot), confirming architecture-level rather than benchmark-specific properties.', 'evidence': '~5K SimplerEnv episodes across 2 environments, 24 counterfactual conditions.', 'confidence': 'medium', 'category': 'injection'},
            ],
            'metrics': {'total_episodes': 50000, 'layers_analyzed': 24, 'sae_width': 8192, 'hidden_dim': 1024, 'concepts_identified': 82, 'architecture': 'single_pathway', 'benchmarks': ['LIBERO', 'SimplerEnv']},
            'limitations': ['SimplerEnv experiments limited to 2 environments', 'Mean-pooling improves SAE fidelity on X-VLA (unlike other architectures where per-token is better)'],
        }
    elif model == 'smolvla':
        findings = {
            'model': 'smolvla', 'model_name': 'SmolVLA',
            'summary': 'SmolVLA (450M, interleaved VLM + Expert dual pathway, 480/960-dim) enables pathway-specific analysis across LIBERO and MetaWorld. Expert pathways encode motor programs (2x greater displacement from expert injection), VLM pathways encode goal semantics.',
            'key_findings': [
                {'id': 'expert_motor_programs', 'title': 'Expert Pathway Encodes Motor Programs', 'description': 'Expert pathway injections override destination behavior in 15.8% of episodes vs 9.0% for VLM pathway (1.76x ratio, 732 pairs, 4 difficulty levels). This confirms expert pathway as the primary action computation pathway, while VLM encodes goal semantics.', 'evidence': '732 injection pairs across 4 MetaWorld difficulty levels.', 'confidence': 'high', 'category': 'injection'},
                {'id': 'narrow_expert', 'title': 'Narrow Expert Pathway (480-dim) Shows Severe Sensitivity', 'description': 'SmolVLA expert pathway (480-dim) shows 27.6% zero-effect rate and 6.3% destruction rate under concept ablation, with mean delta of -14.5pp. L00/L01 are most destructive. 10+ kill-switches identified (motion/push, motion/open, microwave all cause -100pp).', 'evidence': '1,696 concept ablation pairs across expert layers.', 'confidence': 'high', 'category': 'causal'},
                {'id': 'vlm_fragile_l0', 'title': 'VLM L0 is Severeally Fragile', 'description': 'VLM pathway L0 ablation causes -56.4pp mean delta with 64.3% destructive rate. Later VLM layers show mixed effects. Kill-switches: cat/object (-78.5pp), cat/motion (-73.5pp), object/orange_juice (-70pp).', 'evidence': '210 VLM concept ablation pairs.', 'confidence': 'high', 'category': 'causal'},
                {'id': 'cross_benchmark', 'title': 'Cross-Benchmark Validation (LIBERO + MetaWorld)', 'description': 'SmolVLA findings validated across both LIBERO (4 suites, 10 tasks each) and MetaWorld (50 tasks, 4 difficulty levels). Grid ablation shows 21 conditions with baseline ~73% success rate.', 'evidence': '42,000+ episodes across LIBERO and MetaWorld.', 'confidence': 'high', 'category': 'architecture'},
                {'id': 'subspace_separation', 'title': 'Expert and VLM Occupy Separable Activation Subspaces', 'description': 'Subspace injection confirms expert and VLM pathways occupy separable activation subspaces. Expert injection produces 2x greater behavioral displacement than VLM injection.', 'evidence': 'Cross-task injection comparing expert vs VLM pathway effects.', 'confidence': 'high', 'category': 'architecture'},
            ],
            'metrics': {'total_episodes': 42000, 'layers_analyzed': 64, 'vlm_layers': 32, 'expert_layers': 32, 'sae_width_vlm': 7680, 'sae_width_expert': 3840, 'hidden_dim_vlm': 960, 'hidden_dim_expert': 480, 'concepts_identified': 45, 'architecture': 'dual_pathway_interleaved', 'benchmarks': ['LIBERO', 'MetaWorld']},
            'limitations': ['MetaWorld cross-task injection only for MetaWorld (not LIBERO)', 'SmolVLA actual dims 960/480 (not 1536 as originally reported)'],
        }
    elif model == 'groot':
        findings = {
            'model': 'groot', 'model_name': 'GR00T N1.5',
            'summary': 'GR00T N1.5 (3B, triple-pathway: 16 DiT + 12 Eagle LM + 4 VL-SA, diffusion action generation) enables cross-pathway analysis. DiT layers encode motor programs, Eagle LM provides language grounding, VL-SA handles cross-modal attention. 164,700+ episodes analyzed.',
            'key_findings': [
                {'id': 'triple_pathway_specialization', 'title': 'Triple-Pathway Specialization', 'description': 'DiT layers (16) encode motor programs with highest displacement (avg max 0.232). Eagle LM (12 layers) shows 73% zero-effect ablation rate (most resilient). VL-SA (4 layers) shows 84% zero-effect rate. Pathway sensitivity ordering: DiT (56%/10% zero/destroy) > Eagle (73%/3%) > VL-SA (84%/1%).', 'evidence': '6,500 concept ablation pairs across 64 files, 123,612 displacement scenes.', 'confidence': 'high', 'category': 'architecture'},
                {'id': 'all_dit_critical', 'title': 'All 16 DiT Layers Critical', 'description': "Zeroing any single DiT layer causes complete task failure (0% success, baselines 70-100%). This mirrors X-VLA's narrow architecture finding. GR00T's DiT pathway is non-redundant despite 2048-dim hidden space.", 'evidence': 'Per-layer grid ablation across 3 LIBERO suites (Goal, Object, Long).', 'confidence': 'high', 'category': 'causal'},
                {'id': 'cross_task_injection', 'title': 'Cross-Task Injection (17 pairs x 3 suites)', 'description': 'Cross-task injection across DiT, Eagle, and VL-SA pathways with both cross-prompt and own-prompt conditions. EEF trajectory analysis available for displacement comparison.', 'evidence': '17 injection pairs per suite, 3 LIBERO suites, 8+ conditions per pair.', 'confidence': 'high', 'category': 'injection'},
                {'id': 'concept_ablation_profiles', 'title': 'Pathway-Specific Concept Sensitivity', 'description': 'GR00T concept ablation across 15 concept groups shows 59.2% overall zero-effect rate with 9.1% destruction. DiT layers are most sensitive (56% zero, 10% destroy), Eagle LM most resilient (73% zero, 3% destroy), VL-SA nearly inert (84% zero, 1% destroy).', 'evidence': '96 baked concept ablation entries across all pathways.', 'confidence': 'high', 'category': 'causal'},
                {'id': 'diffusion_unique', 'title': 'Diffusion Action Generation Unique Properties', 'description': 'GR00T uses diffusion-based action generation (unlike flow-matching or L1 regression in other models). Action horizon of 16 steps with inference every 16 steps produces distinct temporal dynamics in ablation response.', 'evidence': 'Temporal ablation and FTF analysis across 3 suites.', 'confidence': 'medium', 'category': 'architecture'},
            ],
            'metrics': {'total_episodes': 164700, 'layers_analyzed': 32, 'dit_layers': 16, 'eagle_layers': 12, 'vlsa_layers': 4, 'sae_width': 16384, 'hidden_dim': 2048, 'concepts_identified': 36, 'architecture': 'triple_pathway', 'benchmarks': ['LIBERO']},
            'limitations': ['Only LIBERO benchmark (3 suites: Goal, Object, Long)', 'DiT grid ablation shows uniform 0% (all layers critical, no gradient)'],
        }
    else:
        return jsonify({'status': 404, 'error': {'code': 'MODEL_NOT_FOUND', 'message': f'No findings available for model: {model}'}}), 404

    return jsonify({'status': 200, 'data': findings})


# Experiment Stats

@experiments_bp.route('/api/vla/experiment_stats', methods=['GET'])
def get_experiment_stats():
    # Get aggregated experiment statistics for analytics
    model = request.args.get('model', 'pi05')
    video_index = load_video_index(model)
    if video_index is None:
        return jsonify({'status': 404, 'error': {'code': 'INDEX_NOT_FOUND', 'message': f'No video index found for model: {model}'}}), 404

    videos = video_index.get('videos', [])
    by_experiment_type = {}
    by_suite = {}
    by_experiment_and_suite = {}

    for video in videos:
        exp_type = video.get('experiment_type', 'unknown')
        suite = video.get('suite', 'unknown')
        success = parse_success_from_path(video.get('path', ''), video)

        for bucket_key, bucket_dict in [(exp_type, by_experiment_type), (suite, by_suite)]:
            if bucket_key not in bucket_dict:
                bucket_dict[bucket_key] = {'total': 0, 'success': 0, 'failure': 0, 'unknown': 0}
            bucket_dict[bucket_key]['total'] += 1
            if success is True:
                bucket_dict[bucket_key]['success'] += 1
            elif success is False:
                bucket_dict[bucket_key]['failure'] += 1
            else:
                bucket_dict[bucket_key]['unknown'] += 1

        if exp_type not in by_experiment_and_suite:
            by_experiment_and_suite[exp_type] = {}
        if suite not in by_experiment_and_suite[exp_type]:
            by_experiment_and_suite[exp_type][suite] = {'total': 0, 'success': 0, 'failure': 0, 'unknown': 0}
        nested = by_experiment_and_suite[exp_type][suite]
        nested['total'] += 1
        if success is True:
            nested['success'] += 1
        elif success is False:
            nested['failure'] += 1
        else:
            nested['unknown'] += 1

    for bucket_dict in [by_experiment_type, by_suite]:
        for data in bucket_dict.values():
            known = data['success'] + data['failure']
            data['rate'] = round(data['success'] / known, 4) if known > 0 else None
    for exp_type_dict in by_experiment_and_suite.values():
        for data in exp_type_dict.values():
            known = data['success'] + data['failure']
            data['rate'] = round(data['success'] / known, 4) if known > 0 else None

    ablation_results = load_ablation_results(model)
    ablation_summary = {'total_experiments': len(ablation_results), 'by_concept': {}, 'by_ablation_type': {'temporal': [], 'fractional': []}}

    for result in ablation_results:
        concept = result.get('concept', 'unknown')
        filename = result.get('filename', '')
        results_data = result.get('results', {})
        ablation_type = 'temporal' if 'temporal' in filename else 'fractional'
        baseline_rate = results_data.get('baseline', {}).get('success_rate')
        intervention_rates = {phase: phase_data.get('success_rate') for phase, phase_data in results_data.items() if phase != 'baseline' and isinstance(phase_data, dict)}
        summary_entry = {'concept': concept, 'layer': result.get('layer'), 'suite': result.get('suite'), 'baseline_success_rate': baseline_rate, 'intervention_rates': intervention_rates, 'total_trials': results_data.get('baseline', {}).get('total', 0), 'filename': filename}
        ablation_summary['by_concept'].setdefault(concept, []).append(summary_entry)
        ablation_summary['by_ablation_type'][ablation_type].append(summary_entry)

    concept_averages = {}
    for concept, exps in ablation_summary['by_concept'].items():
        baseline_rates = [e['baseline_success_rate'] for e in exps if e['baseline_success_rate'] is not None]
        concept_averages[concept] = {'experiment_count': len(exps), 'avg_baseline_rate': round(sum(baseline_rates) / len(baseline_rates), 4) if baseline_rates else None}
    ablation_summary['concept_averages'] = concept_averages

    return jsonify({'model': model, 'total_videos': len(videos), 'by_experiment_type': by_experiment_type, 'by_suite': by_suite, 'by_experiment_and_suite': by_experiment_and_suite, 'ablation_summary': ablation_summary, 'experiment_types': list(video_index.get('experiment_types', {}).keys()), 'suites': list(video_index.get('suites', {}).keys())})


# Experiment Results

@experiments_bp.route('/api/vla/experiment_results', methods=['GET'])
def get_experiment_results():
    # Get experiment results for a video or experiment directory
    video_path = request.args.get('video_path')
    experiment_dir = request.args.get('experiment_dir')

    if not video_path and not experiment_dir:
        return jsonify({'success': False, 'error': 'Either video_path or experiment_dir parameter required'}), 400

    if video_path:
        results_data = find_results_json_for_video(video_path)
        extracted = extract_video_results(video_path, results_data)
        return jsonify({'success': True, 'video_path': video_path, 'results': extracted, 'results_file': results_data.get('results_path') if results_data else None})

    exp_dir = Path(experiment_dir)
    if not exp_dir.is_absolute():
        for base in [VLA_VIDEO_DIR / "pi05", PI05_ROLLOUTS_DIR]:
            candidate = base / experiment_dir
            if candidate.exists():
                exp_dir = candidate
                break

    if not exp_dir.exists():
        return jsonify({'success': False, 'error': f'Experiment directory not found: {experiment_dir}'}), 404

    results_json = exp_dir / "results.json"
    if not results_json.exists():
        return jsonify({'success': False, 'error': f'No results.json found in {experiment_dir}'}), 404

    try:
        with open(results_json) as f:
            data = json.load(f)

        summary = {'suite': data.get('suite'), 'seed': data.get('seed'), 'max_steps': data.get('max_steps'), 'task_prompts': data.get('task_prompts', {}), 'n_pairs': len(data.get('results', {})), 'pairs': {}}

        for pair_name, pair_data in data.get('results', {}).items():
            pair_summary = {
                'task_a': pair_data.get('task_a'), 'task_b': pair_data.get('task_b'),
                'prompt_a': pair_data.get('prompt_a'), 'prompt_b': pair_data.get('prompt_b'),
                'baseline_0_success': pair_data.get('baseline_task_0', {}).get('success'),
                'baseline_0_steps': pair_data.get('baseline_task_0', {}).get('steps'),
                'baseline_1_success': pair_data.get('baseline_task_1', {}).get('success'),
                'baseline_1_steps': pair_data.get('baseline_task_1', {}).get('steps'),
            }
            for inject_name in ['inject_0_into_1', 'inject_1_into_0']:
                if inject_name in pair_data:
                    inject_data = pair_data[inject_name]
                    successes = sum(1 for exp in inject_data.values() if isinstance(exp, dict) and exp.get('success') is True)
                    total = sum(1 for exp in inject_data.values() if isinstance(exp, dict) and 'success' in exp)
                    pair_summary[f'{inject_name}_success_rate'] = f'{successes}/{total}' if total > 0 else 'N/A'
            summary['pairs'][pair_name] = pair_summary

        return jsonify({'success': True, 'experiment_dir': str(exp_dir), 'results_file': str(results_json), 'summary': summary})
    except (json.JSONDecodeError, IOError) as e:
        return jsonify({'success': False, 'error': f'Failed to parse results.json: {str(e)}'}), 500


# Experiment Types

@experiments_bp.route('/api/vla/experiment_types', methods=['GET'])
def get_experiment_types():
    # Get available experiment types with counts and suites
    model = request.args.get('model', 'pi05')
    experiment_types = {}

    if model == 'pi05':
        video_index = load_video_index(model)
        if video_index:
            type_counts = {}
            type_suites = {}
            for video in video_index.get('videos', []):
                exp_type = video.get('experiment_type', 'unknown')
                suite = video.get('suite', 'unknown')
                type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
                if suite:
                    type_suites.setdefault(exp_type, set()).add(suite)
            for exp_type in type_counts:
                experiment_types[exp_type] = {'count': type_counts[exp_type], 'suites': sorted(type_suites.get(exp_type, set()))}
        else:
            base_dirs = {
                'counterfactual': PI05_ROLLOUTS_DIR / "counterfactual",
                'cross_task': PI05_ROLLOUTS_DIR / "cross_task_goal",
                'vision_perturbation': PI05_ROLLOUTS_DIR / "vision_perturbation",
            }
            for exp_type, base_dir in base_dirs.items():
                if not base_dir.exists():
                    continue
                try:
                    video_count = sum(1 for _ in base_dir.rglob("*.mp4"))
                    suites = set()
                    for subdir in base_dir.iterdir():
                        if subdir.is_dir():
                            name = subdir.name
                            if name.startswith('libero_'):
                                suites.add(name.replace('libero_', ''))
                            elif name in ('goal', 'spatial', 'object'):
                                suites.add(name)
                    experiment_types[exp_type] = {'count': video_count, 'suites': sorted(suites)}
                except Exception:
                    pass

        descriptions = {'counterfactual': 'Prompt manipulation experiments testing concept understanding', 'cross_task': 'Cross-task activation injection experiments', 'vision_perturbation': 'Vision perturbation robustness experiments'}
        for exp_type in experiment_types:
            if exp_type in descriptions:
                experiment_types[exp_type]['description'] = descriptions[exp_type]

    elif model in ('openvla', 'openvla_oft'):
        n_ablation_vids = 11892
        if OFT_ABLATION_VIDEO_DIR.exists():
            try:
                n_ablation_vids = sum(1 for _ in OFT_ABLATION_VIDEO_DIR.rglob("*.mp4"))
            except Exception:
                pass
        libero_suites = ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10']
        experiment_types = {
            'concept_identification': {'count': 128, 'suites': libero_suites, 'description': "Contrastive concept identification (Cohen's d x freq) across 32 layers, 15 unique concept categories"},
            'concept_ablation': {'count': n_ablation_vids, 'suites': libero_suites, 'description': 'Concept ablation rollout videos (18 ablation JSONs, 1,810 task-concept pairs, 91.6% zero effect)'},
            'counterfactual': {'count': 1110, 'suites': libero_suites, 'description': 'Counterfactual prompting experiments'},
            'cross_task_injection': {'count': 168, 'suites': libero_suites, 'description': 'Cross-task activation injection pairs (only libero_goal drops -40pp, others ~0pp)'},
            'same_scene_injection': {'count': 400, 'suites': libero_suites, 'description': 'Same-scene injection HURTS all suites (-17pp to -57pp)'},
            'sae_validation': {'count': 120, 'suites': libero_suites, 'description': 'SAE hook validation: 119/120 (99.2%) success rate'},
        }

    elif model == 'xvla':
        libero_suites = ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10']
        _scan_dirs_to_experiment_types(experiment_types, [
            (XVLA_CONCEPT_ID_DIR, "*.json", False, 'concept_identification', libero_suites, 'Contrastive concept identification across 24 layers'),
            (XVLA_ABLATION_DIR, "*.json", True, 'concept_ablation', libero_suites, 'Concept ablation (single pathway, 24 layers)'),
            (XVLA_STEERING_DIR, "*.json", True, 'concept_steering', libero_suites, 'Concept steering experiments'),
        ])

    elif model == 'smolvla':
        libero_suites = ['libero_goal', 'libero_object', 'libero_spatial', 'libero_10']
        _scan_dirs_to_experiment_types(experiment_types, [
            (SMOLVLA_CONCEPT_ID_DIR, "*.json", False, 'concept_identification', libero_suites, 'Contrastive concept identification across 64 layers (32 VLM + 32 expert)'),
            (SMOLVLA_ABLATION_DIR, "*.json", True, 'concept_ablation', libero_suites, 'Concept ablation (dual pathway, interleaved)'),
        ])

    elif model == 'groot':
        groot_suites = ['libero_object', 'libero_goal', 'libero_long']
        _scan_dirs_to_experiment_types(experiment_types, [
            (GROOT_ABLATION_DIR, "*.json", True, 'concept_ablation', groot_suites, 'SAE feature ablation across DiT (16L) + Eagle (12L) + VL-SA (4L)'),
            (GROOT_STEERING_DIR, "*.json", True, 'concept_steering', groot_suites, 'SAE feature steering experiments'),
            (GROOT_PROBING_DIR, "*.json", True, 'probing', groot_suites, 'Linear probing across triple-pathway layers'),
        ])

    elif model in ('act', 'act_aloha'):
        experiment_types = {
            'grid_ablation': {'count': 76, 'suites': [], 'description': 'Grid ablation (4x4 region masking) for AlohaInsertion and AlohaTransferCube'},
            'injection': {'count': 3, 'suites': [], 'description': 'Cross-task and same-task activation injection'},
        }

    return jsonify({'experiment_types': experiment_types, 'model': model, 'total_experiments': sum(et.get('count', 0) for et in experiment_types.values())})


# Layer Connections

@experiments_bp.route('/api/vla/layer_connections', methods=['GET'])
def get_layer_connections():
    # Get data-driven layer connection information for the wire visualization
    model = request.args.get('model', 'openvla')
    suite = request.args.get('suite', 'libero_goal')
    pathway = request.args.get('pathway', 'expert')
    static_dir = _API_DATA_DIR / 'layer_connections'

    try:
        if model in ('openvla', 'openvla_oft'):
            static_file = static_dir / f'openvla_{suite}.json'
            result = json.load(open(static_file)) if static_file.exists() else _load_layer_connections_openvla(suite)
        elif model == 'pi05':
            if pathway not in ('expert', 'paligemma'):
                pathway = 'expert'
            result = _load_layer_connections_pi05(pathway=pathway)
        elif model in ('xvla', 'smolvla', 'groot', 'act'):
            static_file = static_dir / f'{model}_{suite}.json'
            result = json.load(open(static_file)) if static_file.exists() else _build_layer_connections_from_config(model, suite)
        else:
            return jsonify({'status': 400, 'error': {'code': 'INVALID_MODEL', 'message': f'Unknown model: {model}'}}), 400

        return jsonify({'status': 200, 'data': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 500, 'error': {'code': 'INTERNAL_ERROR', 'message': str(e)}}), 500


@experiments_bp.route('/api/vla/oft_ablation_videos', methods=['GET'])
def get_oft_ablation_videos():
    # Get available OFT concept ablation videos
    suite = request.args.get('suite', 'libero_goal')
    concept_filter = request.args.get('concept', None)
    layer_filter = request.args.get('layer', None)
    limit = int(request.args.get('limit', '50'))

    suite_dir = OFT_ABLATION_VIDEO_DIR / suite
    if not suite_dir.exists():
        available = [d.name for d in OFT_ABLATION_VIDEO_DIR.iterdir() if d.is_dir()] if OFT_ABLATION_VIDEO_DIR.exists() else []
        return jsonify({'status': 404, 'error': {'code': 'SUITE_NOT_FOUND', 'message': f'No ablation videos for suite {suite}', 'available_suites': available}}), 404

    videos = []
    for video_path in sorted(suite_dir.glob("*.mp4")):
        parsed = _parse_ablation_video_filename(video_path.stem)
        full_concept = f"{parsed['concept_type']}/{parsed['concept_name']}" if parsed['concept_type'] and parsed['concept_name'] else video_path.stem

        if concept_filter and concept_filter not in full_concept and concept_filter != parsed['concept_name']:
            continue
        if layer_filter is not None:
            try:
                if parsed['layer_num'] != int(layer_filter):
                    continue
            except ValueError:
                pass

        videos.append({
            'path': f'/api/vla/video/oft_ablation/{suite}/{video_path.name}',
            'filename': video_path.name, 'suite': suite, 'layer': parsed['layer_num'],
            'concept_type': parsed['concept_type'], 'concept': parsed['concept_name'],
            'full_concept': full_concept, 'task': parsed['task_num'], 'episode': parsed['ep_num'],
        })
        if len(videos) >= limit:
            break

    all_concepts = set()
    all_layers = set()
    for v in sorted(suite_dir.glob("*.mp4")):
        parsed = _parse_ablation_video_filename(v.stem)
        if parsed['layer_num'] is not None:
            all_layers.add(parsed['layer_num'])
        if parsed['concept_type'] and parsed['concept_name']:
            all_concepts.add(f"{parsed['concept_type']}/{parsed['concept_name']}")

    return jsonify({'status': 200, 'data': {
        'suite': suite, 'videos': videos, 'total': len(videos),
        'available_suites': [d.name for d in OFT_ABLATION_VIDEO_DIR.iterdir() if d.is_dir()] if OFT_ABLATION_VIDEO_DIR.exists() else [],
        'available_concepts': sorted(all_concepts), 'available_layers': sorted(all_layers),
    }})


# ACT Results

@experiments_bp.route('/api/vla/act_results', methods=['GET'])
def get_act_results():
    # Get ACT-ALOHA experiment results (grid ablation + injection)
    experiment = request.args.get('experiment', 'all')
    task_filter = request.args.get('task', None)

    all_results_file = ACT_RESULTS_DIR / "all_results.json"
    if not all_results_file.exists():
        return jsonify({'status': 404, 'error': {'code': 'DATA_NOT_FOUND', 'message': 'No ACT results data found'}}), 404

    with open(all_results_file) as f:
        all_data = json.load(f)

    result = {'model': 'act', 'experiment': experiment}

    if experiment in ('grid_ablation', 'all'):
        grid_results = {}
        if ACT_GRID_ABLATION_DIR.exists():
            for jf in sorted(ACT_GRID_ABLATION_DIR.glob("*.json")):
                name = jf.stem
                if task_filter and task_filter not in name:
                    continue
                with open(jf) as f:
                    grid_results[name] = json.load(f)

        tasks = {}
        for name, data in grid_results.items():
            env = data.get('env_name', name.split('_')[0])
            if env not in tasks:
                tasks[env] = {'baseline': None, 'grid': {}, 'noise': {}}
            if 'baseline' in name:
                tasks[env]['baseline'] = {'success_rate': data.get('success_rate', 0), 'mean_reward': data.get('mean_reward', 0), 'n_episodes': data.get('n_episodes', 0)}
            elif 'grid_' in name:
                parts = name.split('_')
                for i, p in enumerate(parts):
                    if p == 'grid' and i + 2 < len(parts):
                        row, col = parts[i+1], parts[i+2]
                        tasks[env]['grid'][f"{row}_{col}"] = {'success_rate': data.get('success_rate', 0), 'mean_reward': data.get('mean_reward', 0), 'n_episodes': data.get('n_episodes', 0), 'row': int(row), 'col': int(col), 'bbox': data.get('perturbation_kwargs', {}).get('bbox', [])}
                        break
            elif 'noise_' in name:
                noise_level = name.split('noise_')[-1]
                tasks[env]['noise'][noise_level] = {'success_rate': data.get('success_rate', 0), 'mean_reward': data.get('mean_reward', 0), 'n_episodes': data.get('n_episodes', 0)}
        result['grid_ablation'] = tasks

    if experiment in ('injection', 'all'):
        result['injection'] = all_data.get('injection', {})
        result['baselines'] = {'transfer_cube': all_data.get('baseline_transfer_cube', {}), 'insertion': all_data.get('baseline_insertion', {})}

    return jsonify({'status': 200, 'data': result})


# Model Experiment Results (aggregated JSON)

@experiments_bp.route('/api/experiments/<model>', methods=['GET'])
def get_model_experiment_results(model: str):
    # Get aggregated experiment results for a model
    if model not in VALID_MODELS:
        return jsonify({'status': 400, 'error': {'code': 'INVALID_MODEL', 'message': f'Model must be one of: {VALID_MODELS}'}}), 400

    file_model = _resolve_file_model(model)
    data = _load_experiment_results(file_model)
    if data is None:
        return jsonify({'status': 404, 'error': {'code': 'NO_DATA', 'message': f'No experiment results found for {model}. Run scripts/aggregate_experiment_results.py first.'}}), 404

    section = request.args.get('section')
    suite = request.args.get('suite')

    if section:
        section_data = data.get(section)
        if section_data is None:
            skip_keys = ('model', 'model_name', 'description', 'architecture', 'params', 'timestamp', 'environments')
            available = [k for k, v in data.items() if isinstance(v, dict) and k not in skip_keys]
            return jsonify({'status': 404, 'error': {'code': 'SECTION_NOT_FOUND', 'message': f'Section "{section}" not found for {model}. Available: {available}'}}), 404
        if suite and isinstance(section_data, dict) and suite in section_data:
            section_data = section_data[suite]
        return jsonify({'status': 200, 'data': {'model': model, 'model_name': data.get('model_name', model), 'section': section, 'suite': suite, 'results': section_data}})

    KNOWN_STATS = _load_known_stats()
    known = KNOWN_STATS.get(file_model, KNOWN_STATS.get(model, {}))

    skip_keys = ('model', 'model_name', 'description', 'architecture', 'params', 'timestamp', 'environments')
    sections_summary = {}
    for key, value in data.items():
        if not isinstance(value, dict) or key in skip_keys:
            continue
        known_sec = known.get(key, {})
        sec_episodes = known_sec.get('episodes', 0)
        sec_sr = known_sec.get('success_rate', None)
        if sec_episodes == 0:
            for sk, sv in value.items():
                if sk == 'summary':
                    continue
                if isinstance(sv, dict):
                    sec_episodes += sv.get('n_episodes', sv.get('episodes', sv.get('num_episodes', sv.get('n_total_episodes', 0))))
            if sec_episodes == 0:
                direct_ep = value.get('episodes', value.get('n_episodes', value.get('total_episodes', 0)))
                if isinstance(direct_ep, (int, float)):
                    sec_episodes = int(direct_ep)
        sections_summary[key] = {'n_entries': len(value), 'episodes': sec_episodes, 'success_rate': sec_sr, 'keys': list(value.keys()) if len(value) <= 20 else list(value.keys())[:20] + ['...']}

    for key, sec_data in known.items():
        if key in ('total', 'label', 'baseline_success_rate'):
            continue
        if isinstance(sec_data, dict) and sec_data.get('episodes', 0) > 0:
            if key not in sections_summary:
                sections_summary[key] = {'n_entries': 0, 'episodes': sec_data.get('episodes', 0), 'success_rate': sec_data.get('success_rate'), 'keys': []}
            elif sections_summary[key].get('episodes', 0) == 0:
                sections_summary[key]['episodes'] = sec_data.get('episodes', 0)
                if sec_data.get('success_rate') is not None:
                    sections_summary[key]['success_rate'] = sec_data.get('success_rate')

    return jsonify({'status': 200, 'data': {
        'model': model, 'model_name': data.get('model_name', model), 'description': data.get('description', ''),
        'architecture': data.get('architecture', ''), 'params': data.get('params', ''),
        'environments': data.get('environments', []), 'timestamp': data.get('timestamp', ''),
        'sections': sections_summary, 'total_episodes': known.get('total', 0),
        'total_episodes_label': known.get('label', 'N/A'), 'overall_success_rate': known.get('overall_success_rate'),
        'baseline_success_rate': known.get('baseline_success_rate'), 'full_data': data,
    }})


# List Available Experiment Models

@experiments_bp.route('/api/experiments', methods=['GET'])
def list_available_experiment_models():
    # List all models with available experiment data
    available = []
    skip_keys = ('model', 'model_name', 'description', 'architecture', 'params', 'timestamp', 'environments')
    for model in ['smolvla', 'xvla', 'groot']:
        results_path = _API_DATA_DIR / f"experiment_results_{model}.json"
        if not results_path.exists():
            continue
        data = _load_experiment_results(model)
        if not data:
            continue
        sections = [k for k, v in data.items() if isinstance(v, dict) and k not in skip_keys]
        available.append({
            'model': model, 'model_name': data.get('model_name', model),
            'description': data.get('description', ''), 'architecture': data.get('architecture', ''),
            'params': data.get('params', ''), 'environments': data.get('environments', []),
            'sections': sections, 'file_size_kb': round(results_path.stat().st_size / 1024, 1),
        })
    return jsonify({'status': 200, 'data': {'models': available, 'n_models': len(available)}})


# Temporal Ablation

@experiments_bp.route('/api/vla/temporal_ablation/<model>', methods=['GET'])
def get_temporal_ablation_data(model: str):
    # Get temporal ablation data for a model. Supported: groot, smolvla
    supported_models = ['groot', 'smolvla']
    if model not in supported_models:
        return jsonify({'status': 200, 'data': {'model': model, 'available': False, 'message': f'Temporal ablation data not yet available for {model}. Available for: GR00T N1.5, SmolVLA.', 'supported_models': supported_models}})
    if model == 'groot':
        return _get_groot_temporal_ablation()
    return _get_smolvla_temporal_ablation()


@experiments_bp.route('/api/experiments/<model>/summary', methods=['GET'])
def get_model_experiment_summary(model: str):
    # Get a compact summary of experiment results for dashboard views
    if model not in VALID_MODELS:
        return jsonify({'status': 400, 'error': {'code': 'INVALID_MODEL', 'message': f'Model must be one of: {VALID_MODELS}'}}), 400

    file_model = _resolve_file_model(model)
    data = _load_experiment_results(file_model)
    if data is None:
        return jsonify({'status': 404, 'error': {'code': 'NO_DATA', 'message': f'No experiment results found for {model}.'}}), 404

    summary = {'model': model, 'model_name': data.get('model_name', model), 'architecture': data.get('architecture', '')}

    baselines = data.get('baselines', {})
    baseline_summary = {}
    for suite_key, suite_data in baselines.items():
        if isinstance(suite_data, dict):
            baseline_summary[suite_key] = {'overall_success_rate': suite_data.get('overall_success_rate', 0), 'n_tasks': len(suite_data.get('tasks', {})), 'environment': suite_data.get('environment', '')}
    summary['baselines'] = baseline_summary

    grid = data.get('grid_ablation', {})
    grid_summary = {}
    for suite_key, suite_data in grid.items():
        if isinstance(suite_data, dict):
            per_condition = suite_data.get('per_condition', suite_data.get('per_layer', {}))
            condition_srs = {cond: cdata.get('overall_success_rate', 0) for cond, cdata in per_condition.items() if isinstance(cdata, dict)}
            grid_summary[suite_key] = {
                'baseline_overall': suite_data.get('baseline_overall', sum(v for v in suite_data.get('baseline', {}).values() if isinstance(v, (int, float))) / max(len(suite_data.get('baseline', {})), 1) if suite_data.get('baseline') else 0),
                'n_conditions': len(condition_srs), 'per_condition': condition_srs,
            }
    summary['grid_ablation'] = grid_summary

    cf = data.get('counterfactual', {})
    cf_summary = {}
    for suite_key, suite_data in cf.items():
        if isinstance(suite_data, dict):
            cats = suite_data.get('categories', {})
            cf_summary[suite_key] = {'n_total_episodes': suite_data.get('n_total_episodes', 0), 'categories': {cat: cdata.get('success_rate', 0) if isinstance(cdata, dict) else 0 for cat, cdata in cats.items()}}
    summary['counterfactual'] = cf_summary

    vp = data.get('vision_perturbation', {})
    vp_summary = {}
    for suite_key, suite_data in vp.items():
        if isinstance(suite_data, dict):
            per_pert = suite_data.get('per_perturbation', {})
            baseline_sr = per_pert.get('baseline', {}).get('overall_success_rate', per_pert.get('baseline', {}).get('success_rate', 0))
            worst_pert = None
            worst_sr = 1.0
            for pert, pdata in per_pert.items():
                if pert == 'baseline':
                    continue
                sr = pdata.get('overall_success_rate', pdata.get('success_rate', 0))
                if sr < worst_sr:
                    worst_sr = sr
                    worst_pert = pert
            vp_summary[suite_key] = {'baseline_success_rate': baseline_sr, 'worst_perturbation': worst_pert, 'worst_success_rate': worst_sr, 'n_perturbations': suite_data.get('n_perturbations', len(per_pert))}
    summary['vision_perturbation'] = vp_summary

    disp = data.get('displacement', {})
    if disp:
        disp_summary = {}
        for key, ddata in disp.items():
            if isinstance(ddata, dict):
                per_cond = ddata.get('per_condition', ddata.get('per_group', {}))
                override_rates = {cond: cdata.get('override_rate', cdata.get('source_behavior', 0) / max(cdata.get('total', 1), 1)) for cond, cdata in per_cond.items() if isinstance(cdata, dict)}
                disp_summary[key] = override_rates
        summary['displacement'] = disp_summary

    return jsonify({'status': 200, 'data': summary})
