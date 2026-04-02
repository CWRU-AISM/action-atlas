# Experiments

Mechanistic interpretability experiments for VLA models.
Supports X-VLA, SmolVLA, GR00T N1.5, Pi0.5, and OpenVLA-OFT.

## Supported Models

| Model | Architecture | Environments |
|-------|-------------|-------------|
| `xvla` | 24 transformer blocks (1024D) | LIBERO, SimplerEnv |
| `smolvla` | Expert + VLM layers (32 pairs) | LIBERO, MetaWorld |
| `groot` | Eagle LM + VL-SA + DiT | LIBERO |
| `pi05` | PaliGemma + Gemma Expert | LIBERO |
| `oft` | LLaMA-2 32 layers (4096D) | LIBERO |

## Scripts

Run any script with `--help` for options.

| Script | Purpose |
|--------|---------|
| `baseline.py` | Rollouts with optional activation collection |
| `grid_ablation.py` | Layer-by-layer zeroing |
| `vision_perturbation.py` | Visual robustness (24 perturbation types) |
| `counterfactual.py` | Language prompt variations |
| `cross_task_injection.py` | Activation swapping between tasks |
| `train_sae.py` | TopK sparse autoencoder training |
| `concept_id.py` | Concept identification using Cohen's d |
| `concept_ablation.py` | Remove concept features, measure impact |
| `concept_steering.py` | Amplify/suppress concept features |
| `launch_parallel.py` | Multi-GPU launcher |

## Workflow

```bash
# 1. Collect activations
python experiments/baseline.py --model xvla --suite libero_object \
    --collect-activations --n-episodes 3

# 2. Find critical layers
python experiments/grid_ablation.py --model xvla --suite libero_object

# 3. Train SAEs
python experiments/train_sae.py \
    --activations-dir outputs/xvla_experiments/baseline_libero_object/activations

# 4. Identify concepts
python experiments/concept_id.py \
    --sae-dir outputs/xvla_saes/libero_object \
    --activations-dir outputs/xvla_experiments/baseline_libero_object/activations \
    --suite libero_object

# 5. Verify causality
python experiments/concept_ablation.py --model xvla --suite libero_object \
    --sae-dir outputs/xvla_saes/libero_object \
    --concept-results results/concept_id/libero_object/all_layers.json \
    --layer transformer_L12
```

## SimplerEnv Experiments (X-VLA cross-embodiment)

Scripts in `experiments/simplerenv/` run X-VLA on SimplerEnv (WidowX and
Google Robot). These require the `simpler_env` conda environment (Python 3.10,
numpy<2). See `setup/SETUP.md` for installation.

```bash
conda activate simpler_env
export MUJOCO_GL=egl

python experiments/simplerenv/xvla_simplerenv_eval.py --model widowx --all-tasks --n_episodes 20
python experiments/simplerenv/xvla_simplerenv_grid_ablation.py --model widowx --n-episodes 3
python experiments/simplerenv/xvla_simplerenv_counterfactual.py --model widowx --all-tasks --n_episodes 3
python experiments/simplerenv/xvla_simplerenv_vision_perturbation.py --model widowx --all-tasks --n_episodes 3
python experiments/simplerenv/xvla_simplerenv_cross_task_injection.py --model widowx --all-pairs
```

| Script | Purpose |
|--------|---------|
| `xvla_simplerenv_eval.py` | Baseline eval with activation collection |
| `xvla_simplerenv_grid_ablation.py` | Layer-by-layer zeroing |
| `xvla_simplerenv_counterfactual.py` | Prompt perturbations |
| `xvla_simplerenv_cross_task_injection.py` | Cross-task activation injection |
| `xvla_simplerenv_vision_perturbation.py` | Image perturbation robustness |
| `xvla_simplerenv_temporal_injection.py` | Temporal activation injection |
| `xvla_simplerenv_concept_ablation.py` | SAE concept ablation |
| `xvla_simplerenv_concept_steering.py` | SAE concept steering |
| `xvla_simplerenv_reconstruction_eval.py` | SAE reconstruction fidelity |

## MetaWorld Experiments (SmolVLA)

Scripts in `experiments/metaworld/` run SmolVLA on MetaWorld MT50 tasks
(50 tasks across easy/medium/hard/very_hard). Uses the `actionatlas`
conda environment (Python 3.12). Checkpoint: `jadechoghari/smolvla_metaworld`.

```bash
conda activate actionatlas
export MUJOCO_GL=egl

python experiments/metaworld/run_smolvla_metaworld.py --difficulty easy --n-episodes 10
python experiments/metaworld/smolvla_metaworld_grid_ablation.py --difficulty easy --n-episodes 3
python experiments/metaworld/smolvla_metaworld_counterfactual.py --difficulty easy --output-dir outputs/mw_cf
python experiments/metaworld/smolvla_metaworld_vision_perturbation.py --difficulty easy --n-episodes 3
python experiments/metaworld/smolvla_metaworld_cross_task_injection.py --difficulty easy
```

| Script | Purpose |
|--------|---------|
| `run_smolvla_metaworld.py` | Baseline eval with activation collection |
| `smolvla_metaworld_grid_ablation.py` | Layer-by-layer MLP zeroing |
| `smolvla_metaworld_counterfactual.py` | Prompt perturbations (5 categories) |
| `smolvla_metaworld_enhanced_counterfactual.py` | Compositional/paraphrase/specificity prompts |
| `smolvla_metaworld_cross_task_injection.py` | Cross-task activation injection |
| `smolvla_metaworld_vision_perturbation.py` | Image perturbation robustness (23 perturbations) |
| `smolvla_metaworld_concept_id.py` | SAE concept identification (Cohen's d) |
| `validate_smolvla_sae_metaworld.py` | SAE reconstruction fidelity |

## Shared Modules

| Module | Purpose |
|--------|---------|
| `model_adapters.py` | Uniform model interface (loading, layers, episodes) |
| `hooks.py` | Forward hooks for ablation, capture, injection |
| `sae_hooks.py` | Per-token SAE ablation and steering hooks |
| `utils.py` | Scene state, video I/O, perturbations |
| `groot_common.py` | GR00T model loading and episode runner |
| `libero_utils.py` | LIBERO environment setup |
| `concept_identification.py` | Concept-to-task mappings |
