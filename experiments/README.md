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
