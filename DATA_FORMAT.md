# Action Atlas Data Formats

Standard output formats for all experiments. Follow these formats to ensure
compatibility with the Action Atlas visualization and to contribute results.

## Experiment Results

### Baseline / Grid Ablation / Vision Perturbation / Counterfactual

All produce per-condition JSON files:

```json
{
  "0": {
    "task_description": "pick up the alphabet soup and place it in the basket",
    "success_rate": 0.67,
    "successes": 2,
    "n_episodes": 3
  },
  "1": { ... }
}
```

Grid ablation adds `delta_from_baseline` per task.

### Cross-Task Injection

Per-pair results:

```json
{
  "source_task": 0,
  "target_task": 1,
  "source_desc": "task A description",
  "target_desc": "task B description",
  "success": false,
  "steps": 280,
  "injection_stats": {
    "transformer_L0": {"injections": 280, "shape_mismatches": 0}
  },
  "scene_summary": {
    "n_steps": 280,
    "object_displacements": {
      "object_name": {"distance": 0.05, "init_pos": [...], "final_pos": [...]}
    }
  }
}
```

### SAE Training Output

Per-layer checkpoint (`sae_best.pt`):

```python
{
    "sae_state_dict": {...},       # TopKSAE model weights
    "activation_mean": tensor,     # [input_dim] normalization mean
    "activation_std": tensor,      # [input_dim] normalization std
    "config": {
        "input_dim": 1024,
        "hidden_dim": 8192,
        "expansion": 8.0,
        "k": 64,
        "n_samples": 500000,
    },
    "history": [{"epoch": 0, "loss": ..., "explained_var": ..., "l0": ..., "dead_features_frac": ...}],
    "best_loss": 0.001,
}
```

### Concept Identification

Per-layer JSON:

```json
{
  "motion": {
    "put": {
      "tasks": [3, 4, 5],
      "n_in_samples": 15000,
      "n_out_samples": 35000,
      "top_features": [
        {"rank": 0, "feature_idx": 4231, "score": 2.31, "cohens_d": 3.1, "frequency": 0.75}
      ]
    }
  },
  "object": { ... },
  "spatial": { ... }
}
```

### Feature Descriptions

Per-layer JSON:

```json
{
  "model": "xvla",
  "pathway": "transformer",
  "layer": 12,
  "suite": "libero_goal",
  "n_features": 150,
  "llm": "gemini",
  "descriptions": {
    "4231": "Encodes rightward arm movements during object placement",
    "1087": "Activates when gripper approaches bowl-shaped objects"
  }
}
```

## Activations

Per-layer `.pt` files containing `torch.Tensor` in bfloat16:

```
activations/
  task{N}/
    ep{M}/
      transformer_L00.pt   # shape: [n_steps, seq_len, hidden_dim]
      transformer_L01.pt
      ...
```

## Video

MP4 files recorded at 10 fps, H.264 codec:

```
task{N}_ep{M}.mp4
```

## Contributing Data to Action Atlas

To contribute your experiment results to the hosted Action Atlas:

1. Run experiments using the standard scripts (they produce the formats above)
2. Generate feature descriptions:
   ```bash
   python action_atlas/generate_feature_descriptions.py \
       --model <model> --concept-id-dir <path> --suites <suites>
   ```
3. Package your results following the directory structure above
4. Open a pull request

Consistent formats ensure your data integrates with the visualization tool.
