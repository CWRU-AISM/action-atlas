# Contributing to Action Atlas

Contributions of new model analyses, feature descriptions,
experiment results, and code improvements are welcome.

## Adding a New Model

1. Create a model adapter in `experiments/model_adapters.py`
   - Implement `load_model()`, `get_layer_groups()`, `create_env()`, `run_episode()`
   - Register in `ADAPTERS` dict
   - For interleaved architectures (SmolVLA, Pi0.5), hook `.mlp` submodules
   - Add env caching in `__init__`

2. Test with grid ablation:
   ```bash
   python experiments/grid_ablation.py --model <name> --suite libero_object \
       --n-episodes 1 --tasks 0 --layers <first_layer> --gpu 0
   ```

## Contributing Experiment Results

All experiments produce standardized output formats (see [DATA_FORMAT.md](DATA_FORMAT.md)).

```bash
python experiments/baseline.py --model <model> --suite <suite> --collect-activations
python experiments/grid_ablation.py --model <model> --suite <suite>
python experiments/train_sae.py --activations-dir <path>
python experiments/concept_id.py --sae-dir <path> --activations-dir <path> --suite <suite>
```

To generate feature descriptions:
```bash
export GOOGLE_API_KEY=<your_key>

python action_atlas/generate_feature_descriptions.py \
    --model <model> --pathway <pathway> \
    --concept-id-dir <path_to_concept_id_results> \
    --suites libero_goal libero_object libero_spatial
```

## Running Action Atlas Locally

```bash
pip install flask flask-cors

cd action_atlas/backend
python run.py --port 6006
```

To point at your own experiment data:
```bash
export XVLA_ROLLOUTS_DIR=/path/to/your/xvla/data
export SMOLVLA_ROLLOUTS_DIR=/path/to/your/smolvla/data
```

## Reporting Issues

Please include:
- Model name and checkpoint
- Experiment script and arguments
- Full error traceback
- Python/torch/lerobot versions
