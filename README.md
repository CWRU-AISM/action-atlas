# Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models

<p align="center">
  <a href="https://arxiv.org/abs/2603.19233"><img src="https://img.shields.io/badge/arXiv-2603.19233-b31b1b.svg" alt="arXiv"></a>
  <a href="https://cwru-aism.github.io/vla-interp-page/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="#"><img src="https://img.shields.io/badge/🤗_Data-Coming_Soon-yellow" alt="HuggingFace"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>

<p align="center">
  A unified toolkit for mechanistic interpretability of Vision-Language-Action (VLA) models.<br>
  Supports <b>X-VLA</b>, <b>SmolVLA</b>, <b>GR00T N1.5</b>, <b>Pi0.5</b>, and <b>OpenVLA-OFT</b>.
</p>


## Overview

Action Atlas provides tools for understanding *what* VLA models learn and *how* they translate visual and language inputs into robot actions. The toolkit includes:

- **Grid Ablation**: layer-by-layer zeroing to identify critical components
- **Vision Perturbation**: 24 image corruptions to test visual robustness
- **Counterfactual Prompting**: language input variations (null, negation, wrong objects)
- **Cross-Task Injection**: swap activations between tasks to test causal structure
- **Sparse Autoencoder (SAE) Training**: learn interpretable feature dictionaries
- **Concept Identification**: find task-selective SAE features using Cohen's d
- **Concept Ablation & Steering**: remove or amplify features to verify causal roles
- **Action Atlas Visualization**: interactive web app for exploring results

All experiments use a unified CLI and model adapter interface. Run the same experiment on any supported model with `--model xvla` / `--model groot` / etc.

## Supported Models

| Model | Params | Architecture | Environments |
|-------|--------|-------------|-------------|
| [X-VLA](https://github.com/huggingface/lerobot) | 1B | Florence-2 + 24 TransformerBlocks | LIBERO, SimplerEnv |
| [SmolVLA](https://github.com/huggingface/lerobot) | 450M | Interleaved VLM + Expert (32+32 layers) | LIBERO, MetaWorld |
| [GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T) | 3B | Eagle LM + VL-SA + DiT (12+4+16) | LIBERO |
| [Pi0.5](https://github.com/Physical-Intelligence/openpi) | 3B | PaliGemma + Gemma Expert (18 layers) | LIBERO |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | 7B | Llama 2 + OFT (32 layers) | LIBERO |

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/CWRU-AISM/action-atlas.git
cd action-atlas

# Create environment
conda create -y -n actionatlas python=3.12 && conda activate actionatlas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
cd lerobot && pip install -e ".[pi]" && cd ..
cd LIBERO && pip install -e . && cd ..
pip install -e .

# Run an experiment
python experiments/grid_ablation.py --model xvla --suite libero_object --n-episodes 3
```

## Experiments

Run any script with `--help` for full options.

```bash
# Layer-by-layer ablation
python experiments/grid_ablation.py --model xvla --suite libero_object

# Visual robustness (24 perturbation types)
python experiments/vision_perturbation.py --model groot --suite libero_goal

# Counterfactual language prompts
python experiments/counterfactual.py --model smolvla --suite libero_spatial

# Cross-task activation injection
python experiments/cross_task_injection.py --model xvla --suite libero_object --phase both

# Baseline rollouts with activation collection
python experiments/baseline.py --model pi05 --suite libero_goal --collect-activations

# Train SAEs on collected activations
python experiments/train_sae.py \
    --activations-dir outputs/pi05_experiments/baseline_libero_goal/activations

# Concept identification (Cohen's d scoring)
python experiments/concept_id.py \
    --sae-dir outputs/saes/libero_goal \
    --activations-dir outputs/.../activations \
    --suite libero_goal

# Concept ablation & steering
python experiments/concept_ablation.py --model xvla --suite libero_object \
    --sae-dir outputs/saes --concept-results results/concept_id/all_layers.json \
    --layer transformer_L12

python experiments/concept_steering.py --model xvla --suite libero_object \
    --sae-dir outputs/saes --concept-results results/concept_id/all_layers.json \
    --layer transformer_L12 --strengths -2.0 -1.0 1.0 2.0
```

### Replicating Paper Results

See [setup/SETUP.md](setup/SETUP.md) for detailed installation instructions and [experiments/README.md](experiments/README.md) for the full experiment reference.

**Typical workflow:**
1. Collect baseline activations (`baseline.py --collect-activations`)
2. Run grid ablation to find critical layers (`grid_ablation.py`)
3. Train SAEs on activations (`train_sae.py`)
4. Identify concepts with contrastive scoring (`concept_id.py`)
5. Validate causality with ablation/steering (`concept_ablation.py`, `concept_steering.py`)
6. Test robustness with vision perturbation and counterfactual prompting
7. Visualize results in Action Atlas (`action_atlas/`)

## Action Atlas Visualization

Action Atlas is an interactive web app for exploring experiment results.

```bash
# Start the backend
cd action_atlas/backend
python run.py --port 6006

# Open http://localhost:6006 in your browser
```

Features:
- Cross-model comparison of layer criticality and concept representations
- Interactive perturbation and ablation result browsers
- SAE feature explorer with concept search
- Scene state visualization with object displacement tracking

To use with your own data, point the config at your experiment outputs:
```bash
python run.py --port 6006 --data-dir /path/to/your/outputs
```

## Repository Structure

```
action-atlas/
├── experiments/           # Experiment scripts
│   ├── baseline.py        # Baseline rollouts + activation collection
│   ├── grid_ablation.py   # Layer-by-layer ablation
│   ├── vision_perturbation.py
│   ├── counterfactual.py  # Language prompt variations
│   ├── cross_task_injection.py
│   ├── train_sae.py       # SAE training
│   ├── concept_id.py      # Concept identification
│   ├── concept_ablation.py
│   ├── concept_steering.py
│   ├── model_adapters.py  # Uniform model interface
│   ├── hooks.py           # Forward hooks
│   └── launch_parallel.py # Multi-GPU launcher
├── action_atlas/          # Visualization web app
│   ├── api/               # REST API modules
│   ├── backend/           # Flask server
│   ├── frontend/          # Next.js frontend
│   └── data/              # Pre-computed visualization data
├── scripts/               # Analysis and figure generation
├── lerobot/               # LeRobot (submodule)
├── LIBERO/                # LIBERO benchmark (submodule)
└── openvla_oft/           # OpenVLA-OFT (submodule)
```

## Data

Experiment data (activations, SAE checkpoints, rollout videos) will be released on HuggingFace in the coming weeks.

## Citation

```bibtex
@article{grant2026actionatlas,
  title={Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models},
  author={Grant, Bryce and Zhao, Xijia and Wang, Peng},
  journal={arXiv preprint arXiv:2603.19233},
  year={2026},
  url={https://arxiv.org/abs/2603.19233}
}
```

## Acknowledgments

The Action Atlas visualization tool draws inspiration from:
- [**Neuronpedia**](https://neuronpedia.org/) - Interactive platform for exploring SAE features in language models ([Neuronpedia, 2024](https://neuronpedia.org/))
- [**ConceptViz**](https://github.com/Happy-Hippo209/ConceptViz) - Visual analytics tool for SAE feature exploration ([Zhao et al., 2025](https://github.com/Happy-Hippo209/ConceptViz))


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
