# Action Atlas: Setup Guide

Setup for running mechanistic interpretability experiments on VLA models
(X-VLA, SmolVLA, GR00T N1.5, Pi0.5, OpenVLA-OFT).

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/CWRU-AISM/action-atlas.git
cd action-atlas

# Create conda environment
conda create -y -n actionatlas python=3.12 && conda activate actionatlas

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install LeRobot (provides X-VLA, SmolVLA, GR00T, Pi0.5 policies)
cd lerobot && pip install -e ".[pi]" && cd ..

# Install LIBERO (evaluation environments)
cd LIBERO && pip install -e . && cd ..

# Install project dependencies
pip install -e .

# Verify
python -c "from lerobot.policies.xvla.modeling_xvla import XVLAPolicy; print('X-VLA OK')"
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('SmolVLA OK')"
python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print('Pi0.5 OK')"
python -c "import libero; print('LIBERO OK')"
```

## Submodules

| Submodule | Source | Purpose |
|-----------|--------|---------|
| `lerobot/` | huggingface/lerobot | Model policies: X-VLA, SmolVLA, GR00T N1.5, Pi0.5 |
| `LIBERO/` | Lifelong-Robot-Learning/LIBERO | LIBERO benchmark environments |
| `openvla_oft/` | moojink/openvla-oft | OpenVLA-OFT model (separate conda env) |

## Conda Environments

| Environment | Python | Models | Key constraint |
|-------------|--------|--------|---------------|
| **actionatlas** | 3.12 | X-VLA, SmolVLA, Pi0.5 | transformers==4.53 |
| **groot** | 3.12 | GR00T N1.5 | transformers>=5.0, flash-attn, peft |
| **openvla-oft** | 3.10 | OpenVLA-OFT | torch==2.2, prismatic |

GR00T requires `transformers>=5.0` for its Eagle VLM processor, which
conflicts with the other models. A separate environment avoids this.

## Detailed Installation

### System Dependencies (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

### Primary Environment (X-VLA, SmolVLA, Pi0.5)

```bash
conda create -y -n actionatlas python=3.12
conda activate actionatlas
conda install -y ffmpeg -c conda-forge

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd lerobot && pip install -e ".[pi]" && cd ..
cd LIBERO && pip install -e . && cd ..
pip install -e .
pip install flask flask-cors  # for Action Atlas visualization
```

### GR00T N1.5 Environment

```bash
conda create -y -n groot python=3.12
conda activate groot
conda install -y ffmpeg -c conda-forge

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd lerobot && pip install -e ".[pi]" && cd ..
cd LIBERO && pip install -e . && cd ..
pip install -e .

# GR00T requires these additional packages
pip install "transformers>=5.0" peft
conda install -y cuda-nvcc cuda-toolkit -c nvidia
pip install flash-attn --no-build-isolation
```

### OpenVLA-OFT Environment

```bash
conda create -y -n openvla-oft python=3.10
conda activate openvla-oft

pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla_oft && pip install -e . && cd ..
cd LIBERO && pip install -e . && cd ..
```

If `import libero` fails after installation, add it to your PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO
```

### HuggingFace Authentication

Pi0.5 and GR00T require gated model access:

```bash
pip install huggingface_hub
hf auth login
```

Accept the license at https://huggingface.co/google/paligemma-3b-pt-224

## Download Checkpoints

The model adapters in `experiments/model_adapters.py` and `experiments/groot_common.py` know which HuggingFace repo to load for each (model, suite) pair, so most checkpoints download lazily on first use. Pre-download is optional but recommended for offline runs and Docker builds.

### Pi0.5

| Suite | HF repo |
|-------|---------|
| All 4 LIBERO suites | `lerobot/pi05_libero_finetuned` |

```bash
hf download lerobot/pi05_libero_finetuned --local-dir checkpoints/pi05_libero_finetuned
```

### X-VLA

| Suite | HF repo |
|-------|---------|
| All 4 LIBERO suites | `lerobot/xvla-libero` |
| SimplerEnv WidowX | `lerobot/xvla-widowx` |
| SimplerEnv Google Robot | `lerobot/xvla-google_robot` |

X-VLA downloads automatically on first load; pre-download with `hf download lerobot/xvla-libero` if you want it cached locally.

### SmolVLA

| Suite | HF repo |
|-------|---------|
| All 4 LIBERO suites | `HuggingFaceVLA/smolvla_libero` |
| MetaWorld MT50 | `jadechoghari/smolvla_metaworld` |

```bash
hf download HuggingFaceVLA/smolvla_libero --local-dir checkpoints/smolvla_libero
hf download jadechoghari/smolvla_metaworld --local-dir checkpoints/smolvla_metaworld
```

### GR00T N1.5

GR00T uses a different checkpoint per LIBERO suite (community-maintained, no single official LIBERO release).

| Suite | HF repo |
|-------|---------|
| `libero_goal` | `aractingi/libero-groot-goal` |
| `libero_object` | `liorbenhorin-nv/groot-libero_object-64_40000` |
| `libero_spatial` | `liorbenhorin-nv/groot-libero_spatial-128_20000` |
| `libero_10` | `aractingi/groot-libero-10` |

```bash
hf download aractingi/libero-groot-goal                    --local-dir checkpoints/groot_libero_goal
hf download liorbenhorin-nv/groot-libero_object-64_40000   --local-dir checkpoints/groot_libero_object
hf download liorbenhorin-nv/groot-libero_spatial-128_20000 --local-dir checkpoints/groot_libero_spatial
hf download aractingi/groot-libero-10                      --local-dir checkpoints/groot_libero_10
```

The `libero_spatial` checkpoint underperforms the published reference (community fine-tunes range 68 to 94 percent; official reports 97.65 percent); spatial intervention experiments are reported with that caveat.

### OpenVLA-OFT

OFT uses one checkpoint per LIBERO suite (official `moojink` releases). Loaded only inside the `openvla-oft` conda environment.

| Suite | HF repo |
|-------|---------|
| `libero_spatial` | `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| `libero_object` | `moojink/openvla-7b-oft-finetuned-libero-object` |
| `libero_goal` | `moojink/openvla-7b-oft-finetuned-libero-goal` |
| `libero_10` | `moojink/openvla-7b-oft-finetuned-libero-10` |
| Combined (4 suites) | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |

```bash
conda activate openvla-oft
hf download moojink/openvla-7b-oft-finetuned-libero-spatial --local-dir checkpoints/openvla-oft-spatial
hf download moojink/openvla-7b-oft-finetuned-libero-object  --local-dir checkpoints/openvla-oft-object
hf download moojink/openvla-7b-oft-finetuned-libero-goal    --local-dir checkpoints/openvla-oft-goal
hf download moojink/openvla-7b-oft-finetuned-libero-10      --local-dir checkpoints/openvla-oft-10
```

Each OFT checkpoint is ~16 GB (7B base + LoRA adapter + dataset statistics).

## Running Experiments

```bash
conda activate actionatlas
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1

# Grid ablation
python experiments/grid_ablation.py --model xvla --suite libero_object --n-episodes 3

# Vision perturbation
python experiments/vision_perturbation.py --model groot --suite libero_goal

# Multi-GPU
python experiments/launch_parallel.py grid_ablation \
    --gpus 0 1 2 3 --suites libero_goal libero_object libero_spatial libero_10 \
    --model xvla --n-episodes 3

# See experiments/README.md for the full experiment reference
```

## Troubleshooting

**CUDA out of memory:** Set `TORCH_COMPILE_DISABLE=1` or reduce `--n-episodes`

**PaliGemma access denied:** Accept license at https://huggingface.co/google/paligemma-3b-pt-224

**NumPy 2.x errors:** `pip install "numpy<2"`

**MuJoCo rendering:** `export MUJOCO_GL=egl`
