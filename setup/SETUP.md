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
| `SimplerEnv/` | squarefk/SimplerEnv | SimplerEnv simulation (WidowX, Google Robot) |

## Conda Environments

| Environment | Python | Models / Envs | Key constraint |
|-------------|--------|---------------|---------------|
| **actionatlas** | 3.12 | X-VLA, SmolVLA, Pi0.5 on LIBERO; SmolVLA on MetaWorld | transformers==4.53 |
| **groot** | 3.12 | GR00T N1.5 on LIBERO | transformers>=5.0, flash-attn, peft |
| **openvla-oft** | 3.10 | OpenVLA-OFT on LIBERO | torch==2.2, prismatic |
| **simpler_env** | 3.10 | X-VLA on SimplerEnv (WidowX, Google Robot) | numpy<2, sapien 2.2.x |

GR00T requires `transformers>=5.0` for its Eagle VLM processor, which
conflicts with the other models. A separate environment avoids this.

SimplerEnv requires Python 3.10 and numpy<2 because SAPIEN 2.x is compiled
against the NumPy 1.x C ABI. NumPy 2.x causes a segfault in
`env.step()` (SAPIEN's `PinocchioModel.compute_forward_kinematics`).

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

### SimplerEnv Environment (X-VLA cross-embodiment)

SimplerEnv provides WidowX and Google Robot simulation environments for
X-VLA cross-embodiment experiments. Requires a separate conda env because
SAPIEN 2.x needs Python 3.10 and numpy<2.

```bash
# Create env (use /data if root disk is low on space)
conda create -y -n simpler_env python=3.10 -p /data/miniconda3/envs/simpler_env
conda activate simpler_env

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SimplerEnv + ManiSkill2 (included as submodule)
cd SimplerEnv && git submodule update --init --recursive
cd ManiSkill2_real2sim && pip install -e . && cd ..
pip install -e . && cd ..

# Lerobot (need pre-3.12 version for Python 3.10 compat)
# Checkout the last commit before the Python 3.12 requirement (d324ffe8)
cd /path/to/lerobot && git worktree add /data/lerobot_py310 d324ffe8
cd /data/lerobot_py310 && pip install -e .

# CRITICAL: downgrade numpy after all installs (SAPIEN segfaults with numpy 2.x)
pip install 'numpy<2'

# Extra dependencies
pip install transformers tqdm charset_normalizer httpx

# Verify
python -c "import simpler_env; print('SimplerEnv OK')"
python -c "
import sys, types
for m in ['lerobot.policies.groot', 'lerobot.policies.groot.configuration_groot',
          'lerobot.policies.groot.modeling_groot', 'lerobot.policies.groot.groot_n1']:
    sys.modules[m] = types.ModuleType(m)
sys.modules['lerobot.policies.groot.configuration_groot'].GrootConfig = type('GrootConfig', (), {})
sys.modules['lerobot.policies.groot.modeling_groot'].GrootPolicy = type('GrootPolicy', (), {})
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
print('X-VLA OK')
"
```

The GR00T stub is needed because lerobot's policy `__init__` imports GR00T,
which triggers a xformers/diffusers import chain that segfaults in the
simpler_env conda env due to an incompatible xformers binary.

SimplerEnv experiments are in `experiments/simplerenv/`:
```bash
conda activate simpler_env
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1

# Smoke test
python experiments/simplerenv/xvla_simplerenv_eval.py \
    --model widowx --task widowx_stack_cube --n_episodes 1

# Full evaluation
python experiments/simplerenv/xvla_simplerenv_eval.py \
    --model widowx --all-tasks --n_episodes 20
```

### HuggingFace Authentication

Pi0.5 and GR00T require gated model access:

```bash
pip install huggingface_hub
hf auth login
```

Accept the license at https://huggingface.co/google/paligemma-3b-pt-224

## Download Checkpoints

```bash
mkdir -p checkpoints

# Pi0.5
hf download lerobot/pi05_libero_finetuned \
    --local-dir checkpoints/pi05_libero_finetuned

# X-VLA and SmolVLA download automatically when loading()

# GR00T N1.5 (per-suite, from lerobot community)
hf download aractingi/libero-groot-goal \
    --local-dir checkpoints/groot_libero_goal
hf download liorbenhorin-nv/groot-libero_object-64_40000 \
    --local-dir checkpoints/groot_libero_object
hf download liorbenhorin-nv/groot-libero_spatial-128_20000 \
    --local-dir checkpoints/groot_libero_spatial
```

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
