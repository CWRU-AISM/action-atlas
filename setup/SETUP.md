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

# Install LeRobot (provides X-VLA, SmolVLA, GR00T, Pi0.5 policies).
# The [libero] extra pulls hf-libero, which provides robosuite/mujoco/bddl
# built to work with numpy 2.x. Requires system cmake (see System Dependencies
# below) to build egl_probe. Do NOT `pip install -r LIBERO/requirements.txt`;
# its pins (numpy==1.22.4 etc.) are stale upstream pins that conflict with
# lerobot and are not needed.
cd lerobot && pip install -e ".[pi,smolvla,libero,metaworld]" && cd ..

# Install LIBERO (evaluation environments)
cd LIBERO && pip install -e . && cd ..

# Install project dependencies
pip install -e .

# Verify
python -c "from lerobot.policies.xvla.modeling_xvla import XVLAPolicy; print('X-VLA OK')"
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('SmolVLA OK')"
python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print('Pi0.5 OK')"
python -c "from libero.libero.envs import OffScreenRenderEnv; print('LIBERO OK')"
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
| **actionatlas** | 3.12 | X-VLA, SmolVLA, Pi0.5 | transformers==5.3 (pinned by lerobot), numpy 2.2.x |
| **groot** | 3.12 | GR00T N1.5 | flash-attn (CUDA 12.8 toolchain), peft |
| **openvla-oft** | 3.10 | OpenVLA-OFT | torch==2.2, numpy<2, prismatic |

GR00T additionally requires flash-attn and peft; compiling flash-attn
needs a CUDA toolchain matching the cu128 PyTorch build, so a separate
environment keeps that isolated.

The SimplerEnv X-VLA experiments (`experiments/simplerenv/`) use a separate
`simpler_env` environment (Python 3.10, numpy<2). See "SimplerEnv environment"
below.

Note on numpy: the actionatlas and groot envs run numpy 2.2.x; robosuite
1.4.x works fine with it (via hf-libero). The openvla-oft env must stay on
numpy<2 because torch 2.2.0 predates numpy 2 and cannot interoperate with it.

## Detailed Installation

### System Dependencies (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libosmesa6-dev libgl1 libglx-mesa0 libglfw3 patchelf
```

This list is confirmed on Ubuntu 24.04. On Ubuntu 22.04 and older, replace
`libgl1 libglx-mesa0` with `libgl1-mesa-glx` (the package was renamed in
24.04).

This step is required before the pip installs below: building `egl_probe`
(pulled in by the lerobot `[libero]` extra) needs system `cmake` and the
GL headers.

### Primary Environment (X-VLA, SmolVLA, Pi0.5)

```bash
conda create -y -n actionatlas python=3.12
conda activate actionatlas
conda install -y ffmpeg -c conda-forge

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd lerobot && pip install -e ".[pi,smolvla,libero,metaworld]" && cd ..
cd LIBERO && pip install -e . && cd ..
pip install -e .
pip install flask flask-cors  # for Action Atlas visualization
```

The `[libero]` extra (hf-libero) provides the robosuite/mujoco/bddl stack
that LIBERO environments need at runtime; without it `import libero`
succeeds but creating an environment fails with
`ModuleNotFoundError: No module named 'robosuite'`. `[metaworld]` is
needed for the MetaWorld experiment suite (`experiments/metaworld/`).

### GR00T N1.5 Environment

```bash
conda create -y -n groot python=3.12
conda activate groot
conda install -y ffmpeg -c conda-forge

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd lerobot && pip install -e ".[pi,libero]" && cd ..
cd LIBERO && pip install -e . && cd ..
pip install -e .

# GR00T requires these additional packages. The lerobot [groot] extra brings
# dm-tree, timm, decord etc.; transformers stays at 5.3.0 (lerobot's pin).
cd lerobot && pip install -e ".[groot]" && cd ..
pip install peft

# The pinned lerobot commit's GR00T code predates transformers 5.3; apply the
# compat patch (4 small hunks: lazy Beta init, all_tied_weights_keys shim,
# Eagle processor fixes backported from upstream lerobot PR #3652):
cd lerobot && git apply ../setup/patches/lerobot-groot-transformers5.patch && cd ..

# CUDA toolchain for compiling flash-attn. MUST be pinned to the same CUDA
# version as the PyTorch build (cu128); an unpinned install pulls CUDA 13+
# and flash-attn fails with "The detected CUDA version mismatches ...".
conda install -y cuda-nvcc=12.8 cuda-toolkit=12.8 -c nvidia

# conda's cuda-toolkit also installs a conda compiler toolchain whose
# CC/CXX/CXXFLAGS break nvcc during the flash-attn build. Compile with the
# system gcc instead, pointing it at conda's CUDA headers:
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/targets/x86_64-linux/lib/stubs
MAX_JOBS=12 pip install flash-attn --no-build-isolation
```

No prebuilt flash-attn wheel exists for the torch version currently served by
the cu128 index (2.10), so the source build above is required (roughly 1-2 h).
Verify:
```bash
python -c "import flash_attn; print('flash-attn OK')"
python -c "from lerobot.policies.groot.modeling_groot import GrootPolicy; print('GR00T OK')"
python experiments/baseline.py --model groot --suite libero_goal --tasks 0 --n-episodes 1
```

Note: upstream lerobot has since fixed GR00T for transformers >=5.4 (PR #3652
and follow-ups). Updating the lerobot submodule would make
`setup/patches/lerobot-groot-transformers5.patch` unnecessary, but requires
re-validating the other policies against the newer lerobot API.

### OpenVLA-OFT Environment

```bash
conda create -y -n openvla-oft python=3.10
conda activate openvla-oft

pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla_oft && pip install -e . && cd ..
cd LIBERO && pip install -e . && cd ..

# LIBERO runtime deps (robosuite etc.) for the OFT eval scripts
pip install -r openvla_oft/experiments/robot/libero/libero_requirements.txt

# Project deps (tyro etc.) so experiments/*.py can run in this env
pip install -e .

# The two steps above upgrade numpy to 2.x, which torch 2.2.0 cannot use at
# all (RuntimeError: Numpy is not available). Downgrade it back LAST:
pip install "numpy<2"

# libero_requirements.txt leaves mujoco unpinned; current mujoco (3.10+)
# changed the mj_fullM() signature and breaks robosuite 1.4 controllers:
pip install "mujoco==3.1.6"

# tensorflow_datasets pulls a tensorflow_metadata that needs protobuf 5.x
# while tensorflow 2.15 pins protobuf 4.x; without these pins `import
# prismatic` fails with "cannot import name 'runtime_version'". protobuf is
# then re-pinned to 4.25 so wandb keeps working (tensorflow_metadata's
# declared <4.21 pin is stricter than its code actually needs):
pip install "tensorflow_metadata<1.17"
pip install "protobuf==4.25.*"
```

With current pip, the editable LIBERO install does not put `libero` on the
import path (LIBERO has no top-level `__init__.py`, so the PEP 660 editable
finder maps nothing). Add it to your PYTHONPATH; this is required, not
optional, in this environment:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO
```

Verify:
```bash
cd action-atlas   # repo root
export PYTHONPATH=$PYTHONPATH:$(pwd)/LIBERO
python -c "from libero.libero.envs import OffScreenRenderEnv; print('LIBERO OK')"
python -c "import prismatic; print('prismatic OK')"
```

### SimplerEnv Environment

For the X-VLA SimplerEnv experiments in `experiments/simplerenv/`. SAPIEN needs
an NVIDIA GPU; ray-traced scenes are slow on non-RTX cards.

```bash
conda create -y -n simpler_env python=3.10 && conda activate simpler_env

# ManiSkill2 real-to-sim environments, then SimplerEnv itself
cd SimplerEnv/ManiSkill2_real2sim && pip install -e . && cd ../..
cd SimplerEnv && pip install -e . && cd ..

# Pin these AFTER the two installs above, which pull incompatible versions:
#   numpy>=2 breaks pinocchio IK, and sapien imports pkg_resources, which
#   setuptools 81 removed. opencv 5.x hard-requires numpy>=2.
pip install "numpy==1.24.4" "setuptools<81" "opencv-python==4.9.0.80"

# Verify
MUJOCO_GL=egl python -c "
import simpler_env
env = simpler_env.make('widowx_put_eggplant_in_sink')
obs, info = env.reset()
print(env.unwrapped.get_language_instruction())
env.step(env.action_space.sample())
print('SimplerEnv OK')"
```

The pin ordering matters. The install instructions in the vendored
`SimplerEnv/README.md` put numpy first, which does not hold: `ManiSkill2_real2sim` upgrades it back to 2.x.

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

Unlike the other models, this pre-download is REQUIRED for Pi0.5: the adapter's
default checkpoint is the local `checkpoints/pi05_libero_finetuned` directory
(alternatively pass `--checkpoint lerobot/pi05_libero_finetuned` to load from
the Hub directly).

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
hf download moojink/openvla-7b-oft-finetuned-libero-spatial --local-dir data/checkpoints/openvla-oft-spatial
hf download moojink/openvla-7b-oft-finetuned-libero-object  --local-dir data/checkpoints/openvla-oft-object
hf download moojink/openvla-7b-oft-finetuned-libero-goal    --local-dir data/checkpoints/openvla-oft-goal
hf download moojink/openvla-7b-oft-finetuned-libero-10      --local-dir data/checkpoints/openvla-oft-10
```

Each OFT checkpoint is ~16 GB (7B base + LoRA adapter + dataset statistics).
Note the `data/` prefix: the OFT adapter resolves checkpoints under
`ACTION_ATLAS_DATA_ROOT` (default `data/`), unlike the other models.

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

**numpy version conflicts (e.g. "robosuite needs numpy==1.22.4 but lerobot
needs 2.2.x"):** Do not install `LIBERO/requirements.txt`; those pins are
stale upstream pins and are not used by this setup (LIBERO's `setup.py`
declares no dependencies). The robosuite stack comes from the lerobot
`[libero]` extra (hf-libero) and works with numpy 2.2.x. The only env that
needs `numpy<2` is openvla-oft (torch 2.2.0 cannot use numpy 2.x).

**`ModuleNotFoundError: No module named 'robosuite'` when creating a LIBERO
env:** The lerobot `[libero]` extra is missing; run `cd lerobot && pip install
-e ".[pi,libero]"` (actionatlas/groot envs), or install
`openvla_oft/experiments/robot/libero/libero_requirements.txt` (openvla-oft env).

**Checkpoint download hangs forever (stuck in `xet_get`):** The HuggingFace
Xet transfer backend can stall. Set `export HF_HUB_DISABLE_XET=1` and retry.

**MuJoCo rendering:** `export MUJOCO_GL=egl`
