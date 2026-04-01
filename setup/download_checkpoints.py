"""
Download model checkpoints from HuggingFace (LeRobot).

Usage:
    python setup/download_checkpoints.py --model pi0_base
    python setup/download_checkpoints.py --model pi0_libero_finetuned
    python setup/download_checkpoints.py --model pi05_libero_finetuned
    python setup/download_checkpoints.py --model smolvla
    python setup/download_checkpoints.py --model all
    python setup/download_checkpoints.py --list

Available checkpoints (from https://huggingface.co/lerobot):
    Pi0 Models (~14GB each):
    - pi0_base: Pi0 base model
    - pi0_libero_base: Pi0 pretrained on LIBERO
    - pi0_libero_finetuned: Pi0 finetuned on LIBERO

    Pi0.5 Models (~14GB each):
    - pi05_base: Pi0.5 base model
    - pi05_libero_base: Pi0.5 pretrained on LIBERO
    - pi05_libero_finetuned: Pi0.5 finetuned on LIBERO
    - pi05_libero_finetuned_quantiles: Pi0.5 with quantile actions

    Smaller Models:
    - smolvla: SmolVLA base model (smaller, faster)
    - xvla_base: xVLA base model (0.9B params)
    - xvla_libero: xVLA finetuned on LIBERO (0.9B params)
    - xvla_widowx: xVLA for WidowX robot
    - xvla_folding: xVLA for folding tasks
    - xvla_google_robot: xVLA for Google robot
    - xvla_agibot_world: xVLA for Agibot

    Other Models:
    - diffusion_pusht: Diffusion policy for PushT
    - act_aloha_transfer: ACT for ALOHA transfer cube
    - act_aloha_insertion: ACT for ALOHA insertion
    - vqbet_pusht: VQ-BeT for PushT
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


CHECKPOINTS = {
    # ==================== Pi0 Models (4B params, ~14GB) ====================
    "pi0_base": {
        "repo_id": "lerobot/pi0_base",
        "description": "Pi0 base model (4B params, ~14GB)",
        "category": "pi0",
    },
    "pi0_libero_base": {
        "repo_id": "lerobot/pi0_libero_base",
        "description": "Pi0 pretrained on LIBERO (4B params)",
        "category": "pi0",
    },
    "pi0_libero_finetuned": {
        "repo_id": "lerobot/pi0_libero_finetuned",
        "description": "Pi0 finetuned on LIBERO (4B params)",
        "category": "pi0",
    },
    "pi0_old": {
        "repo_id": "lerobot/pi0_old",
        "description": "Pi0 legacy version",
        "category": "pi0",
    },
    # ==================== Pi0.5 Models (4B params, ~14GB) ====================
    "pi05_base": {
        "repo_id": "lerobot/pi05_base",
        "description": "Pi0.5 base model (4B params, ~14GB)",
        "category": "pi05",
    },
    "pi05_libero_base": {
        "repo_id": "lerobot/pi05_libero_base",
        "description": "Pi0.5 pretrained on LIBERO (4B params)",
        "category": "pi05",
    },
    "pi05_libero_finetuned": {
        "repo_id": "lerobot/pi05_libero_finetuned",
        "description": "Pi0.5 finetuned on LIBERO (4B params)",
        "category": "pi05",
    },
    "pi05_libero_finetuned_quantiles": {
        "repo_id": "lerobot/pi05_libero_finetuned_quantiles",
        "description": "Pi0.5 with quantile action outputs",
        "category": "pi05",
    },
    # ==================== SmolVLA ====================
    "smolvla": {
        "repo_id": "lerobot/smolvla_base",
        "description": "SmolVLA base model (smaller, efficient)",
        "category": "smolvla",
    },
    # ==================== xVLA Models (0.9B params) ====================
    "xvla_base": {
        "repo_id": "lerobot/xvla-base",
        "description": "xVLA base model (0.9B params)",
        "category": "xvla",
    },
    "xvla_libero": {
        "repo_id": "lerobot/xvla-libero",
        "description": "xVLA finetuned on LIBERO (0.9B)",
        "category": "xvla",
    },
    "xvla_widowx": {
        "repo_id": "lerobot/xvla-widowx",
        "description": "xVLA for WidowX robot (0.9B)",
        "category": "xvla",
    },
    "xvla_folding": {
        "repo_id": "lerobot/xvla-folding",
        "description": "xVLA for folding tasks (0.9B)",
        "category": "xvla",
    },
    "xvla_google_robot": {
        "repo_id": "lerobot/xvla-google-robot",
        "description": "xVLA for Google robot (0.9B)",
        "category": "xvla",
    },
    "xvla_agibot_world": {
        "repo_id": "lerobot/xvla-agibot-world",
        "description": "xVLA for Agibot (0.9B)",
        "category": "xvla",
    },
    # ==================== Other Models ====================
    "diffusion_pusht": {
        "repo_id": "lerobot/diffusion_pusht",
        "description": "Diffusion policy for PushT task",
        "category": "other",
    },
    "diffusion_pusht_keypoints": {
        "repo_id": "lerobot/diffusion_pusht_keypoints",
        "description": "Diffusion policy for PushT (keypoints)",
        "category": "other",
    },
    "act_aloha_transfer": {
        "repo_id": "lerobot/act_aloha_sim_transfer_cube_human",
        "description": "ACT for ALOHA transfer cube task",
        "category": "other",
    },
    "act_aloha_insertion": {
        "repo_id": "lerobot/act_aloha_sim_insertion_human",
        "description": "ACT for ALOHA insertion task",
        "category": "other",
    },
    "vqbet_pusht": {
        "repo_id": "lerobot/vqbet_pusht",
        "description": "VQ-BeT for PushT task",
        "category": "other",
    },
}


def download_checkpoint(name: str, output_dir: Path, force: bool = False):
    """Download a checkpoint from HuggingFace."""
    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint: {name}. Available: {list(CHECKPOINTS.keys())}")

    info = CHECKPOINTS[name]
    repo_id = info["repo_id"]
    target_dir = output_dir / repo_id.replace("/", "_")

    if target_dir.exists() and not force:
        print(f"Checkpoint {name} already exists at {target_dir}")
        print("  Use --force to re-download")
        return target_dir

    print(f"Downloading {info['description']}...")
    print(f"  Repo: {repo_id}")
    print(f"  Target: {target_dir}")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
    )

    print(f"  Downloaded to: {path}")
    return Path(path)


def list_checkpoints():
    """Print available checkpoints organized by category."""
    print("Available checkpoints:\n")

    categories = [
        ("pi0", "Pi0 Models (4B params, ~14GB each)"),
        ("pi05", "Pi0.5 Models (4B params, ~14GB each)"),
        ("smolvla", "SmolVLA"),
        ("xvla", "xVLA Models (0.9B params)"),
        ("other", "Other Models (Diffusion, ACT, VQ-BeT)"),
    ]

    for cat_id, cat_name in categories:
        cat_models = [(n, i) for n, i in CHECKPOINTS.items() if i.get("category") == cat_id]
        if cat_models:
            print(f"{cat_name}:")
            for name, info in cat_models:
                print(f"  {name:30} {info['description']}")
            print()

    print("See all models: https://huggingface.co/lerobot")


def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints from HuggingFace (LeRobot)")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(CHECKPOINTS.keys()) + ["all"],
        default=None,
        help="Which checkpoint to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints"
    )

    args = parser.parse_args()

    if args.list or args.model is None:
        list_checkpoints()
        if args.model is None:
            print("\nUsage: python setup/download_checkpoints.py --model <name>")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Model Checkpoint Downloader")
    print(f"Output directory: {output_dir.absolute()}\n")

    if args.model == "all":
        models = list(CHECKPOINTS.keys())
    else:
        models = [args.model]

    print(f"Downloading: {models}\n")

    downloaded = []
    for model in models:
        try:
            path = download_checkpoint(model, output_dir, args.force)
            downloaded.append((model, path))
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDownload complete!")
    print("Downloaded checkpoints:")
    for model, path in downloaded:
        print(f"  {model}: {path}")

    print("\nUsage:")
    print("  Set checkpoint path in experiment scripts:")
    print(f"  --checkpoint {output_dir}/<model_name>")


if __name__ == "__main__":
    main()
