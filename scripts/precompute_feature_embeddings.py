#!/usr/bin/env python3
"""Pre-compute sentence embeddings for all SAE feature descriptions.

Uses sentence-transformers on A100 GPUs to encode feature descriptions
into embeddings for fast semantic search in the Action Atlas frontend.

Output: action_atlas/data/feature_embeddings/{model}_embeddings.npz
Each file contains:
  - embeddings: (N, 384) float32 array of description embeddings
  - feature_ids: (N,) array of feature indices
  - descriptions: (N,) array of description strings
  - layers: (N,) array of layer names
  - suites: (N,) array of suite names
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os
import time

# Use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUTPUT_DIR = Path("action_atlas/data/feature_embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

DESC_DIR = Path("action_atlas/data/descriptions/contrastive")

MODEL_CONFIGS = {
    'pi05_expert': {'dir': 'pi05_expert', 'model_name': 'pi05'},
    'pi05_paligemma': {'dir': 'pi05_paligemma', 'model_name': 'pi05'},
    'oft_single': {'dir': 'oft_single', 'model_name': 'oft'},
    'xvla': {'dir': 'xvla', 'model_name': 'xvla'},
    'smolvla': {'dir': 'smolvla', 'model_name': 'smolvla'},
    'groot': {'dir': 'groot', 'model_name': 'groot'},
}


def load_all_descriptions(desc_dir: Path) -> list[dict]:
    """Load all feature descriptions from a model's description directory."""
    entries = []
    if not desc_dir.exists():
        return entries

    for json_file in sorted(desc_dir.glob("*.json")):
        try:
            data = json.load(open(json_file))
            descriptions = data.get("descriptions", {})
            layer = data.get("layer", "")
            suite = data.get("suite", "")
            pathway = data.get("pathway", "")
            model = data.get("model", "")

            # Build layer name from metadata
            if pathway and layer is not None:
                layer_name = f"{pathway}_layer_{layer}"
            elif "hook_point" in data:
                layer_name = data["hook_point"]
            else:
                layer_name = str(layer)

            for feat_id, desc in descriptions.items():
                if desc and isinstance(desc, str) and len(desc) > 10:
                    entries.append({
                        "feature_id": int(feat_id) if feat_id.isdigit() else feat_id,
                        "description": desc,
                        "layer": layer_name,
                        "suite": suite,
                    })
        except (json.JSONDecodeError, IOError):
            continue

    return entries


def main():
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    t0 = time.time()
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    total_features = 0

    for config_name, config in MODEL_CONFIGS.items():
        desc_dir = DESC_DIR / config['dir']
        print(f"\nProcessing {config_name} from {desc_dir}...")

        entries = load_all_descriptions(desc_dir)
        if not entries:
            print(f"  No descriptions found, skipping")
            continue

        print(f"  {len(entries)} feature descriptions loaded")

        # Extract descriptions for encoding
        descriptions = [e["description"] for e in entries]
        feature_ids = np.array([e["feature_id"] for e in entries])
        layers = np.array([e["layer"] for e in entries])
        suites = np.array([e["suite"] for e in entries])

        # Encode in batches
        t1 = time.time()
        embeddings = encoder.encode(
            descriptions,
            batch_size=512,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )
        print(f"  Encoded {len(descriptions)} descriptions in {time.time()-t1:.1f}s")
        print(f"  Embedding shape: {embeddings.shape}")

        # Save as npz
        out_path = OUTPUT_DIR / f"{config_name}_embeddings.npz"
        np.savez_compressed(
            out_path,
            embeddings=embeddings.astype(np.float32),
            feature_ids=feature_ids,
            descriptions=np.array(descriptions),
            layers=layers,
            suites=suites,
        )
        print(f"  Saved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
        total_features += len(entries)

    # Also save the encoder's query embedding function as a small index
    # Pre-compute common query embeddings for autocomplete
    common_queries = [
        # Motion concepts
        "grasp", "grip", "pick up", "place", "put down", "push", "pull",
        "open", "close", "turn", "rotate", "slide", "lift", "lower",
        "approach", "reach", "retract", "release", "press", "insert",
        # Object concepts
        "bowl", "plate", "cup", "mug", "bottle", "drawer", "cabinet",
        "stove", "microwave", "door", "handle", "button", "rack",
        "cream cheese", "wine bottle", "book", "basket", "spoon",
        # Spatial concepts
        "on top", "inside", "in front", "behind", "left", "right",
        "above", "below", "next to", "between", "center", "edge",
        # Action phases
        "transport", "pre-grasp", "post-place", "alignment", "contact",
        # MetaWorld
        "hammer", "screw", "peg", "lever", "faucet", "window", "shelf",
        "coffee", "basketball", "soccer", "bin", "sweep",
    ]

    t2 = time.time()
    query_embeddings = encoder.encode(
        common_queries,
        batch_size=128,
        normalize_embeddings=True,
    )
    print(f"\nEncoded {len(common_queries)} common queries in {time.time()-t2:.1f}s")

    np.savez_compressed(
        OUTPUT_DIR / "common_queries.npz",
        queries=np.array(common_queries),
        embeddings=query_embeddings.astype(np.float32),
    )

    print(f"\nTotal: {total_features} feature descriptions across {len(MODEL_CONFIGS)} model configs")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
