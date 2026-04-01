#!/usr/bin/env python3
"""Generate feature descriptions for SAE features using LLM APIs.

Supports all VLA models (Pi0.5, OpenVLA-OFT, X-VLA, SmolVLA, GR00T).
Uses Claude, Gemini, or rule-based fallback to produce human-readable
descriptions of what each SAE feature detects.

Generated descriptions power the Action Atlas feature explorer and can
be contributed back to the hosted instance.

Examples:
    # X-VLA features using Gemini
    GOOGLE_API_KEY=... python action_atlas/generate_feature_descriptions.py \\
        --model xvla --concept-id-dir results/xvla_concept_id \\
        --suites libero_goal libero_object

    # SmolVLA VLM pathway
    GOOGLE_API_KEY=... python action_atlas/generate_feature_descriptions.py \\
        --model smolvla --pathway vlm \\
        --concept-id-dir /data/smolvla_rollouts/concept_id

    # Pi0.5 with Claude
    ANTHROPIC_API_KEY=... python action_atlas/generate_feature_descriptions.py \\
        --model pi05 --pathway expert --llm claude

    # Rule-based only (no API key needed)
    python action_atlas/generate_feature_descriptions.py \\
        --model xvla --concept-id-dir results/xvla_concept_id --llm rules
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tyro

# LLM client imports (optional)
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_GEMINI = True
    except ImportError:
        HAS_GEMINI = False


# Model-specific configuration
MODEL_CONFIGS = {
    "pi05": {
        "full_name": "Pi0.5",
        "n_layers": {"expert": 18, "paligemma": 18},
        "pathways": ["expert", "paligemma"],
        "file_pattern": "pi05_{pathway}_concept_id_layer{layer:02d}_{suite}.json",
        "concept_key": "concepts",
    },
    "xvla": {
        "full_name": "X-VLA",
        "n_layers": {"transformer": 24},
        "pathways": ["transformer"],
        "file_pattern": "xvla_concept_id_layer{layer:02d}_{suite}.json",
        "concept_key": "concepts",
    },
    "oft": {
        "full_name": "OpenVLA-OFT",
        "n_layers": {"single": 32},
        "pathways": ["single"],
        "file_pattern": "oft_concept_id_layer{layer:02d}_{suite}.json",
        "concept_key": "concepts",
    },
    "smolvla": {
        "full_name": "SmolVLA",
        "n_layers": {"expert": 32, "vlm": 32},
        "pathways": ["expert", "vlm"],
        "file_pattern": "smolvla_{pathway}_concept_id_L{layer:02d}_{suite}_{pooling}.json",
        "concept_key": None,  # top-level concepts (no wrapper key)
    },
    "groot": {
        "full_name": "GR00T N1.5",
        "n_layers": {"eagle": 12, "dit": 16, "vlsa": 4},
        "pathways": ["eagle", "dit", "vlsa"],
        "file_pattern": "groot_{pathway}_concept_id_layer{layer:02d}_{suite}.json",
        "concept_key": "concepts",
    },
}

SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]


@dataclass
class DescriptionConfig:
    """Generate SAE feature descriptions using LLM or rules."""

    model: str = "xvla"
    """Model name: pi05, xvla, oft, smolvla, groot"""

    pathway: str = "expert"
    """Model pathway: expert, vlm, paligemma, transformer, eagle, dit, vlsa, single"""

    concept_id_dir: str = ""
    """Directory containing concept ID JSON files."""

    output_dir: str = ""
    """Output directory for descriptions. Default: action_atlas/data/descriptions/{model}/"""

    suites: Tuple[str, ...] = ("libero_goal",)
    """Task suites to process."""

    layers: Optional[List[int]] = None
    """Specific layers. Default: all layers for the pathway."""

    llm: str = "auto"
    """LLM backend: auto (try claude then gemini), claude, gemini, rules"""

    max_features_per_concept: int = 10
    """Max features to describe per concept."""

    batch_size: int = 20
    """Features per API call."""

    max_retries: int = 3

    pooling: str = "mean"
    """Pooling mode for SmolVLA concept ID files: mean or pertoken"""


def init_llm_client(llm_choice: str):
    """Initialize LLM client. Returns (client, llm_type) or (None, 'rules')."""
    if llm_choice == "rules":
        return None, "rules"

    # Try Claude
    if llm_choice in ("auto", "claude") and HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
        try:
            client = anthropic.Anthropic()
            client.messages.create(model="claude-sonnet-4-20250514", max_tokens=10,
                                   messages=[{"role": "user", "content": "test"}])
            print("Using Claude API")
            return client, "claude"
        except Exception as e:
            print(f"Claude init failed: {e}")

    # Try Gemini
    if llm_choice in ("auto", "gemini") and HAS_GEMINI and os.getenv("GOOGLE_API_KEY"):
        try:
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            client.models.generate_content(model="gemini-2.5-flash", contents="test")
            print("Using Gemini API")
            return client, "gemini"
        except Exception as e:
            print(f"Gemini init failed: {e}")

    if llm_choice != "rules":
        print("No LLM API available. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY, or use --llm rules")
        print("Falling back to rule-based descriptions.")
    return None, "rules"


def call_llm(client, llm_type: str, prompt: str) -> str:
    """Call LLM and return response text."""
    if llm_type == "claude":
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    elif llm_type == "gemini":
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return resp.text
    return ""


def load_concept_id_file(filepath: Path, concept_key: Optional[str]) -> Dict:
    """Load concept ID JSON and return {concept_name: [(feature_idx, score, cohens_d), ...]}."""
    data = json.loads(filepath.read_text())

    if concept_key:
        concepts = data.get(concept_key, {})
    else:
        # SmolVLA format: top-level concepts
        concepts = {k: v for k, v in data.items() if isinstance(v, dict) and "top_features" in v}

    result = {}
    for cname, cdata in concepts.items():
        features = []
        top_features = cdata.get("top_features", [])

        for i, feat in enumerate(top_features):
            if isinstance(feat, dict):
                # SmolVLA format: {feature_idx, score, cohens_d}
                features.append((
                    feat["feature_idx"],
                    feat.get("score", 0),
                    feat.get("cohens_d", 0),
                ))
            else:
                # Pi0.5/OFT/X-VLA format: separate lists
                scores = cdata.get("top_scores", [])
                ds = cdata.get("top_cohens_d", [])
                features.append((
                    feat,
                    scores[i] if i < len(scores) else 0,
                    ds[i] if i < len(ds) else 0,
                ))
        result[cname] = features
    return result


def build_feature_associations(concepts: Dict, max_per_concept: int) -> Dict[int, List[Tuple[str, float, float]]]:
    """Build feature_idx -> [(concept_name, score, cohens_d), ...] mapping."""
    associations = {}
    for cname, features in concepts.items():
        for fidx, score, d in features[:max_per_concept]:
            if fidx not in associations:
                associations[fidx] = []
            associations[fidx].append((cname, score, d))
    return associations


def build_prompt(features_batch: List[Tuple[int, List]], model_name: str,
                 pathway: str, layer: int, n_layers: int, suite: str) -> str:
    """Build batched LLM prompt for feature descriptions."""
    lines = [
        f"Analyze these SAE features from a robot manipulation policy.",
        f"",
        f"Model: {model_name}",
        f"Layer: {layer} (of {n_layers} {pathway} layers)",
        f"Suite: {suite}",
        f"",
        f"Concept naming: motion/X = action type, object/X = target object, spatial/X = spatial relation.",
        f"",
        f"For each feature below, write a concise 1-sentence description of what the feature detects.",
        f"Focus on the manipulation behavior (grasping, placing, rotating) rather than abstract statistics.",
        f"",
    ]

    for fidx, assocs in features_batch:
        concept_strs = []
        for cname, score, d in assocs[:3]:
            concept_strs.append(f"{cname} (d={d:.2f})")
        lines.append(f"Feature {fidx}: top concepts = {', '.join(concept_strs)}")

    lines.extend([
        "",
        "Respond with EXACTLY one line per feature in format:",
        "[FEATURE <idx>] <description>",
        "No extra text.",
    ])
    return "\n".join(lines)


def generate_descriptions(cfg: DescriptionConfig):
    """Main description generation pipeline."""
    model_cfg = MODEL_CONFIGS.get(cfg.model)
    if not model_cfg:
        raise ValueError(f"Unknown model: {cfg.model}. Options: {list(MODEL_CONFIGS.keys())}")

    if cfg.pathway not in model_cfg["n_layers"]:
        raise ValueError(f"Unknown pathway '{cfg.pathway}' for {cfg.model}. "
                         f"Options: {model_cfg['pathways']}")

    n_layers = model_cfg["n_layers"][cfg.pathway]
    layers = cfg.layers if cfg.layers else list(range(n_layers))
    concept_id_dir = Path(cfg.concept_id_dir)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path("action_atlas/data/descriptions") / cfg.model
    output_dir.mkdir(parents=True, exist_ok=True)

    client, llm_type = init_llm_client(cfg.llm)
    parse_re = re.compile(r"\[FEATURE\s+(\d+)\]\s*(.*)")

    print(f"Generating descriptions: {model_cfg['full_name']} ({cfg.pathway})")
    print(f"  Concept ID dir: {concept_id_dir}")
    print(f"  Output: {output_dir}")
    print(f"  LLM: {llm_type}")
    print(f"  Suites: {list(cfg.suites)}")
    print(f"  Layers: {layers}")

    for suite in cfg.suites:
        # Find concept ID files for this suite
        pattern = model_cfg["file_pattern"].format(
            pathway=cfg.pathway, layer="*", suite=suite, pooling=cfg.pooling,
        ).replace("*", "*")

        # Try glob with the pattern
        files = sorted(concept_id_dir.glob(f"*{suite}*"))
        if not files:
            # Try with pathway prefix
            files = sorted(concept_id_dir.glob(f"*{cfg.pathway}*{suite}*"))
        if not files:
            print(f"  No concept ID files found for {suite}")
            continue

        for filepath in files:
            # Extract layer number from filename
            layer_match = re.search(r"[Ll](?:ayer)?_?(\d+)", filepath.stem)
            if not layer_match:
                continue
            layer = int(layer_match.group(1))
            if layer not in layers:
                continue

            out_file = output_dir / f"descriptions_{cfg.model}_{cfg.pathway}_layer{layer:02d}_{suite}.json"
            if out_file.exists():
                print(f"  [SKIP] {out_file.name} (exists)")
                continue

            print(f"\n  Layer {layer}, {suite}")
            concepts = load_concept_id_file(filepath, model_cfg["concept_key"])
            associations = build_feature_associations(concepts, cfg.max_features_per_concept)

            if not associations:
                print(f"    No features found")
                continue

            print(f"    {len(associations)} unique features from {len(concepts)} concepts")

            descriptions = {}
            feature_list = list(associations.items())

            for batch_start in range(0, len(feature_list), cfg.batch_size):
                batch = feature_list[batch_start:batch_start + cfg.batch_size]

                if llm_type == "rules":
                    for fidx, assocs in batch:
                        primary = assocs[0][0] if assocs else "unknown"
                        descriptions[str(fidx)] = f"Associated with {primary} (layer {layer})"
                    continue

                prompt = build_prompt(batch, model_cfg["full_name"], cfg.pathway,
                                      layer, n_layers, suite)

                for attempt in range(cfg.max_retries):
                    try:
                        response = call_llm(client, llm_type, prompt)
                        for line in response.strip().split("\n"):
                            m = parse_re.match(line.strip())
                            if m:
                                descriptions[m.group(1)] = m.group(2).strip()
                        break
                    except Exception as e:
                        if attempt < cfg.max_retries - 1:
                            time.sleep(2 ** (attempt + 1))
                        else:
                            print(f"    API failed after {cfg.max_retries} attempts: {e}")
                            for fidx, assocs in batch:
                                if str(fidx) not in descriptions:
                                    primary = assocs[0][0] if assocs else "unknown"
                                    descriptions[str(fidx)] = f"Associated with {primary} (layer {layer})"

            # Fill any remaining gaps with rule-based
            for fidx, assocs in feature_list:
                if str(fidx) not in descriptions:
                    primary = assocs[0][0] if assocs else "unknown"
                    descriptions[str(fidx)] = f"Associated with {primary} (layer {layer})"

            result = {
                "model": cfg.model,
                "pathway": cfg.pathway,
                "layer": layer,
                "suite": suite,
                "n_features": len(descriptions),
                "llm": llm_type,
                "descriptions": descriptions,
            }
            out_file.write_text(json.dumps(result, indent=2))
            print(f"    Saved {len(descriptions)} descriptions to {out_file.name}")

    print(f"\nDone. Descriptions saved to {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(DescriptionConfig)
    generate_descriptions(cfg)
