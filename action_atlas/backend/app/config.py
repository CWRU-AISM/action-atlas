"""Action Atlas configuration. Model configs, concept definitions, server settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data directories per VLA model
VLA_CONFIGS = {
    "pi05_libero_goal": {
        "name": "Pi0.5 LIBERO Goal",
        "description": "Pi0.5 finetuned on LIBERO Goal Suite (10 tasks)",
        "sae_dir": PROJECT_ROOT / "outputs/pi05_saes/goal",
        "activations_dir": Path("/data/robotsteering/pi05_activations/goal"),
        "analysis_dir": PROJECT_ROOT / "outputs/concept_analysis",
        "viz_data": PROJECT_ROOT / "action_atlas/data/processed",
        "layers": [f"action_expert_layer_{i}" for i in range(18)],
        "hidden_dim": 1024,
        "sae_hidden_mult": 16,
        "n_features": 16384,
        "tasks": {
            0: "open the middle drawer of the cabinet",
            1: "open the top drawer and put the bowl inside",
            2: "push the plate to the front of the stove",
            3: "put the bowl on the plate",
            4: "put the bowl on the stove",
            5: "put the bowl on top of the cabinet",
            6: "put the cream cheese in the bowl",
            7: "put the wine bottle on the rack",
            8: "put the wine bottle on top of the cabinet",
            9: "turn on the stove",
        }
    },
    "pi05_libero_spatial": {
        "name": "Pi0.5 LIBERO Spatial",
        "description": "Pi0.5 finetuned on LIBERO Spatial Suite",
        "sae_dir": PROJECT_ROOT / "outputs/pi05_saes/spatial",
        "activations_dir": Path("/data/robotsteering/pi05_activations/spatial"),
        "analysis_dir": PROJECT_ROOT / "outputs/concept_analysis",
        "viz_data": PROJECT_ROOT / "outputs/viz_data",
        "layers": [f"action_expert_layer_{i}" for i in range(12)],
        "hidden_dim": 1024,
        "sae_hidden_mult": 16,
        "n_features": 16384,
        "tasks": {}
    },
    "openvla_oft": {
        "name": "OpenVLA-OFT",
        "description": "OpenVLA 7B with OFT fine-tuning on LIBERO (L1 regression, 4 suites)",
        "sae_dir": Path("/data/openvla_sae_checkpoints"),
        "activations_dir": Path("/data/openvla_activations"),
        "analysis_dir": PROJECT_ROOT / "results/experiment_results/oft_concept_id",
        "concept_id_dir": PROJECT_ROOT / "results/experiment_results/oft_concept_id",
        "ablation_video_dir": PROJECT_ROOT / "results/experiment_results/oft_concept_ablation/videos",
        "viz_data": PROJECT_ROOT / "outputs/openaction_atlas",
        "layers": [f"layer_{i}" for i in range(32)],
        "hidden_dim": 4096,
        "sae_hidden_mult": 8,
        "n_features": 32768,
        "suites": ["libero_goal", "libero_object", "libero_spatial", "libero_10"],
        "tasks": {
            0: "open the middle drawer of the cabinet",
            1: "open the top drawer and put the bowl inside",
            2: "push the plate to the front of the stove",
            3: "put the bowl on the plate",
            4: "put the bowl on the stove",
            5: "put the bowl on top of the cabinet",
            6: "put the cream cheese in the bowl",
            7: "put the wine bottle on the rack",
            8: "put the wine bottle on top of the cabinet",
            9: "turn on the stove",
        }
    },
    "act_aloha": {
        "name": "ACT-ALOHA",
        "description": "ACT (Action Chunking Transformer) on ALOHA sim tasks (Insertion, TransferCube)",
        "sae_dir": None,
        "activations_dir": None,
        "analysis_dir": PROJECT_ROOT / "results/act_aloha_interp",
        "grid_ablation_dir": PROJECT_ROOT / "results/act_aloha_interp/grid_ablation",
        "injection_dir": PROJECT_ROOT / "results/act_aloha_interp/injection",
        "rollout_dir": Path("/data/robotsteering/aloha_rollouts/act_aloha_interp"),
        "viz_data": None,
        "layers": [],  # ACT does not use per-layer SAEs
        "hidden_dim": 512,
        "sae_hidden_mult": 1,
        "n_features": 0,
        "tasks": {
            "AlohaInsertion-v0": "Bimanual peg insertion task",
            "AlohaTransferCube-v0": "Cube transfer between grippers",
        }
    },
    # X-VLA: 24 TransformerBlocks, Florence-2 VLM, soft-prompted flow-matching
    "xvla": {
        "name": "X-VLA",
        "description": "X-VLA 1B with Florence-2 backbone, 24 TransformerBlocks, flow-matching (LIBERO + SimplerEnv)",
        "sae_dir": Path("/data/batch_1/xvla_saes"),
        "activations_dir": Path("/data/batch_1/xvla_libero"),
        "analysis_dir": Path("/data/batch_1/xvla_concept_id"),
        "concept_id_dir": Path("/data/batch_1/xvla_concept_id"),
        "feature_descriptions_dir": Path("/data/batch_1/xvla_feature_descriptions"),
        "ablation_dir": Path("/data/batch_1/xvla_concept_ablation"),
        "steering_dir": Path("/data/batch_1/xvla_concept_steering"),
        "oracle_probes_dir": Path("/data/batch_1/xvla_matched_oracle_probe"),
        "displacement_analysis": Path("/data/batch_1/XVLA_DISPLACEMENT_ANALYSIS.md"),
        "viz_data": PROJECT_ROOT / "action_atlas/data/processed/xvla",
        "layers": [f"layer_{i}" for i in range(24)],
        "hidden_dim": 1024,
        "sae_hidden_mult": 8,
        "n_features": 8192,
        "suites": ["libero_goal", "libero_object", "libero_spatial", "libero_10"],
        "environments": ["libero", "simplerenv_widowx", "simplerenv_google_robot"],
        "tasks": {
            0: "open the middle drawer of the cabinet",
            1: "open the top drawer and put the bowl inside",
            2: "push the plate to the front of the stove",
            3: "put the bowl on the plate",
            4: "put the bowl on the stove",
            5: "put the bowl on top of the cabinet",
            6: "put the cream cheese in the bowl",
            7: "put the wine bottle on the rack",
            8: "put the wine bottle on top of the cabinet",
            9: "turn on the stove",
        },
        "architecture": {
            "type": "single_pathway",
            "backbone": "Florence-2",
            "action_gen": "flow_matching",
            "params": "1B",
            "layer_types": {"transformer": list(range(24))},
        }
    },
    # SmolVLA: 32 VLM + 32 Expert interleaved, LIBERO + MetaWorld
    "smolvla": {
        "name": "SmolVLA",
        "description": "SmolVLA 450M with interleaved VLM (960-dim) + Expert (480-dim) pathways (LIBERO + MetaWorld)",
        "sae_dir": Path("/data/smolvla_rollouts/sae_models"),
        "activations_dir": Path("/data/smolvla_rollouts/smolvla/activations"),
        "analysis_dir": Path("/data/smolvla_rollouts/concept_id"),
        "concept_id_dir": Path("/data/smolvla_rollouts/concept_id"),
        "ablation_dir": Path("/data/smolvla_rollouts/concept_ablation"),
        "ffn_dir": Path("/data/smolvla_rollouts/ffn_contrastive"),
        "oracle_probes_dir": Path("/data/smolvla_rollouts/oracle_probes"),
        "displacement_analysis": Path("/data/smolvla_rollouts/metaworld_cross_task/SMOLVLA_DISPLACEMENT_ANALYSIS.md"),
        "viz_data": PROJECT_ROOT / "action_atlas/data/processed/smolvla",
        "layers": (
            [f"vlm_layer_{i}" for i in range(32)] +
            [f"expert_layer_{i}" for i in range(32)]
        ),
        "hidden_dim": {"vlm": 960, "expert": 480},
        "sae_hidden_mult": 8,
        "n_features": {"vlm": 7680, "expert": 3840},
        "suites": ["libero_goal", "libero_object", "libero_spatial", "libero_10"],
        "environments": ["libero", "metaworld"],
        "metaworld_difficulties": ["easy", "medium", "hard", "very_hard"],
        "tasks": {
            0: "open the middle drawer of the cabinet",
            1: "open the top drawer and put the bowl inside",
            2: "push the plate to the front of the stove",
            3: "put the bowl on the plate",
            4: "put the bowl on the stove",
            5: "put the bowl on top of the cabinet",
            6: "put the cream cheese in the bowl",
            7: "put the wine bottle on the rack",
            8: "put the wine bottle on top of the cabinet",
            9: "turn on the stove",
        },
        "architecture": {
            "type": "dual_pathway_interleaved",
            "backbone": "SmolVLM",
            "action_gen": "continuous",
            "params": "450M",
            "layer_types": {
                "vlm": list(range(32)),
                "expert": list(range(32)),
            },
        }
    },
    # GR00T N1.5: 16 DiT + 12 Eagle LM + 4 VL-SA, 3B params
    "groot": {
        "name": "GR00T N1.5",
        "description": "GR00T N1.5 3B with DiT (16L) + Eagle LM (12L) + VL-SA (4L) triple-pathway",
        "sae_dir": Path("/data/groot_rollouts/sae_checkpoints_pertoken"),
        "sae_meanpooled_dir": Path("/data/groot_rollouts/sae_checkpoints_meanpooled"),
        "activations_dir": Path("/data/groot_rollouts"),
        "analysis_dir": Path("/data/groot_rollouts/sae_feature_analysis"),
        "concept_id_dir": PROJECT_ROOT / "results/experiment_results/groot_concept_id",
        "ablation_dir": Path("/data/groot_rollouts/sae_feature_ablation"),
        "steering_dir": Path("/data/groot_rollouts_batch2/sae_steering"),
        "fraction_to_failure_dir": Path("/data/groot_rollouts_batch2/sae_fraction_to_failure"),
        "temporal_ablation_dir": Path("/data/groot_rollouts_batch2/sae_temporal_ablation"),
        "cross_suite_dir": Path("/data/groot_rollouts_batch2/sae_cross_suite_ablation"),
        "probing_dir": Path("/data/groot_rollouts/sae_probing"),
        "viz_data": PROJECT_ROOT / "action_atlas/data/processed/groot",
        "layers": (
            [f"dit_layer_{i}" for i in range(16)] +
            [f"eagle_layer_{i}" for i in range(12)] +
            [f"vlsa_layer_{i}" for i in range(4)]
        ),
        "hidden_dim": {"dit": 2048, "eagle": 2048, "vlsa": 2048},
        "sae_hidden_mult": 8,
        "n_features": 16384,
        "suites": ["libero_object", "libero_goal", "libero_long"],
        "tasks": {
            0: "open the middle drawer of the cabinet",
            1: "open the top drawer and put the bowl inside",
            2: "push the plate to the front of the stove",
            3: "put the bowl on the plate",
            4: "put the bowl on the stove",
            5: "put the bowl on top of the cabinet",
            6: "put the cream cheese in the bowl",
            7: "put the wine bottle on the rack",
            8: "put the wine bottle on top of the cabinet",
            9: "turn on the stove",
        },
        "architecture": {
            "type": "triple_pathway",
            "backbone": "Eagle LM + VL-SA",
            "action_gen": "diffusion",
            "params": "3B",
            "layer_types": {
                "dit": list(range(16)),
                "eagle": list(range(12)),
                "vlsa": list(range(4)),
            },
        }
    },
}

# Default model
DEFAULT_VLA_MODEL = "pi05_libero_goal"

# Server config
HOST = "0.0.0.0"
PORT = 6006
DEBUG = True

# Concept definitions (shared across models)
MOTION_CONCEPTS = {
    "put": {"keywords": ["put", "place"], "color": "#FF6B6B"},
    "open": {"keywords": ["open"], "color": "#4ECDC4"},
    "push": {"keywords": ["push"], "color": "#45B7D1"},
    "interact": {"keywords": ["turn", "press", "switch"], "color": "#96CEB4"},
    "pick": {"keywords": ["pick", "grab"], "color": "#DDA0DD"},
    "close": {"keywords": ["close"], "color": "#636E72"},
    "stack": {"keywords": ["stack"], "color": "#E17055"},
    "turn_on": {"keywords": ["turn on"], "color": "#00CEC9"},
    "turn_off": {"keywords": ["turn off"], "color": "#81ECEC"},
}

# Action Phase Concepts - fine-grained manipulation phases
ACTION_PHASE_CONCEPTS = {
    "approach": {"keywords": ["approach", "reach"], "color": "#FF9F43", "description": "Moving toward target object"},
    "grasp": {"keywords": ["grasp", "grip", "grab"], "color": "#EE5A24", "description": "Closing gripper on object"},
    "lift": {"keywords": ["lift", "raise"], "color": "#A3CB38", "description": "Upward motion with object"},
    "transport": {"keywords": ["transport", "carry", "move"], "color": "#1289A7", "description": "Horizontal motion with object"},
    "align": {"keywords": ["align", "position"], "color": "#6C5CE7", "description": "Fine positioning above target"},
    "lower": {"keywords": ["lower", "descend"], "color": "#5758BB", "description": "Downward motion toward placement"},
    "release": {"keywords": ["release", "drop", "let go"], "color": "#D980FA", "description": "Opening gripper to release object"},
    "retract": {"keywords": ["retract", "withdraw"], "color": "#9980FA", "description": "Moving away after release"},
}

OBJECT_CONCEPTS = {
    # Containers
    "bowl": {"keywords": ["bowl"], "color": "#FFEAA7"},
    "plate": {"keywords": ["plate"], "color": "#DFE6E9"},
    "mug": {"keywords": ["mug", "cup"], "color": "#E056FD"},
    "basket": {"keywords": ["basket"], "color": "#F8C291"},
    # "tray" removed - not present in any LIBERO suite's actual ablation data
    # Furniture
    "cabinet": {"keywords": ["cabinet"], "color": "#A29BFE"},
    "drawer": {"keywords": ["drawer"], "color": "#6C5CE7"},
    "shelf": {"keywords": ["shelf"], "color": "#786FA6"},
    "caddy": {"keywords": ["caddy"], "color": "#574B90"},
    # Appliances
    "stove": {"keywords": ["stove", "burner"], "color": "#FD79A8"},
    "microwave": {"keywords": ["microwave"], "color": "#00B894"},
    # Kitchen items
    "frying_pan": {"keywords": ["frying pan", "pan"], "color": "#E17055"},
    "moka_pot": {"keywords": ["moka pot", "coffee"], "color": "#B53471"},
    "wine_bottle": {"keywords": ["wine", "bottle"], "color": "#E17055"},
    "wine_rack": {"keywords": ["rack"], "color": "#00B894"},
    # Food items
    "butter": {"keywords": ["butter"], "color": "#FDCB6E"},
    "cream_cheese": {"keywords": ["cream cheese"], "color": "#F5F6FA"},
    "chocolate_pudding": {"keywords": ["chocolate", "pudding"], "color": "#7B4B36"},
    "ketchup": {"keywords": ["ketchup"], "color": "#EB4D4B"},
    "tomato_sauce": {"keywords": ["tomato sauce"], "color": "#FF6348"},
    "alphabet_soup": {"keywords": ["alphabet", "soup"], "color": "#FFA502"},
    "milk": {"keywords": ["milk"], "color": "#F1F2F6"},
    "orange_juice": {"keywords": ["orange", "juice"], "color": "#FF9F43"},
    "salad_dressing": {"keywords": ["salad", "dressing"], "color": "#20BF6B"},
    # Study items
    "book": {"keywords": ["book"], "color": "#45AAF2"},
}

SPATIAL_CONCEPTS = {
    "on": {"keywords": ["on"], "color": "#74B9FF"},
    "in": {"keywords": ["in", "inside"], "color": "#55EFC4"},
    "top": {"keywords": ["top", "on top"], "color": "#81ECEC"},
    "bottom": {"keywords": ["bottom"], "color": "#636E72"},
    "front": {"keywords": ["front"], "color": "#B2BEC3"},
    "back": {"keywords": ["back", "behind"], "color": "#95A5A6"},
    "left": {"keywords": ["left"], "color": "#48DBFB"},
    "right": {"keywords": ["right"], "color": "#1DD1A1"},
    "middle": {"keywords": ["middle", "center"], "color": "#DFE6E9"},
    "under": {"keywords": ["under", "below"], "color": "#576574"},
}
