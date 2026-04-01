#!/usr/bin/env python3
"""
Concept definitions for VLA interpretability experiments.

Defines concept-to-task mappings for:
- LIBERO suites (goal, object, spatial, libero_10)
- SimplerEnv (WidowX, Google Robot)

Task IDs follow alphabetical ordering of task descriptions (matching
SUITE_TASK_MAPPINGS in concept_extraction_unified.py).

Imported by:
- experiments/xvla_contrastive_concept_id.py
- experiments/pi05_contrastive_concept_id.py
"""

from typing import Dict, List, Tuple

# LIBERO GOAL SUITE (task IDs: alphabetical order of task descriptions)
# 0: open the middle drawer of the cabinet
# 1: open the top drawer and put the bowl inside
# 2: push the plate to the front of the stove
# 3: put the bowl on the plate
# 4: put the bowl on the stove
# 5: put the bowl on top of the cabinet
# 6: put the cream cheese in the bowl
# 7: put the wine bottle on the rack
# 8: put the wine bottle on top of the cabinet
# 9: turn on the stove

MOTION_CONCEPTS = {
    "put": {"tasks": [1, 3, 4, 5, 6, 7, 8]},
    "open": {"tasks": [0, 1]},
    "push": {"tasks": [2]},
    "interact": {"tasks": [9]},
}

OBJECT_CONCEPTS = {
    "bowl": {"tasks": [1, 3, 4, 5, 6]},
    "plate": {"tasks": [2, 3]},
    "stove": {"tasks": [4, 9]},
    "cabinet": {"tasks": [0, 5, 8]},
    "drawer": {"tasks": [0, 1]},
    "wine_bottle": {"tasks": [7, 8]},
    "cream_cheese": {"tasks": [6]},
    "rack": {"tasks": [7]},
}

SPATIAL_CONCEPTS = {
    "on": {"tasks": [3, 4, 7, 8]},
    "in": {"tasks": [1, 6]},
    "top": {"tasks": [1, 5, 8]},
    "front": {"tasks": [2]},
    "middle": {"tasks": [0]},
}


# LIBERO OBJECT SUITE
# 0: pick up the alphabet soup and place it in the basket
# 1: pick up the bbq sauce and place it in the basket
# 2: pick up the butter and place it in the basket
# 3: pick up the chocolate pudding and place it in the basket
# 4: pick up the cream cheese and place it in the basket
# 5: pick up the ketchup and place it in the basket
# 6: pick up the milk and place it in the basket
# 7: pick up the orange juice and place it in the basket
# 8: pick up the salad dressing and place it in the basket
# 9: pick up the tomato sauce and place it in the basket

OBJECT_MOTION_CONCEPTS = {
    "pick": {"tasks": list(range(10))},
    "place": {"tasks": list(range(10))},
}

OBJECT_OBJECT_CONCEPTS = {
    "alphabet_soup": {"tasks": [0]},
    "bbq_sauce": {"tasks": [1]},
    "butter": {"tasks": [2]},
    "chocolate_pudding": {"tasks": [3]},
    "cream_cheese": {"tasks": [4]},
    "ketchup": {"tasks": [5]},
    "milk": {"tasks": [6]},
    "orange_juice": {"tasks": [7]},
    "salad_dressing": {"tasks": [8]},
    "tomato_sauce": {"tasks": [9]},
    "basket": {"tasks": list(range(10))},
}

OBJECT_SPATIAL_CONCEPTS = {
    "in": {"tasks": list(range(10))},
}


# LIBERO SPATIAL SUITE
# 0: pick up the black bowl between the plate and the ramekin and place it on the plate
# 1: pick up the black bowl from table center and place it on the plate
# 2: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
# 3: pick up the black bowl next to the cookie box and place it on the plate
# 4: pick up the black bowl next to the plate and place it on the plate
# 5: pick up the black bowl next to the ramekin and place it on the plate
# 6: pick up the black bowl on the cookie box and place it on the plate
# 7: pick up the black bowl on the ramekin and place it on the plate
# 8: pick up the black bowl on the stove and place it on the plate
# 9: pick up the black bowl on the wooden cabinet and place it on the plate

SPATIAL_MOTION_CONCEPTS = {
    "pick": {"tasks": list(range(10))},
    "place": {"tasks": list(range(10))},
}

SPATIAL_OBJECT_CONCEPTS = {
    "bowl": {"tasks": list(range(10))},
    "plate": {"tasks": list(range(10))},
    "ramekin": {"tasks": [0, 5, 7]},
    "cookie_box": {"tasks": [3, 6]},
    "stove": {"tasks": [8]},
    "cabinet": {"tasks": [2, 9]},
    "drawer": {"tasks": [2]},
}

SPATIAL_SPATIAL_CONCEPTS = {
    "between": {"tasks": [0]},
    "center": {"tasks": [1]},
    "in_drawer": {"tasks": [2]},
    "next_to": {"tasks": [3, 4, 5]},
    "on": {"tasks": [6, 7, 8, 9]},
}


# LIBERO 10 SUITE
# 0: pick up the book and place it in the back compartment of the caddy
# 1: put both moka pots on the stove
# 2: put both the alphabet soup and the cream cheese box in the basket
# 3: put both the alphabet soup and the tomato sauce in the basket
# 4: put both the cream cheese box and the butter in the basket
# 5: put the black bowl in the bottom drawer of the cabinet and close it
# 6: put the white mug on the left plate and put the yellow and white mug on the right plate
# 7: put the white mug on the plate and put the chocolate pudding to the right of the plate
# 8: put the yellow and white mug in the microwave and close it
# 9: turn on the stove and put the moka pot on it

LIBERO_10_MOTION_CONCEPTS = {
    "pick": {"tasks": [0]},
    "put": {"tasks": [1, 2, 3, 4, 5, 6, 7, 8]},
    "close": {"tasks": [5, 8]},
    "turn_on": {"tasks": [9]},
}

LIBERO_10_OBJECT_CONCEPTS = {
    "book": {"tasks": [0]},
    "caddy": {"tasks": [0]},
    "moka_pot": {"tasks": [1, 9]},
    "stove": {"tasks": [1, 9]},
    "alphabet_soup": {"tasks": [2, 3]},
    "cream_cheese": {"tasks": [2, 4]},
    "tomato_sauce": {"tasks": [3]},
    "butter": {"tasks": [4]},
    "basket": {"tasks": [2, 3, 4]},
    "bowl": {"tasks": [5]},
    "drawer": {"tasks": [5]},
    "cabinet": {"tasks": [5]},
    "mug": {"tasks": [6, 7, 8]},
    "plate": {"tasks": [6, 7]},
    "pudding": {"tasks": [7]},
    "microwave": {"tasks": [8]},
}

LIBERO_10_SPATIAL_CONCEPTS = {
    "on": {"tasks": [1, 6, 7, 9]},
    "in": {"tasks": [0, 2, 3, 4, 5, 8]},
    "left": {"tasks": [6]},
    "right": {"tasks": [6, 7]},
    "bottom": {"tasks": [5]},
}


# SIMPLERENV WIDOWX CONCEPTS
# Task IDs (alphabetical order):
# 0: widowx_carrot_on_plate
# 1: widowx_put_eggplant_in_basket
# 2: widowx_spoon_on_towel
# 3: widowx_stack_cube

WIDOWX_MOTION_CONCEPTS = {
    "pick": {"tasks": [0, 1, 2, 3]},
    "place": {"tasks": [0, 1, 2, 3]},
    "stack": {"tasks": [3]},
}

WIDOWX_OBJECT_CONCEPTS = {
    "carrot": {"tasks": [0]},
    "eggplant": {"tasks": [1]},
    "spoon": {"tasks": [2]},
    "cube": {"tasks": [3]},
    "plate": {"tasks": [0]},
    "basket": {"tasks": [1]},
    "towel": {"tasks": [2]},
}

WIDOWX_SPATIAL_CONCEPTS = {
    "on": {"tasks": [0, 2]},
    "in": {"tasks": [1]},
    "stack": {"tasks": [3]},
}


# SIMPLERENV GOOGLE ROBOT CONCEPTS
# Task IDs (alphabetical order):
# 0: google_robot_close_middle_drawer
# 1: google_robot_close_top_drawer
# 2: google_robot_move_near
# 3: google_robot_open_middle_drawer
# 4: google_robot_open_top_drawer
# 5: google_robot_pick_coke_can

GOOGLE_ROBOT_MOTION_CONCEPTS = {
    "open": {"tasks": [3, 4]},
    "close": {"tasks": [0, 1]},
    "move": {"tasks": [2]},
    "pick": {"tasks": [5]},
}

GOOGLE_ROBOT_OBJECT_CONCEPTS = {
    "drawer": {"tasks": [0, 1, 3, 4]},
    "coke_can": {"tasks": [5]},
}

GOOGLE_ROBOT_SPATIAL_CONCEPTS = {
    "middle": {"tasks": [0, 3]},
    "top": {"tasks": [1, 4]},
    "near": {"tasks": [2]},
}


# LOOKUP HELPERS

# All LIBERO concept dicts by suite
ALL_CONCEPTS = {
    "goal": {
        "motion": MOTION_CONCEPTS,
        "object": OBJECT_CONCEPTS,
        "spatial": SPATIAL_CONCEPTS,
    },
    "libero_goal": {
        "motion": MOTION_CONCEPTS,
        "object": OBJECT_CONCEPTS,
        "spatial": SPATIAL_CONCEPTS,
    },
    "object": {
        "motion": OBJECT_MOTION_CONCEPTS,
        "object": OBJECT_OBJECT_CONCEPTS,
        "spatial": OBJECT_SPATIAL_CONCEPTS,
    },
    "libero_object": {
        "motion": OBJECT_MOTION_CONCEPTS,
        "object": OBJECT_OBJECT_CONCEPTS,
        "spatial": OBJECT_SPATIAL_CONCEPTS,
    },
    "spatial": {
        "motion": SPATIAL_MOTION_CONCEPTS,
        "object": SPATIAL_OBJECT_CONCEPTS,
        "spatial": SPATIAL_SPATIAL_CONCEPTS,
    },
    "libero_spatial": {
        "motion": SPATIAL_MOTION_CONCEPTS,
        "object": SPATIAL_OBJECT_CONCEPTS,
        "spatial": SPATIAL_SPATIAL_CONCEPTS,
    },
    "10": {
        "motion": LIBERO_10_MOTION_CONCEPTS,
        "object": LIBERO_10_OBJECT_CONCEPTS,
        "spatial": LIBERO_10_SPATIAL_CONCEPTS,
    },
    "libero_10": {
        "motion": LIBERO_10_MOTION_CONCEPTS,
        "object": LIBERO_10_OBJECT_CONCEPTS,
        "spatial": LIBERO_10_SPATIAL_CONCEPTS,
    },
    "widowx": {
        "motion": WIDOWX_MOTION_CONCEPTS,
        "object": WIDOWX_OBJECT_CONCEPTS,
        "spatial": WIDOWX_SPATIAL_CONCEPTS,
    },
    "google_robot": {
        "motion": GOOGLE_ROBOT_MOTION_CONCEPTS,
        "object": GOOGLE_ROBOT_OBJECT_CONCEPTS,
        "spatial": GOOGLE_ROBOT_SPATIAL_CONCEPTS,
    },
}


def get_concept_task_mapping(suite: str) -> Dict[str, Dict[str, Dict]]:
    """Get all concepts organized by type for a given suite.

    Returns: {concept_type: {concept_name: {"tasks": [task_ids]}}}
    """
    return ALL_CONCEPTS.get(suite, {})


def get_all_concept_names(suite: str) -> List[Tuple[str, str]]:
    """Get all (concept_type, concept_name) pairs for a suite."""
    result = []
    for ctype, concepts in ALL_CONCEPTS.get(suite, {}).items():
        for cname in concepts:
            result.append((ctype, cname))
    return result
