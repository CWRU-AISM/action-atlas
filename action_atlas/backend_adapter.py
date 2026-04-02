"""
Action Atlas API - backward compatibility shim.

The API has been split into focused modules under action_atlas/api/.
This file re-exports the combined blueprint for existing code that
imports `from backend_adapter import vla_bp`.

Modules:
  api/helpers.py       - Config, data loading, shared constants
  api/models.py        - Model list, suites, layers
  api/features.py      - SAE scatter, feature detail, search
  api/concepts.py      - Concept lists, ablation results
  api/experiments.py   - Experiment results, findings, metrics
  api/videos.py        - Video serving and indices
  api/interventions.py - Grid ablation, counterfactual, injection
  api/scene_state.py   - Scene state, trajectories
  api/perturbation.py  - Vision perturbation results
"""

from flask import Blueprint

vla_bp = Blueprint("vla_compat", __name__)


def register_vla_routes(app):
    """Register all Action Atlas API routes."""
    from .api import register_all_routes
    register_all_routes(app)
