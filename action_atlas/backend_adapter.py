"""
Action Atlas API.

Re-exports the combined Flask blueprint for callers that import
``vla_bp`` from this module.
"""

from flask import Blueprint

vla_bp = Blueprint("vla_compat", __name__)


def register_vla_routes(app):
    # Register all Action Atlas API routes on the Flask app.
    from .api import register_all_routes
    register_all_routes(app)
