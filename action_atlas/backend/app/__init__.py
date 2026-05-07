# Action Atlas Backend
import sys
from pathlib import Path
from flask import Flask
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .routes import explore_bp, ablation_bp


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register API modules
    from action_atlas.api import register_all_routes
    register_all_routes(app)

    # Register additional route blueprints
    app.register_blueprint(explore_bp)
    app.register_blueprint(ablation_bp)

    @app.route('/api/health')
    def health():
        return {"status": "ok", "service": "Action Atlas"}

    return app
