from flask import Flask
from app.routes import input_bp , explore_bp, validate_bp
from app.utils.errors import register_error_handlers
import sys
from pathlib import Path

# Add parent directory to path for backend_adapter import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend_adapter import vla_bp

def create_app():
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(input_bp)
    app.register_blueprint(explore_bp)
    app.register_blueprint(validate_bp)

    # Register VLA routes
    app.register_blueprint(vla_bp)

    # Register error handlers
    register_error_handlers(app)

    return app