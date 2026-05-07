# Action Atlas API - split into focused modules


def register_all_routes(app):
    # Register all API route blueprints
    from .models import models_bp
    from .features import features_bp
    from .search import search_bp
    from .concepts import concepts_bp
    from .experiments import experiments_bp
    from .videos import videos_bp
    from .interventions import interventions_bp
    from .injection import injection_bp
    from .scene_state import scene_state_bp
    from .perturbation import perturbation_bp

    for bp in [models_bp, features_bp, search_bp, concepts_bp, experiments_bp,
               videos_bp, interventions_bp, injection_bp, scene_state_bp,
               perturbation_bp]:
        app.register_blueprint(bp)
