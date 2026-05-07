#!/usr/bin/env python3
# Action Atlas Backend Server. Run with: python run.py [--port PORT]

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import send_from_directory
from app import create_app
from app.config import HOST, PORT, DEBUG

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
PROJECT_ROOT = Path(__file__).parent.parent.parent
ABLATION_VIDEOS_DIR = PROJECT_ROOT / "ablation_videos"
OFT_ABLATION_VIDEOS_DIR = PROJECT_ROOT / "results" / "experiment_results" / "oft_concept_ablation" / "videos"
ACT_ROLLOUT_DIR = Path("/data/robotsteering/aloha_rollouts/act_aloha_interp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT, help="Port to run on")
    args = parser.parse_args()

    app = create_app()

    @app.route("/ablation_videos/<path:filename>")
    def serve_ablation_video(filename):
        return send_from_directory(ABLATION_VIDEOS_DIR, filename)

    @app.route("/oft_ablation_videos/<path:filename>")
    def serve_oft_ablation_video_static(filename):
        return send_from_directory(OFT_ABLATION_VIDEOS_DIR, filename)

    @app.route("/act_videos/<path:filename>")
    def serve_act_video(filename):
        return send_from_directory(ACT_ROLLOUT_DIR, filename)

    @app.route("/")
    def serve_frontend():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.route("/<path:filename>")
    def serve_static(filename):
        return send_from_directory(FRONTEND_DIR, filename)

    print(f"\nAction Atlas running on http://{HOST}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser\n")
    app.run(host=HOST, port=args.port, debug=DEBUG)
