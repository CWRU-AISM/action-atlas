from app import create_app
from flask_cors import CORS
app = create_app()
CORS(app, resources={r"/api/*": {"origins": "*"}})
if __name__ == '__main__':
    app.run(debug=True, port=6006)
    