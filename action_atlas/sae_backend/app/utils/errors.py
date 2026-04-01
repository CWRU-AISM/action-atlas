from flask import jsonify

class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code

def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = {
            'status': error.status_code,
            'error': {
                'code': error.code,
                'message': error.message
            }
        }
        return jsonify(response), error.status_code

    @app.errorhandler(404)
    def handle_404(error):
        return jsonify({
            'status': 404,
            'error': {
                'code': 'NOT_FOUND',
                'message': 'NOT_FOUND'
            }
        }), 404

    @app.errorhandler(500)
    def handle_500(error):
        return jsonify({
            'status': 500,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': 'INTERNAL_ERROR'
            }
        }), 500