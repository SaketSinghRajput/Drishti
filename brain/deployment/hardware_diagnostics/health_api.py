from flask import Flask, jsonify, request
import threading
from config.config_loader import ConfigLoader
from .checker import HardwareChecker


def create_health_app(checker: HardwareChecker) -> Flask:
    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health():
        status = checker.get_system_health_status()
        if not status:
            return jsonify({'error': 'No health check run yet'}), 503
        # basic schema validation
        if not isinstance(status, dict) or 'overall_status' not in status or 'components' not in status:
            return jsonify({'error': 'Health status missing required keys'}), 500
        return jsonify(status)

    @app.route('/health/trigger', methods=['POST'])
    def trigger():
        # run in background
        try:
            t = threading.Thread(target=checker.run_all_tests, daemon=True)
            t.start()
            return jsonify({'message': 'Health check triggered'}), 202
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


if __name__ == '__main__':
    cfg = ConfigLoader('config/default_config.yaml')
    checker = HardwareChecker(cfg)
    # run initial check
    try:
        checker.run_all_tests()
    except Exception:
        pass
    app = create_health_app(checker)
    app.run(host='0.0.0.0', port=cfg.get('hardware_diagnostics.health_api.port', 5002))
