from app import create_app
from app.modules.logger import logger

app = create_app()

if __name__ == '__main__':
    logger.debug("Starting the application")
    app.run(host='0.0.0.0', port=3000, debug=app.config['DEBUG'])