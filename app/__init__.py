from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

from app.modules.cache import cache
from app.modules.cache import check_and_clear_cache
from config import DevelopmentConfig, ProductionConfig
from .routes import init_app_routes


def create_app():
    app = Flask(__name__)

    env = app.config.get('ENV', 'development')

    if env == 'production':
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=cache.clear_cache, trigger='interval', hours=24)  # Clears cache every 24 hours
        scheduler.start()
        app.config.from_object(ProductionConfig)
    else:
        check_and_clear_cache()
        app.config.from_object(DevelopmentConfig)

    # Initialize the routes
    init_app_routes(app)

    return app