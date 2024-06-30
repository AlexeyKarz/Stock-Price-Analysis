import logging
from logging.handlers import RotatingFileHandler


def setup_logger():
    # Create a custom logs
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG to capture all types of logs

    # Create handlers
    c_handler = logging.StreamHandler()  # Logs to the console
    c_handler.setLevel(logging.WARNING)  # Console handler for only warnings and errors
    # Set up rotating log files which can grow up to 5MB and keep 3 backups
    handler = RotatingFileHandler('logs/app.log', maxBytes=5 * 1024 * 1024, backupCount=3)

    # Create formatters and add it to handlers
    h_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(h_format)
    c_handler.setFormatter(c_format)

    # Add handlers to the logs
    logger.addHandler(c_handler)
    logger.addHandler(handler)

    return logger


logger = setup_logger()
