import logging


def setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG to capture all types of logs

    # Create handlers
    c_handler = logging.StreamHandler()  # Logs to the console
    f_handler = logging.FileHandler('app.log')  # Logs to a file
    c_handler.setLevel(logging.WARNING)  # Console handler for only warnings and errors
    f_handler.setLevel(logging.DEBUG)  # File handler for all debug level messages

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


logger = setup_logger()
