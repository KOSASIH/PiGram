import logging
import os

# Logging configuration
LOGGING_LEVEL = "DEBUG"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create a custom logger
logger = logging.getLogger("app_logger")

# Set the logging level
logger.setLevel(LOGGING_LEVEL)

# Create a file handler
log_file = os.path.join(os.getcwd(), "app.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(LOGGING_LEVEL)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOGGING_LEVEL)

# Create a formatter and attach it to the handlers
formatter = logging.Formatter(LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger():
    return logger
