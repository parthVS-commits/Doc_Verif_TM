import logging
import sys
from config.settings import Config

def setup_logger(name='document_validation', log_level=None):
    """
    Configure and return a logger with console and file handlers
    
    Args:
        name (str): Logger name
        log_level (str, optional): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    # Determine log level
    log_level = log_level or Config.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to prevent duplicate logs
    logger.handlers.clear()
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # File Handler
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logger()

def log_error(message, exc_info=None):
    """
    Log an error message with optional exception info
    
    Args:
        message (str): Error message
        exc_info (Exception, optional): Exception information
    """
    logger.error(message, exc_info=exc_info)

def log_info(message):
    """
    Log an informational message
    
    Args:
        message (str): Informational message
    """
    logger.info(message)

def log_warning(message):
    """
    Log a warning message
    
    Args:
        message (str): Warning message
    """
    logger.warning(message)