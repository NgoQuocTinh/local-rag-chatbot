"""
Centralized logging configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from config.setting import get_settings

def setup_logger(
    name:  str,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with consistent configuration
    
    Args: 
        name: Logger name (usually __name__)
        level: Log level override
    
    Returns:
        Configured logger
    """
    settings = get_settings()
    handlers = settings.logging.handlers
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    log_level = level or settings.logging.level
    logger.setLevel(getattr(logging, log_level))
    
    formatter = logging.Formatter(settings.logging.format)
    
    # Console handler
    if handlers.console.enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(
            getattr(logging, handlers.console.level)
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
        # File handler
    if handlers.file.enabled:
        log_file = Path(settings.paths.log_file)
        log_file.parent.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(
            getattr(logging, handlers.file.level)
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger