"""Centralized logging configuration for MOTHER."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Custom formatter with colors for console
class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: bool = True,
    console: bool = True,
    name: str = "mother"
) -> logging.Logger:
    """Set up logging for MOTHER.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Whether to log to file
        console: Whether to log to console
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_fmt = ColoredFormatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_filename = LOG_DIR / f"mother_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger


# Pre-configured loggers for different modules
def get_logger(module: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        module: Module name (e.g., 'cli', 'memory', 'tts')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"mother.{module}")


# Initialize default logger on import
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """Get the default MOTHER logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


__all__ = [
    "setup_logging",
    "get_logger",
    "get_default_logger",
    "LOG_DIR",
    "ColoredFormatter",
]

# Convenience aliases
logger = get_default_logger()

