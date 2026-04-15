"""Shim: re-exports from mother.core.logging_config for backward compatibility."""
from mother.core.logging_config import *  # noqa: F401,F403
from mother.core.logging_config import (  # explicit for IDE
    setup_logging, get_logger, get_default_logger, LOG_DIR, ColoredFormatter, logger,
)
