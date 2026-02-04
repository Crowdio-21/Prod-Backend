"""
Global logging configuration for CrowdCompute.

This module provides a centralized logging setup that handles Unicode characters
properly on Windows by using UTF-8 encoding for stream handlers.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure global logging with UTF-8 support for Windows compatibility.

    Args:
        level: The logging level to use (default: logging.INFO)
    """
    # Create a stream handler with UTF-8 encoding
    # This fixes UnicodeEncodeError on Windows when logging special characters
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)

    # Force UTF-8 encoding on Windows
    if sys.platform == "win32":
        import io

        # Wrap stdout with UTF-8 encoding, using 'replace' for errors
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
        # Recreate handler with new stdout
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)

    # Set format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: The name for the logger

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


# Auto-setup logging when this module is imported
setup_logging()
