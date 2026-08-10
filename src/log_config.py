"""
Logging configuration using loguru.
Provides colored console output and file logging with rotation.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logging(verbose_level: int = 1, log_dir: str = "logs"):
    """
    Configure loguru logging.

    Args:
        verbose_level: 0=WARNING, 1=INFO (default), 2=DEBUG
        log_dir: Directory to store log files
    """
    # Remove default handler
    logger.remove()

    # Define log levels based on verbose_level
    if verbose_level >= 2:
        console_level = "DEBUG"
    elif verbose_level >= 1:
        console_level = "INFO"
    else:
        console_level = "WARNING"

    # Console output with colors
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=console_level,
        colorize=True,
    )

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # File output with rotation (10MB, keep 3 files)
    logger.add(
        log_path / "hw2_{time:YYYY-MM-DD}.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        level="DEBUG",  # Always log DEBUG to file
        rotation="10 MB",
        retention=3,
        compression="gz",
    )

    logger.debug(f"Logging configured: console={console_level}, file=DEBUG")

    return logger


def get_logger(name: str):
    """Get a logger instance with a specific name."""
    return logger.bind(name=name)