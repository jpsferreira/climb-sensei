"""Centralized logging configuration."""

import logging.config
import os


def configure_logging():
    """Configure logging based on environment.

    Uses structured format for production, human-readable for development.
    Configurable via LOG_LEVEL and ENVIRONMENT env vars.
    """
    env = os.getenv("ENVIRONMENT", "production").lower()
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
            "detailed": {
                "format": (
                    "%(asctime)s [%(levelname)s] %(name)s "
                    "(%(filename)s:%(lineno)d): %(message)s"
                ),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard" if env == "production" else "detailed",
                "level": level,
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }
    logging.config.dictConfig(config)
