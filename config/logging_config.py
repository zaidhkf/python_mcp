"""Application diagnostics shared by Flask and the LangChain agent."""
import logging
from logging.handlers import RotatingFileHandler

from config.settings import LOG_DIR

LOGGER_NAME = 'mcp_analytics'


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [request=%(request_id)s] %(message)s',
            defaults={'request_id': '-'},
        )
        console = logging.StreamHandler()
        file = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=1_000_000, backupCount=3,
                                   encoding='utf-8')
        for handler in (console, file):
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger
