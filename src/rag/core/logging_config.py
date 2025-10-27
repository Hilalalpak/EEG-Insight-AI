"""
Simple, one-time logging setup (LoggingConfig.setup_logging())
"""

import logging
import sys
from infrastructure.conf.interfaces import LoggingConfigInterface

class LoggingConfig:

    _logging_configured = False

    @staticmethod
    def setup_logging(config: LoggingConfigInterface):
        """
        Sets logging. It should only be called once.
        """
        if LoggingConfig._logging_configured:
            return

        level_str = config.get_log_level()
        format_str = config.get_log_format()
        datefmt_str = config.get_log_datefmt()

        log_level = getattr(logging, level_str.upper(), logging.INFO)

        logging.basicConfig(
            level=log_level,
            format=format_str,
            datefmt=datefmt_str,
            stream=sys.stdout,
            force=True)

        # Log that the configuration is complete.
        logging.getLogger(__name__).info(f"Logger configured (level={level_str.upper()})")
        LoggingConfig._logging_configured = True