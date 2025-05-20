# log_utils.py
#
# Author: Javier CG

import sys
import logging
from logging import DEBUG, INFO
from typing import Optional, Dict

import colorlog

# Create a logger with colorlog
_logger = colorlog.getLogger('bwcore')

# log_format = '%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - [%(name)s:%(filename)s:%(lineno)d] %(log_color)s%(message)s%(reset)s'
_LOG_FORMAT = '%(asctime)s - %(levelname)-8s - [%(name)s:%(filename)s:%(lineno)d] %(message)s'
_LOG_FORMAT_COLOUR = '%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - [%(name)s:%(filename)s:%(lineno)d] %(log_color)s%(message)s%(reset)s'

_DEFAULT_LOG_COLORS = {
	'DEBUG': 'cyan',
	'INFO': 'light_white',
	'WARNING': 'yellow',
	'ERROR': 'red',
	'CRITICAL': 'bold_red,bg_white',
}


# Create a handler with a colored log format
# log_format = '%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - [%(name)s:%(filename)s:%(lineno)d] %(log_color)s%(message)s%(reset)s'
# formatter = colorlog.ColoredFormatter(
# 	log_format,
# 	datefmt='%Y-%m-%d %H:%M:%S',
# 	reset=True,
# 	log_colors=_DEFAULT_LOG_COLORS
# )

# Function to adjust the log level of a logger by name
def set_logger_level(logger_name, log_level):
	logger_to_configure = logging.getLogger(logger_name)
	logger_to_configure.setLevel(log_level)


# Finally set up loggers
# only set up loggers once, so output does not repeat
# if not len(_logger.handlers):
# 	# Create a StreamHandler and set the formatter
# 	ch = logging.StreamHandler()
# 	ch.setFormatter(formatter)
# 	_logger.addHandler(ch)


_setup_logger = False

# Define default logging configuration, hiding messages from some libs
DEFAULT_LOG_CONFIG = {
	'PIL.PngImagePlugin': logging.WARNING,
	'urllib3': logging.WARNING,
	'matplotlib': logging.WARNING,
	'httpcore.http11': logging.WARNING,
	'openai._base_client': logging.WARNING
}


def _configure_logging(
	logger_level: int = INFO,
	log_file: Optional[str] = None,
	module_levels: Optional[Dict[str, int]] = None
) -> logging.Logger:
	"""Configure the root logger with color formatting and optional file output.

	Args:
		logger_level: Base logging level for the root logger
		log_file: Optional file path to write logs to
		module_levels: Optional dict of logger name to level mappings
		log_colors: Optional dict to override default color scheme

	Returns:
		The configured root logger
	"""
	global _setup_logger

	if _setup_logger:
		return _logger

	# Configure color formatter
	formatter = colorlog.ColoredFormatter(
		_LOG_FORMAT_COLOUR,
		datefmt='%Y-%m-%d %H:%M:%S',
		reset=True,
		log_colors=_DEFAULT_LOG_COLORS
	)

	# Console handler
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	_logger.addHandler(console_handler)

	# Optional file handler
	if log_file:
		_add_file_handler(log_file)

	# Set module-specific log levels
	levels = DEFAULT_LOG_CONFIG.copy()
	if module_levels:
		levels.update(module_levels)

	for module, level in levels.items():
		set_logger_level(module, level)

	_logger.setLevel(logger_level)
	_setup_logger = True
	_logger.info(f"Logger started (level={logging.getLevelName(_logger.getEffectiveLevel())})")

	return _logger


def _add_file_handler(log_file: str) -> logging.FileHandler:
	# Create file handler with same format as console
	global _logger
	file_handler = logging.FileHandler(log_file)
	file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
	_logger.addHandler(file_handler)


def configure_file_logging(log_file: str) -> None:
	"""Add a file handler to the main logger to record all logs to a file.

	Args:
		log_file: Path to the file where logs will be written

	Raises:
		RuntimeError: If the logger has not been configured yet
	"""
	if not _setup_logger:
		raise RuntimeError("Logger must be configured first. Call configure_logging() before adding a file handler.")

	_add_file_handler(log_file)


def get_logger() -> logging.Logger:
	"""Get or create the configured logger instance.

	Args:
		logger_level: Logging level if logger needs to be configured

	Returns:
		The configured logger instance
	"""
	# if not _setup_logger:
	# 	return configure_logging(logger_level)
	return _logger
