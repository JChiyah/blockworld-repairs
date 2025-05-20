"""
Author: JChiyah

This module is the core of the blockworld-repairs project.

Any scripts that need this module should import it and set up the logging as follows:

import bwcore
bwcore.configure_logging(bwcore.log_utils.INFO)
logger = bwcore.get_logger()
"""

import os

# Get the directory where the main.py file is located
__script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the project root directory (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__script_dir))

if not PROJECT_ROOT.endswith('blockworld-repairs'):
	print(f"Something is wrong with the project root: {PROJECT_ROOT}, fix manually in src/bwcore/__init__.py")
	exit()


from .utils import log_utils
from .utils.log_utils import get_logger, _configure_logging


def configure_logging(logger_level: int = log_utils.INFO):
	_configure_logging(logger_level)
	logger = get_logger()
	logger.info(f"bwcore initialised with PROJECT_ROOT: '{PROJECT_ROOT}'")


from . import modelling, evaluation, data, parsing
