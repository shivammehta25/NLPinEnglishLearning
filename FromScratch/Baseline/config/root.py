"""
Application level configurations
"""

import logging
import random


SEED = 1234
random.seed(SEED)

LOGGING_LEVEL = logging.DEBUG
LOGGING_FORMAT = (
    "[%(levelname)s | %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)
