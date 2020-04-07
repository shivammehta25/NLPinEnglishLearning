"""
Application level configurations
"""
import os
import logging
import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOGGING_LEVEL = logging.DEBUG
LOGGING_FORMAT = (
    "[%(levelname)s | %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)

SEED = 1234


def seed_all(seed):
    """Seed the results for duplication"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


TRAINED_CLASSIFIER_FOLDER = "trained"
TRAINED_CLASSIFIER_RNNHIDDEN = "RNNHidden.pt"
