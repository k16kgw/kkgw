import os
import random

import numpy as np

def seed_setting(seed: int):
    """ 複数シードをまとめて設定する """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)