import random
import os

import numpy as np
import torch


def seed_everything(seed: int = 1333):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = False
    os.environ["PYTHONHASHSEED"] = str(seed)
