from abc import ABC, abstractmethod
from typing import Callable
import shutil
from rich.console import Console

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import torch
from torch import nn
# import matplotlib.pyplot as plt
# import matplotlib.cm as mplcm
# import matplotlib as mpl
from tqdm import tqdm
from enum import Enum
import gc
import random

from dynuq.models.diffusion import Diffusion

class LossWrapperConfig():
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossWrapper(ABC):
    def __init__(
        self,
        cfg: LossWrapperConfig = LossWrapperConfig(),
    ):  
        # init console
        self._init_console()
        
        # config
        self.config = cfg
        
        # device
        self.device = self.config.device
        
    def _init_console(self):
        """
        Console for logging/printing
        """
        # terminal width
        terminal_width = shutil.get_terminal_size((80, 20)).columns

        # rich console
        self.console = Console(width=terminal_width)

    @abstractmethod
    def compute_loss(
        self,
        diffusion: Diffusion,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        rel_contact_emphasis: float = None,  
        conf_acc_function: Callable = None,
    ) -> torch.Tensor:
        pass