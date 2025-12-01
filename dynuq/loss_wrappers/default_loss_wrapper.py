from typing import Callable
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
import copy
# import matplotlib.pyplot as plt
# import matplotlib.cm as mplcm
# import matplotlib as mpl
from tqdm import tqdm
from enum import Enum
import gc
import random
# import pandas as pd
# from PIL import Image
import torch.nn.functional as F
# from hydra.utils import instantiate

from dynuq.models.diffusion import Diffusion

# metrics
# from pytorch_msssim import SSIM
# from torchmetrics.image import PeakSignalNoiseRatio
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from dynuq.loss_wrappers.base_loss_wrapper import LossWrapperConfig, LossWrapper


class DefaultLossWrapperConfig(LossWrapperConfig):
    # option to enable verbose print
    verbose_print: bool = True
    
    # base output path
    base_output_path: Path | str = "outputs/eval/"
    

class DefaultLossWrapper(LossWrapper):
    def __init__(
        self,
        cfg: DefaultLossWrapperConfig = DefaultLossWrapperConfig(),
    ):
        # init super-class
        super().__init__(cfg=cfg)
        
        # TODO: Add some other functionality
        
    def compute_loss(
        self,
        diffusion: Diffusion,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        rel_contact_emphasis: float = None,
        conf_acc_function: Callable = None,
    ) -> torch.Tensor:
        """
        Computes an unweighted diffusion loss
        """
        # compute the loss function
        loss = diffusion.loss_fn(model, x, actions, weight=None, conf_acc_function=conf_acc_function)
        
        return loss
    