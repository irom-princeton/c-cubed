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


class CalibrationLossWrapperConfig(LossWrapperConfig):
    # threshold for measuring error
    err_threshold: float = 1e-1
    
    # option to enable verbose print
    verbose_print: bool = False
    
    # base output path
    base_output_path: Path | str = "outputs/eval/"
     

class CalibrationLossWrapper(LossWrapper):
    def __init__(
        self,
        cfg: CalibrationLossWrapperConfig = CalibrationLossWrapperConfig(),
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
        Computes loss
        """
        pass
    
    def compute_accuracy(
        self,
        pred: Any,
        target: Any,
        err_threshold: torch.tensor = None,
    ) -> float:
        """
        Computes accuracy given predicted and target values
        Args:
        pred: predicted values
        target: target values
        err_threshold: error threshold
        """
        # input-checking
        if err_threshold is None:
            err_threshold = self.config.err_threshold
            
            # map to the same shape as the target values
            err_threshold = err_threshold * torch.ones_like(target)

        if len(err_threshold.shape) == 2:
            # reshape from (B, T) to (B, T, H, W, C)
            err_threshold = err_threshold[..., None, None, None]

        # error
        error = torch.abs(pred - target)
        
        # accuracy
        acc = (error <= err_threshold).float()
        
        return acc
    