from abc import ABC, abstractmethod
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
import mediapy as mp
import torch
import warnings

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    warnings.warn("omegaconf not available. Install with: pip install omegaconf")



class BaseMetricConfig():
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # eval configuration
    eval_config: Dict[str, Any] | OmegaConf = None
    
    # max workers
    max_workers: int = os.cpu_count()


class BaseMetric(ABC):
    def __init__(
        self,
        cfg: BaseMetricConfig = BaseMetricConfig(),
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
    def compute(self, preds, targets):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def save_metrics_to_file(
        self,
        metrics: dict,
        save_path: Path | str,
    ):
        
        # create parent directory
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # save to file
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Metrics saved to {save_path}")
              
    def load_video(
        self,
        video_path, 
        map_to_float: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        # video
        video = mp.read_video(video_path)  # T x H x W x C

        # map to tensor and reshape to B x C x T x H x W
        video = torch.tensor(video, device=device).float().permute(3, 0, 1, 2)[None, ...]
        
        # map to [0, 1] range
        if map_to_float:
            video /= 255.0
        
        return video