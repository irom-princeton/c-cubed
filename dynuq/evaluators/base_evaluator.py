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
from torch import nn
from torchvision import transforms
from pytorchvideo.data.encoded_video import EncodedVideo
import einops
from tqdm import tqdm
from enum import Enum
import gc
import random
import warnings

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    warnings.warn("omegaconf not available. Install with: pip install omegaconf")


class BaseEvaluatorConfig():
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # eval configuration
    eval_config: Dict[str, Any] | OmegaConf = None


class BaseEvaluator(ABC):
    def __init__(
        self,
        cfg: BaseEvaluatorConfig = BaseEvaluatorConfig(),
        cfg_path: str | Path = None,
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
        
    def _load_config(self, config_path: str | Path):
        """Load configuration from YAML file using OmegaConf."""
        if not OMEGACONF_AVAILABLE:
            raise ImportError("omegaconf is required. Install with: pip install omegaconf")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        cfg = OmegaConf.load(config_path)
        self.console.log("Loaded config from %s", config_path)
        return cfg
    
    def _load_checkpoint(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> dict:
        """Load model checkpoint from path."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.console.log("Loading checkpoint from %s", checkpoint_path)
        data = torch.load(checkpoint_path, map_location=device)
        return data
    
    def _load_first_frame_from_video(
        self, 
        video_path: str | Path, 
        input_h: int, 
        input_w: int,
    ) -> torch.Tensor:
        """Load and preprocess first frame from video.

        Returns:
            Tensor of shape (H, W, 3) with values in [0, 1]
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.console.log("Extracting first frame from video: %s", video_path)
        
        video = EncodedVideo.from_path(video_path, decode_audio=False)
        
        # Get first frame
        start_sec = 0.0
        end_sec = 0.1  # Small duration to ensure we get at least one frame
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
        clip = einops.rearrange(clip, "c t h w -> t h w c")

        transform = transforms.Resize((int(input_h), int(input_w)))

        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = transform(clip)
        clip = einops.rearrange(clip, "t c h w -> t h w c")

        return clip
    
    def _load_video(
        self,
        video_path, 
        input_h: int, 
        input_w: int,
        map_to_float: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        # video
        video = mp.read_video(video_path)  # T x H x W x C

        # map to tensor 
        video = torch.tensor(video, device=device).float()  # T x H x W x C
        
        # map to [0, 1] range
        if map_to_float:
            video /= 255.0
            
        # apply video (image) transforms
        transform = transforms.Resize((int(input_h), int(input_w)))

        # reshape to (T x C x H x W)
        video = video.permute(0, 3, 1, 2)
        
        # apply transform
        video = transform(video)
        
        # reshape from to (T x C x H x W) to (B x T x H x W x C), move channel dimension
        video = video.permute(0, 2, 3, 1)[None, ...]
        
        return video

    @abstractmethod
    def evaluate(
        self,
        cfg_path: str | Path = None, 
    ) -> torch.Tensor:
        pass
        