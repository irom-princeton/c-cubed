import fire
from pathlib import Path
import imageio
import os
import datetime
import logging
from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.distributed as dist

try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    logging.warning("omegaconf not available. Install with: pip install omegaconf")


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    # https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/train.py#L40
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Set the requires_grad flag for all parameters of ``model``."""
    for p in model.parameters():
        p.requires_grad = flag


def collate_with_indices(batch):
    """Custom collate function that preserves sample indices."""
    indices = [item[0] for item in batch]
    videos = torch.stack([item[1] for item in batch])
    actions = torch.stack([item[2] for item in batch])
    return indices, videos, actions


def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility, with different seeds per rank."""
    import random
    import numpy as np

    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed if available.

    Returns a tuple of (local_rank, global_rank, world_size, is_distributed).
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, world_size, True
    return 0, 0, 1, False

