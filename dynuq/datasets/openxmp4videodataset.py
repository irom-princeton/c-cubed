from pathlib import Path
from typing import Sequence
from tqdm import tqdm
import os
import json

import numpy as np
import torch
import einops
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset
from torchvision import transforms


# Some of the code was built off open-source code in https://github.com/world-model-eval/world-model-eval


class OpenXMP4VideoDataset(Dataset):
    def __init__(
        self,
        dataset_paths: Sequence[str | Path] | str | Path,
        input_h: int,
        input_w: int,
        n_frames: int,
        *,
        frame_skip: int = 1,
        action_dim: int = 10,
        split: str = "train",
        max_videos: int | None = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")

        if isinstance(dataset_paths, str):
            dataset_paths = dataset_paths.split(",")
        self.dataset_paths = list(dataset_paths)

        self.n_frames = int(n_frames)
        self.frame_skip = int(frame_skip)
        self.clip_len = self.n_frames * self.frame_skip
        self.action_dim = int(action_dim)

        self.transform = transforms.Resize((int(input_h), int(input_w)))

        self.video_paths: list[Path] = []
        self.video_lengths: list[int] = []
        self.action_types: list[str] = []
        self.action_paths: list[Path] = []
    
        for dataset_dir in self.dataset_paths:
            dataset_dir = Path(dataset_dir)
            # get name of dataset name from lowest level directory
            dataset_name = dataset_dir.name
            
            if dataset_name == "bridge":
                video_dir_path = dataset_dir / "videos" / split
                annotation_dir_path = dataset_dir / "annotation" / split
                
                # iterate through all subdirectory names of video_dir_path
                trajectory_names = [
                    os.path.splitext(f)[0]
                    for f in os.listdir(annotation_dir_path)
                    if f.endswith(".json")
                ]
                
                if max_videos is not None:
                    trajectory_names = trajectory_names[:max_videos]
                for trajectory_name in trajectory_names:
                    mp4 = video_dir_path / trajectory_name / "rgb.mp4"
                    action_path = annotation_dir_path / f"{trajectory_name}.json"
                    if not action_path.exists() or not mp4.exists():
                        continue
                    try:
                        with open(action_path, 'r') as f:
                            data = json.load(f)
                        length = int(np.array(data['action']).shape[0])
                    except Exception:
                        continue
                    if length >= self.clip_len:
                        self.video_paths.append(mp4)
                        self.video_lengths.append(length)
                        self.action_types.append("json")
                        self.action_paths.append(action_path)
                

        if not self.video_paths:
            raise RuntimeError(f"No valid videos found in {self.save_dir} for subsets {self.subset_names}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        video_path = self.video_paths[idx]
        action_path = self.action_paths[idx]
        action_type = self.action_types[idx]
        length = self.video_lengths[idx]
        
        start = np.random.randint(0, length - self.clip_len + 1)

        video = EncodedVideo.from_path(video_path, decode_audio=False)
        fps = video._container.streams.video[0].guessed_rate
        start_sec = start / fps
        end_sec = (start + self.clip_len) / fps
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
        clip = einops.rearrange(clip, "c t h w -> t h w c")
        
        if action_type == "npz":
            actions = np.load(action_path)["arr_0"][start : start + self.clip_len]
        elif action_type == "json":
            with open(action_path, 'r') as f:
                data = json.load(f)
            actions = np.array(data['action'])[start : start + self.clip_len]
            
        assert actions.shape[1] == self.action_dim, f"Unexpected action dim: {actions.shape[1]} != {self.action_dim}"
            
        clip = clip[:: self.frame_skip]
        actions = actions[:: self.frame_skip]
        assert len(clip) == self.n_frames
        assert len(actions) == self.n_frames

        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)
        clip = einops.rearrange(clip, "t c h w -> t h w c")
        actions = torch.from_numpy(actions).float()
        return idx, clip, actions
