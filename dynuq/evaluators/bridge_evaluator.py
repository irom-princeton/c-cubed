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
import warnings
from tqdm import tqdm
from enum import Enum
import gc
import random
from PIL import Image
import torch.nn.functional as F
import imageio
import fire
from natsort import natsorted

from dynuq.models.model import DiT
from dynuq.models.diffusion import Diffusion

from dynuq.evaluators.base_evaluator import BaseEvaluatorConfig, BaseEvaluator



class BridgeEvaluatorConfig(BaseEvaluatorConfig):
    # option to enable verbose print
    verbose_print: bool = False
    
    # base output path
    base_output_path: Path | str = "outputs/eval/"
    

class BridgeEvaluator(BaseEvaluator):
    def __init__(
        self,
        cfg: BridgeEvaluatorConfig = BridgeEvaluatorConfig(),
    ):
        # init super-class
        super().__init__(cfg=cfg)
        
        # TODO: Add some other functionality
        
    def _load_actions(self, action_path: str | Path, action_dim: int) -> torch.Tensor:
        """Load actions from file.

        Returns:
            Tensor of shape (action_length, action_dim)
        """
        action_path = Path(action_path)
        if not action_path.exists():
            raise FileNotFoundError(f"Action file not found: {action_path}")

        self.console.log(f"Loading actions from: {action_path}")

        if action_path.suffix == ".npy":
            actions = torch.from_numpy(np.load(action_path)).float()
        elif action_path.suffix == ".json":
            with open(action_path, 'r') as f:
                data = json.load(f)
            actions = np.array(data['action'])
        elif action_path.suffix == ".pt":
            actions = torch.load(action_path)
        else:
            raise ValueError(f"Unsupported action file format: {action_path.suffix}. Use .npy or .pt")

        if actions.shape[1] != action_dim:
            raise ValueError(
                f"Action dimension ({actions.shape[1]}) doesn't match expected ({action_dim})"
            )

        return actions
        
    def evaluate(
        self,
        cfg_path: str | Path = None,
    ) -> torch.Tensor:
        """Generate video from first frame and actions using trained world model.

            Args:
                checkpoint_path: Path to model checkpoint (.pt file)
                config_path: Path to model config YAML file (same as used for training)
                input_video: Path to input video (first frame will be extracted)
                action_path: Path to action sequence file (.npy or .pt)
                output_path: Path to save generated video (default: generated_video.mp4)
                groundtruth_path: Optional path to groundtruth video for side-by-side comparison
                n_context_frames: Number of context frames (usually 1)
                window_len: Window length for autoregressive generation (None = no windowing)
                horizon: Horizon for generation (1 = frame-by-frame)
                fps: Frames per second for output video
                use_ema: Use EMA model if available (recommended)

            Example:
                python inference.py \\
                    --checkpoint_path=outputs/20240101_120000/ckpt_000100000.pt \\
                    --config_path=configs/train_bridge_512.yaml \\
                    --input_video=data/bridge_video.mp4 \\
                    --action_path=data/actions.npy \\
                    --output_path=output.mp4
        """
            
        assert torch.cuda.is_available(), "CUDA device required for inference"

        # load config
        cfg = self._load_config(cfg_path)
        
        # model config path
        mod_cfg_path = cfg.model.get("config_path", None)
        
        # load model config
        mod_cfg = self._load_config(mod_cfg_path)
        
        # model config
        self.console.log(
            f"Model config: input_h={mod_cfg.input_h}, input_w={mod_cfg.input_w},\
                patch_size={mod_cfg.patch_size}, dim={mod_cfg.patch_size},\
                    layers={mod_cfg.layers}, {mod_cfg.heads}, heads={mod_cfg.model_dim}",
        )
            

        # create output directory, if needed
        base_output_path = Path(cfg.output_path)
        base_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load VAE
        if mod_cfg.vae_type.lower() == "sd":
            from dynuq.tokenizers.vae import StableDiffusionVAE
            vae = StableDiffusionVAE().to(self.device)
            self.console.log("Using StableDiffusionVAE")
        elif mod_cfg.vae_type.lower() == "cosmos":
            from dynuq.tokenizers.vae import CosmosVAE
            vae = CosmosVAE().to(self.device)
            self.console.log("Using CosmosVAE")
        else:
            raise ValueError(f"Unknown VAE type: {mod_cfg.vae_type}. Must be 'sd' or 'cosmos'")

        # Get latent channels from VAE
        if hasattr(vae, 'vae') and hasattr(vae.vae, 'config'):
            latent_channels = vae.vae.config.latent_channels
        else:
            latent_channels = 16

        # Initialize model
        model = DiT(
            in_channels=latent_channels,
            patch_size=mod_cfg.patch_size,
            dim=mod_cfg.model_dim,
            num_layers=mod_cfg.layers,
            num_heads=mod_cfg.heads,
            action_dim=mod_cfg.action_dim,
            max_frames=mod_cfg.n_frames,
            enable_confidence_pred=mod_cfg.enable_confidence_prediction,
        ).to(self.device)

        # Initialize diffusion
        diffusion = Diffusion(
            timesteps=mod_cfg.timesteps,
            sampling_timesteps=mod_cfg.sampling_timesteps,
            enable_confidence_pred=mod_cfg.enable_confidence_prediction,
            device=self.device,
            ref_latent_dir=mod_cfg.ref_latent_dir,
        ).to(self.device)

        # Load checkpoint
        checkpoint_data = self._load_checkpoint(cfg.checkpoint_path, self.device)

        # Load model weights (use EMA if available and requested)
        if cfg.use_ema and "ema" in checkpoint_data:
            self.console.log("Using EMA model weights")
            model.load_state_dict(checkpoint_data["ema"])
        else:
            self.console.log("Using regular model weights")
            model.load_state_dict(checkpoint_data["model"])

        model.eval()
        
        
        # TODO: temporary split
        split_idx = os.environ.get("BRIDGE_SPLIT_IDX", None)
        
        if split_idx is None:
            split_idx = 0
        else:
            split_idx = int(split_idx)
        
        # TODO: temporary split size
        split_size = os.environ.get("BRIDGE_SPLIT_SIZE", None)
        
        if split_size is None:
            split_size = 10
        else:
            split_size = int(split_size)
        
        self.console.log(f"Using split index: {split_idx}")
        
        # all eval videos
        eval_videos = natsorted(list(Path(cfg.input_video).iterdir())) # TODO: natsorted
        eval_videos = eval_videos[split_idx * split_size:(split_idx + 1) * split_size]

        # # TODO: fix input H and W 
        # mod_cfg.input_h = 384
        # mod_cfg.input_w = 384
        
        if cfg.get("err_threshold_conf_pred_min", None) is not None:
            # get error threshold for confidence prediction
            err_threshold_conf_pred_min = cfg.get("err_threshold_conf_pred_min", None)
            err_threshold_conf_pred_max = cfg.get("err_threshold_conf_pred_max", None)
            num_err_threshold_conf_pred_steps = cfg.get("num_err_threshold_conf_pred_steps", None)

            # error thresholds for generation
            err_threshold_conf_pred_all = np.linspace(
                err_threshold_conf_pred_min, 
                err_threshold_conf_pred_max,
                num=num_err_threshold_conf_pred_steps,
            )
        
        for err_threshold_conf_pred in err_threshold_conf_pred_all:
        
            for in_vid in tqdm(eval_videos, desc="Running eval on the dataset"):
                # full path to input video
                input_video_path = in_vid / "rgb.mp4"
                
                # full path to action file
                action_path = Path(cfg.action_path) / f"{in_vid.stem}.json"

                # output video path
                output_path = base_output_path / f"{in_vid.stem}" / f"{in_vid.stem}_generated.mp4"
                
                # create output directory, if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Load first frame from input video
                input_frames = self._load_first_frame_from_video(input_video_path, mod_cfg.input_h, mod_cfg.input_w)

                # Add batch dimension: (1, n_context_frames, H, W, 3)
                input_frames = input_frames.unsqueeze(0).to(self.device)

                # load ground-truth video
                gt_video_frames = self._load_video(input_video_path, mod_cfg.input_h, mod_cfg.input_w)
                
                # Encode input frames to latent space
                latent_gt_video_frames = vae.encode(gt_video_frames)

                # Load actions
                actions = self._load_actions(action_path, mod_cfg.action_dim)
                actions = torch.tensor(actions).float()
                num_frames = actions.shape[0]
                actions = actions.unsqueeze(0).to(self.device)  # Add batch dimension
                
                # # get error threshold for confidence prediction
                # err_threshold_conf_pred = cfg.get("err_threshold_conf_pred", None)

                # Generate video
                self.console.log(f"Generating {num_frames} frames...")
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # Encode input frames to latent space
                        input_latent = vae.encode(input_frames)

                        # Generate latent frames
                        gen_outputs = diffusion.generate(
                            model,
                            input_latent,
                            actions,
                            n_context_frames=cfg.n_context_frames,
                            n_frames=num_frames,
                            window_len=cfg.window_len,
                            horizon=cfg.horizon,
                            err_threshold_conf_pred=err_threshold_conf_pred, #TODO: never pass in None for inference
                        )
                        
                        # TODO: Save generated latents and raw_conf_pred from dict
                        # unpack outputs
                        generated_latent = gen_outputs["x_pred"]                        # predicted RGB latents
                        pred_conf_generated = gen_outputs["x_conf_pred"]                # predicted confidence in RGB latent space
                        pred_conf_composited = gen_outputs["conf_pred_composited"]      # composited predicted confidence
                        raw_conf_pred = gen_outputs["raw_conf_pred"]                    # raw predicted confidence

                        # Decode to pixel space
                        generated_frames = vae.decode(generated_latent)

                        # TODO: decode confidence predictions (revisit this)
                        if pred_conf_generated is not None:
                            pred_conf_frames = vae.decode(pred_conf_generated)
                            pred_conf_composited_frames = vae.decode(pred_conf_composited)
                        else:
                            pred_conf_frames = None
                            pred_conf_composited_frames = None

                # Post-process and save
                self.console.log(f"Saving generated video to {output_path}")
                video_np = (generated_frames[0].float().clamp(0, 1) * 255).byte().cpu().numpy()

                # Ensure correct shape for imageio: (T, H, W, C)
                if video_np.shape[0] == 3:  # If (C, H, W, T)
                    video_np = video_np.transpose(1, 2, 3, 0)
                elif len(video_np.shape) == 4 and video_np.shape[1] == 3:  # If (T, C, H, W)
                    video_np = video_np.transpose(0, 2, 3, 1)

                # Save as mp4
                gen_output_path = output_path.parent / f"conf_thresh_{err_threshold_conf_pred}.mp4"
                    
                imageio.mimsave(gen_output_path, video_np, fps=cfg.fps, codec='libx264')
                self.console.log(f"Done! Generated video saved to {gen_output_path}")

                # save the groundtruth latents
                gt_latent_output_path = Path(output_path).parent / f"gt_latents_conf_thresh_{err_threshold_conf_pred}.npy"
                gt_latent_np = latent_gt_video_frames[0].float().cpu().numpy()
                np.save(gt_latent_output_path, gt_latent_np)
                self.console.log(f"Saved groundtruth latents to {gt_latent_output_path}")

                # save the generated latents
                generated_latent_output_path = Path(output_path).parent / f"generated_latents_conf_thresh_{err_threshold_conf_pred}.npy"
                gen_latent_np = generated_latent[0].float().cpu().numpy()
                np.save(generated_latent_output_path, gen_latent_np)
                self.console.log(f"Saved generated latents to {generated_latent_output_path}")

                # save the predicted confidence
                if pred_conf_frames is not None:
                    conf_video_output_path = Path(output_path).parent / f"pred_conf_gen_video_conf_thresh_{err_threshold_conf_pred}.mp4"
                    self.console.log(f"Saving predicted confidence to {conf_video_output_path}")
                    conf_video_np = (pred_conf_frames[0].float().clamp(0, 1) * 255).byte().cpu().numpy()

                    # Ensure correct shape for imageio: (T, H, W, C)
                    if conf_video_np.shape[0] == 3:  # If (C, H, W, T)
                        conf_video_np = conf_video_np.transpose(1, 2, 3, 0)
                    elif len(conf_video_np.shape) == 4 and conf_video_np.shape[1] == 3:  # If (T, C, H, W)
                        conf_video_np = conf_video_np.transpose(0, 2, 3, 1)

                    # Save as mp4
                    imageio.mimsave(conf_video_output_path, conf_video_np, fps=cfg.fps, codec='libx264')
                    self.console.log(f"Done! Generated video saved to {conf_video_output_path}")

                    # save the composited RGB-pred confidence frames
                    comp_conf_video_output_path = Path(output_path).parent / f"composited_pred_conf_gen_video_conf_thresh_{err_threshold_conf_pred}.mp4"
                    self.console.log(f"Saving composited predicted confidence to {comp_conf_video_output_path}")
                    comp_conf_video_np = (pred_conf_composited_frames[0].float().clamp(0, 1) * 255).byte().cpu().numpy()

                    # Ensure correct shape for imageio: (T, H, W, C)
                    if comp_conf_video_np.shape[0] == 3:  # If (C, H, W, T)
                        comp_conf_video_np = comp_conf_video_np.transpose(1, 2, 3, 0)
                    elif len(comp_conf_video_np.shape) == 4 and comp_conf_video_np.shape[1] == 3:  # If (T, C, H, W)
                        comp_conf_video_np = comp_conf_video_np.transpose(0, 2, 3, 1)

                    # Save as mp4
                    imageio.mimsave(comp_conf_video_output_path, comp_conf_video_np, fps=cfg.fps, codec='libx264')
                    self.console.log(f"Done! Generated video saved to {comp_conf_video_output_path}")

                    # save the raw confidence predictions as a numpy file
                    raw_conf_output_path = Path(output_path).parent / f"raw_conf_pred_conf_thresh_{err_threshold_conf_pred}.npy"
                    raw_conf_np = raw_conf_pred[0].float().cpu().numpy()
                    np.save(raw_conf_output_path, raw_conf_np)
                    self.console.log(f"Saved raw confidence predictions to {raw_conf_output_path}")

                # Print stats
                self.console.log("Video stats:")
                self.console.log(f"  Shape: {video_np.shape}")
                self.console.log(f"  Frames: {video_np.shape[0]}")
                self.console.log(f"  FPS: {cfg.fps}")
                self.console.log(f"  Duration: {video_np.shape[0] / cfg.fps:.2f} seconds")

                # Create side-by-side comparison if groundtruth is provided
                if cfg.groundtruth_path is not None:
                    groundtruth_path = Path(cfg.groundtruth_path)
                    if not cfg.groundtruth_path.exists():
                        warnings.warn(f"Groundtruth video not found: {groundtruth_path}. Skipping comparison.")
                    else:
                        self.console.log("Creating side-by-side comparison with groundtruth...")
                        comparison_path = output_path.parent / f"{output_path.stem}_comparison{output_path.suffix}"

                        # Load groundtruth video
                        gt_reader = imageio.get_reader(groundtruth_path)
                        gt_frames = []
                        for i, frame in enumerate(gt_reader):
                            if i >= len(video_np):
                                break
                            # Resize groundtruth to match generated video size
                            gt_frame = np.array(Image.fromarray(frame).resize((video_np.shape[2], video_np.shape[1])))
                            gt_frames.append(gt_frame)
                        gt_reader.close()

                        # Pad if groundtruth is shorter
                        while len(gt_frames) < len(video_np):
                            gt_frames.append(gt_frames[-1] if gt_frames else np.zeros_like(video_np[0]))

                        gt_frames = np.array(gt_frames[:len(video_np)])

                        # Create side-by-side frames
                        comparison_frames = []
                        for gen_frame, gt_frame in zip(video_np, gt_frames):
                            # Concatenate horizontally: [groundtruth | generated]
                            combined = np.concatenate([gt_frame, gen_frame], axis=1)
                            comparison_frames.append(combined)

                        comparison_frames = np.array(comparison_frames)
                        imageio.mimsave(comparison_path, comparison_frames, fps=cfg.fps, codec='libx264')
                        self.console.log(f"Comparison video saved to {comparison_path}")


def _run_example(cfg_path: str | Path = None):
    # example usage
    evaluator = BridgeEvaluator()
    evaluator.evaluate(cfg_path=cfg_path)

if __name__ == "__main__":
    # example usage
    fire.Fire(_run_example)