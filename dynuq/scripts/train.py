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
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from dynuq.datasets.openxmp4videodataset import OpenXMP4VideoDataset
from dynuq.models.model import DiT
from dynuq.tokenizers.vae import VAE
from dynuq.models.diffusion import Diffusion

from dynuq.loss_wrappers.default_loss_wrapper import DefaultLossWrapper
from dynuq.loss_wrappers.calibration_loss_wrapper import CalibrationLossWrapper
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available. Install with: pip install wandb")

try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    logging.warning("omegaconf not available. Install with: pip install omegaconf")


# Some of the code was built off open-source code in https://github.com/world-model-eval/world-model-eval


@dataclass
class TrainConfig:
    """Training configuration."""
    # Dataset
    # dataset_dir: str = "sample_data"
    dataset_paths: list[str] = field(default_factory=list)
    input_h: int = 256
    input_w: int = 256
    n_frames: int = 10
    frame_skip: int = 1
    # subset_names: str = "bridge"
    action_dim: int = 10
    num_workers: int = 16

    # Training
    batch_size: int = 4
    timesteps: int = 1000
    lr: float = 8e-5
    ema_decay: float = 0.999
    max_train_steps: int = 500_000
    seed: int = 42
    enable_confidence_prediction: bool = False

    # Debug
    log_sample_indices: bool = False  # Log sample indices to verify shuffling

    # Architecture
    vae_type: str = "sd"  # "sd" or "cosmos"
    patch_size: int = 2
    model_dim: int = 1024
    layers: int = 16
    heads: int = 16

    # Logging
    checkpoint_dir: Optional[str] = None
    save_iter: int = 10_000
    # TODO: Update
    validate_every: int = 1  # 1_000 #  20_000
    log_every: int = 100
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    save_best_val_ckpt: bool = True

    # Sampling
    sampling_timesteps: int = 10
    window_len: Optional[int] = None
    horizon: int = 1
    
    # confidence prediction
    ref_latent_dir: Optional[str] = None
    # default error threshold for accuracy function in confidence prediction
    conf_err_threshold: float = 1e-1

from dynuq.scripts.utils.train_utils import (
    # load_config,
    update_ema,
    requires_grad,
    collate_with_indices,
    set_seed,
    init_distributed,
)


def load_config(config_path: Optional[str | Path]) -> DictConfig:
    """Load configuration from YAML file using OmegaConf."""
    if not OMEGACONF_AVAILABLE:
        raise ImportError("omegaconf is required. Install with: pip install omegaconf")

    # Create base config from dataclass
    base_cfg = OmegaConf.structured(TrainConfig)

    # Load and merge file config if provided
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        file_cfg = OmegaConf.load(config_path)
        base_cfg = OmegaConf.merge(base_cfg, file_cfg)

    return base_cfg


def main(config: Optional[str | Path] = None, **kwargs) -> None:
    """Main training function.

    Args:
        config: Path to YAML config file
        **kwargs: Any config parameter to override (e.g., batch_size=8, lr=1e-4)
    """
    assert torch.cuda.is_available(), "CUDA device required for training"

    local_rank, rank, world_size, distributed = init_distributed()

    # Load config from file (or use defaults)
    cfg = load_config(config)
    
    config_name = Path(config).stem

    # Override with any CLI arguments
    if kwargs:
        cli_overrides = OmegaConf.create(kwargs)
        cfg = OmegaConf.merge(cfg, cli_overrides)

    if config and rank == 0:
        logging.info("Loaded config from %s", config)
    if kwargs and rank == 0:
        logging.info("CLI overrides: %s", kwargs)

    # Set random seed (different per rank for proper distributed training)
    set_seed(cfg.seed, rank)

    device = f"cuda:{local_rank}" if distributed else "cuda"
    device = torch.device(device)
    
    train_dataset = OpenXMP4VideoDataset(
        dataset_paths=cfg.dataset_paths,
        # save_dir=Path(cfg.dataset_dir),
        input_h=cfg.input_h,
        input_w=cfg.input_w,
        n_frames=cfg.n_frames,
        frame_skip=cfg.frame_skip,
        action_dim=cfg.action_dim,
        # subset_names=cfg.subset_names,
        split="train",
    )
    val_dataset = OpenXMP4VideoDataset(
        dataset_paths=cfg.dataset_paths,
        # save_dir=Path(cfg.dataset_dir),
        input_h=cfg.input_h,
        input_w=cfg.input_w,
        n_frames=cfg.n_frames,
        frame_skip=cfg.frame_skip,
        action_dim=cfg.action_dim,
        # subset_names=cfg.subset_names,
        split="test",
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_with_indices,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_with_indices,
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    
    logging.info("Setting loss")
    
    # initializer loss wrapper
    loss_wrapper = DefaultLossWrapper()
    logging.info("Using Default loss")
        
    if cfg.enable_confidence_prediction:
        logging.info("Enabling confidence prediction head in the model and loss wrapper")
        
        # instantiate the loss wrapper with confidence prediction enabled
        conf_head_loss_wrapper = CalibrationLossWrapper()
        
        # TODO: Refactor
        # update the default error threshold for accuracy function in confidence prediction
        conf_head_loss_wrapper.config.err_threshold = cfg.conf_err_threshold
        
        logging.info(f"Using a default error threshold for acc fn in confidence prediction: {conf_head_loss_wrapper.config.err_threshold}")
        
        # get the accuracy function for confidence prediction
        conf_head_acc_function = lambda pred, target, err_threshold: conf_head_loss_wrapper.compute_accuracy(pred=pred, target=target, err_threshold=err_threshold)
    else:
        conf_head_acc_function = None
    
        
    # Initialize VAE based on type
    if cfg.vae_type.lower() == "sd":
        from dynuq.tokenizers.vae import StableDiffusionVAE
        vae = StableDiffusionVAE().to(device)
        if rank == 0:
            logging.info("Using StableDiffusionVAE")
    elif cfg.vae_type.lower() == "cosmos":
        from dynuq.tokenizers.vae import CosmosVAE
        vae = CosmosVAE().to(device)
        if rank == 0:
            logging.info("Using CosmosVAE with temporal_window=%d", cfg.vae_temporal_window)
    else:
        raise ValueError(f"Unknown VAE type: {cfg.vae_type}. Must be 'sd' or 'cosmos'")

    # Get latent channels from VAE
    if hasattr(vae, 'vae') and hasattr(vae.vae, 'config'):
        latent_channels = vae.vae.config.latent_channels
    else:
        # Default to 16 for CosmosVAE
        latent_channels = 16

    model = DiT(
        in_channels=latent_channels,
        patch_size=cfg.patch_size,
        dim=cfg.model_dim,
        num_layers=cfg.layers,
        num_heads=cfg.heads,
        action_dim=cfg.action_dim,
        max_frames=cfg.n_frames,
        enable_confidence_pred=cfg.enable_confidence_prediction,
    ).to(device)

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        sampling_timesteps=cfg.sampling_timesteps,
        enable_confidence_pred=cfg.enable_confidence_prediction,
        device=device,
        ref_latent_dir=cfg.ref_latent_dir,
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model_no_ddp = model.module
    else:
        model_no_ddp = model

    # Exponential Moving Average of model parameters
    ema = deepcopy(model_no_ddp).to(device)
    requires_grad(ema, False)
    update_ema(ema, model_no_ddp, cfg.ema_decay)

    # Scale learning rate by number of GPUs for distributed training
    effective_lr = cfg.lr * world_size
    optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.02, betas=(0.9, 0.99))
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # TODO: Revisit and confirm validity
    # TODO: Update max number fo steps for sceheduler (cfg.max_train_steps)
    # use annealing (with min learning rate: 1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=15000, eta_min=1e-6)
    
    # append config name to run_name if available
    run_name += f"_{config_name}"

    if cfg.checkpoint_dir is None:
        checkpoint_dir = Path("outputs") / run_name
        logging.info(
            "No checkpoint_dir specified, using autogenerated directory %s",
            checkpoint_dir,
        )
    else:
        checkpoint_dir = Path(cfg.checkpoint_dir) / run_name
        logging.info("Using provided checkpoint_dir %s", checkpoint_dir)

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        if WANDB_AVAILABLE and cfg.wandb_project is not None:
            # modify the runname to indicate the loss function
            wandb_run_name = cfg.wandb_run_name or checkpoint_dir.name
            
            if cfg.enable_confidence_prediction:
                wandb_run_name = wandb_run_name + f"_conf_pred_{conf_head_loss_wrapper.config.err_threshold}"
            
            # init wandb
            wandb.init(
                project=cfg.wandb_project,
                name=wandb_run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(checkpoint_dir),
            )
            logging.info(f"Initialized wandb project: {cfg.wandb_project} with run {wandb_run_name}")

    if rank == 0:
        # save the config file
        cfg_output_path = Path(checkpoint_dir) / "config.yaml"
        OmegaConf.save(cfg, f=cfg_output_path)
    
    # Synchronize all processes before loading checkpoint
    if distributed:
        dist.barrier()

    # resume from latest checkpoint if available
    ckpts = sorted(checkpoint_dir.glob("ckpt_*.pt"))
    train_steps = 0
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
        data = torch.load(latest, map_location=device)
        state_dict = data["model"]
        model_no_ddp.load_state_dict(state_dict)
        optimizer.load_state_dict(data["optimizer"])
        if "ema" in data:
            ema.load_state_dict(data["ema"])
        else:
            update_ema(ema, model_no_ddp, 0.0)
        train_steps = int(data.get("step", 0))
        logging.info("Loaded checkpoint %s (step %d)", latest, train_steps)

    # Synchronize after loading checkpoint
    if distributed:
        dist.barrier()

    running_loss = torch.tensor(0.0)
    num_batches = 0
    loss_history: list[torch.Tensor] = []
    mse_history: list[torch.Tensor] = []
    min_val_error_stats: dict = {"error": np.inf, "train_steps": -1}
    
    epoch = 0
    batches_in_epoch = 0
    pbar = tqdm(total=cfg.max_train_steps, desc="Training") if rank == 0 else None
    if pbar is not None:
        pbar.n = train_steps
        pbar.refresh()
    while train_steps < cfg.max_train_steps:
        try:
            indices, x, actions = next(train_iter)
            batches_in_epoch += 1
        except StopIteration:
            # New epoch: update sampler and recreate iterator
            epoch += 1
            batches_in_epoch = 0
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            indices, x, actions = next(train_iter)
            batches_in_epoch += 1

        # Log sample indices for first 3 batches of each epoch to verify shuffling
        if cfg.log_sample_indices and batches_in_epoch <= 3:
            logging.info(f"[Rank {rank}] Epoch {epoch}, Batch {batches_in_epoch}: sample_indices={indices}")

        x = x.to(device)
        actions = actions.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = vae.encode(x)
            # # original
            # loss = diffusion.loss_fn(model, x, actions)
            
            # compute loss
            loss = loss_wrapper.compute_loss(
                diffusion=diffusion,
                model=model,
                x=x,
                actions=actions,
                conf_acc_function=conf_head_acc_function,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update the optimizer
        scheduler.step()
        
        # update the model's parameters
        update_ema(ema, model_no_ddp, cfg.ema_decay)

        running_loss += loss.detach().cpu()
        num_batches += 1

        if train_steps == 0 or (train_steps + 1) % cfg.log_every == 0:
            avg_loss = running_loss / num_batches
            if distributed:
                avg_loss = avg_loss.to(device)
                dist.all_reduce(avg_loss)
                avg_loss /= world_size
                avg_loss_cpu = avg_loss.detach().cpu()
            else:
                avg_loss_cpu = avg_loss.detach().cpu()

            if rank == 0:
                loss_history.append(avg_loss_cpu)
                if pbar is not None:
                    pbar.set_postfix({"loss": avg_loss_cpu.item()})

                # Log to wandb
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({"train/loss": avg_loss_cpu.item(), "train/step": train_steps})

                plt.figure()
                plt.plot(
                    [i * cfg.log_every for i in range(len(loss_history))],
                    [loss_tensor.cpu().numpy() for loss_tensor in loss_history],
                )
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.tight_layout()
                plt.savefig(checkpoint_dir / "loss.png")
                plt.close()

            running_loss.zero_()
            num_batches = 0

        # Save checkpoint every save_iter steps
        if train_steps > 0 and train_steps % cfg.save_iter == 0:
            if distributed:
                dist.barrier()  # Synchronize before saving
            if rank == 0:
                step_str = f"{train_steps:09d}"
                ckpt_path = checkpoint_dir / f"ckpt_{step_str}.pt"
                torch.save(
                    {
                        "model": model_no_ddp.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": train_steps,
                    },
                    ckpt_path,
                )
                logging.info("Saved checkpoint to %s", ckpt_path)
            if distributed:
                dist.barrier()  # Wait for rank 0 to finish saving

        if train_steps == 0 or train_steps % cfg.validate_every == 0 and rank == 0:
            model.eval()
            with torch.no_grad():
                try:
                    val_indices, val_x, val_actions = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_indices, val_x, val_actions = next(val_iter)

                val_x = val_x.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    val_latent = vae.encode(val_x)
                    val_actions = val_actions.to(device)
                    ema.eval()
                    
                    # Generate latent frames
                    gen_outputs = diffusion.generate(
                        ema,
                        val_latent,
                        val_actions,
                        n_context_frames=1,
                        n_frames=val_latent.shape[1],
                        window_len=cfg.window_len,
                        horizon=cfg.horizon,
                    )
                    
                    # unpack outputs
                    samples = gen_outputs["x_pred"]
                    conf_samples = gen_outputs["x_conf_pred"]
                    conf_samples_composited = gen_outputs["conf_pred_composited"]

                    samples = vae.decode(samples)
                    
                    # TODO: decode confidence predictions (revisit this)
                    if conf_samples is not None:
                        conf_samples = vae.decode(conf_samples)
                    
                mse = F.mse_loss(samples, val_x)
                mse_history.append(mse.detach().cpu())

                # Log to wandb
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({"val/mse": mse.item(), "val/step": train_steps})

                plt.figure()
                plt.plot(
                    [i * cfg.validate_every for i in range(len(mse_history))],
                    [m.cpu().numpy() for m in mse_history],
                )
                plt.xlabel("step")
                plt.ylabel("mse")
                plt.tight_layout()
                plt.savefig(checkpoint_dir / "mse.png")
                plt.close()
                
                # save checkpoint for best validation error
                if cfg.save_best_val_ckpt and mse.item() < min_val_error_stats["error"]:
                    # get the last saved val checkpoint
                    last_saved_val_ckpt = checkpoint_dir / f"_best_val_error_ckpt_{min_val_error_stats['train_steps']:09d}.pt"
                    
                    # update the min val error stats
                    min_val_error_stats["error"] = mse.item()
                    min_val_error_stats["train_steps"] = train_steps
                    
                    # checkpoint path
                    ckpt_path = checkpoint_dir / f"_best_val_error_ckpt_{train_steps:09d}.pt"
                    
                    torch.save(
                        {
                            "model": model_no_ddp.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": train_steps,
                        },
                        ckpt_path,
                    )
                    logging.info("Saved best validation checkpoint to %s", ckpt_path)
                    
                    # delete the last validation checkpoint
                    try:
                        os.remove(last_saved_val_ckpt)
                        logging.info("Deleted last best validation checkpoint %s", last_saved_val_ckpt)
                    except Exception as excp:
                        print(excp)
                    
            video_np = (samples[0].float().clamp(0, 1) * 255).byte().cpu().numpy()
            step_str = f"{train_steps:09d}"
            video_path = checkpoint_dir / f"gen_{step_str}.gif"
            imageio.mimsave(video_path, video_np, fps=8)
            
            # save a GIF of the predicted confidence
            if conf_samples is not None:
                conf_video_np = (conf_samples[0].float().clamp(0, 1) * 255).byte().cpu().numpy()
                conf_video_path = checkpoint_dir / f"gen_pred_conf_{step_str}.gif"
                imageio.mimsave(conf_video_path, conf_video_np, fps=8)

            # Log video to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "val/generated_video": wandb.Video(str(video_path), fps=8, format="gif"),
                    "val/step": train_steps
                })
                
                if conf_samples is not None:
                    wandb.log({
                        "val/generated_confidence": wandb.Video(str(conf_video_path), fps=8, format="gif"),
                        "val/step": train_steps
                    })
            # torch.save(
            #     {
            #         "model": model_no_ddp.state_dict(),
            #         "ema": ema.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "step": train_steps,
            #     },
            #     checkpoint_dir / f"ckpt_{step_str}.pt",
            # )
            model.train()

        train_steps += 1
        if pbar is not None:
            pbar.update(1)

    if distributed:
        dist.destroy_process_group()

    # Finish wandb run
    if rank == 0 and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
