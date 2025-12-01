from typing import Callable, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import einops
import numpy as np
from pathlib import Path


# Some of the code was built off open-source code in https://github.com/world-model-eval/world-model-eval


class Diffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1_000,
        sampling_timesteps: int = 10,
        *,
        enable_confidence_pred: bool = False,
        device: torch.device | str | None = None,
        ref_latent_dir: str = None,
        default_err_threshold_conf_pred: float = 0.85,
    ) -> None:
        super().__init__()

        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        alphas = 1 - self.sigmoid_beta_schedule(self.timesteps)
        # an unfortunate variable name, but it's the standard one
        self.register_buffer("alphas_cumprod", alphas.cumprod(dim=0), persistent=False)
        self.stabilization_level = 15
        
        # option to enable confidence prediction head
        self.enable_confidence_pred = enable_confidence_pred
        
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        
        # load the refereence latents for the RGB space
        if self.enable_confidence_pred:
            # ref latents
            self.ref_latents = {}
            
            # load latents
            for ref_latent_channel in ["red", "green", "blue"]:
                latent_path = Path(ref_latent_dir) / f"{ref_latent_channel}_latent.pt"
                self.ref_latents[ref_latent_channel] = torch.load(latent_path, map_location="cpu").to(self.device)
            
            # initialize error threshold levels
            # TODO: Refactor
            self.err_threshold_conf_pred = torch.concat(
                (torch.linspace(7e-1, 1e-1, steps=4),
                 torch.linspace(1e0, 7.5e-1, steps=6)),
                dim=-1,
            ).to(self.device)
            
            # sort
            self.err_threshold_conf_pred, _ = torch.sort(self.err_threshold_conf_pred)
            
            # default error threshold level
            self.default_err_threshold_conf_pred = default_err_threshold_conf_pred
            
        # TODO: temporary counter for number of training steps (calls to the loss function)
        self.loss_steps = 0
        
        # TODO: temporary number of steps to warmup the calibration loss weight
        self.calib_warmup_steps = 5000 

    def sigmoid_beta_schedule(self, timesteps: int, start: float = -3, end: float = 3, tau: float = 1) -> torch.Tensor:
        # https://arxiv.org/abs/2212.11972
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).float()

    def q_sample(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        B, T = t.shape

        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)

        return alphas_cumprod.sqrt() * x + (1 - alphas_cumprod).sqrt() * noise

    def loss_fn(
        self,
        model: nn.Module, 
        x: torch.Tensor, 
        actions: torch.Tensor, 
        weight: torch.Tensor = None,
        calib_weight: float = 0.5,
        conf_acc_function: Callable = None,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        B, T, D = actions.shape

        t = torch.randint(0, self.timesteps, (B, T), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        
        if conf_acc_function is not None:
            # TODO: Add support for per-sample error threshold
            # randomly sample an error threshold for evaluating accuracy
            # err_threshold = self.err_threshold_conf_pred[torch.randint(0, len(self.err_threshold_conf_pred))]

            # # map to the same shape as the timestep
            # err_threshold = err_threshold * torch.ones_like(t)
            
            # randomly sample an error threshold for evaluating accuracy
            err_threshold = self.err_threshold_conf_pred[
                torch.randint(0, 
                              len(self.err_threshold_conf_pred),
                              (B, T), 
                              device=self.err_threshold_conf_pred.device, 
                              dtype=torch.long).view(-1,)
            ].view(B, T).to(x.device)
            
        x_t = self.q_sample(x, t, noise)
        pred_v, pred_conf = model(x_t, t, actions, err_threshold_conf_pred=err_threshold)

        # Build target v
        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        target_v = alphas_cumprod.sqrt() * noise - (1 - alphas_cumprod).sqrt() * x

        loss = F.mse_loss(pred_v, target_v, weight=weight)
        
        if self.enable_confidence_pred:
            # compute the accuracy of the confidence prediction
            if conf_acc_function is not None:
                acc = conf_acc_function(pred=pred_v, target=target_v, err_threshold=err_threshold)
            else:
                raise ValueError("conf_acc_function must be provided when enable_confidence_pred is True!")
                
            # include the confidence prediction loss
            calib_loss = F.binary_cross_entropy_with_logits(
                input=pred_conf,
                target=acc,
                weight=weight,
            )
            
            # # TODO: temporary scheduled weight
            # calib_weight = np.exp(-max(0, self.calib_warmup_steps - self.loss_steps)) * calib_weight
            
            # TODO: Add relative weights, with optional scheduler
            # sum of the loss components
            loss = (1 - calib_weight) * loss + calib_weight * calib_loss
            
            # incremenent the loss step counter
            self.loss_steps += 1
            
        return loss

    def ddim_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        t_idx: torch.Tensor,
        t_next_idx: torch.Tensor,
        shift_conf_pred_lower_range: float = 0.35,
        err_threshold_conf_pred: torch.tensor = None,
    ) -> torch.Tensor:
        # Derived from
        # https://github.com/buoyancy99/diffusion-forcing/blob/475e0bcab87545e48b24b39fb46a81fe59d80594/algorithms/diffusion_forcing/models/diffusion.py#L383
        B, T, H, W, C = x.shape
        B, T, D = actions.shape
        B, T = t_idx.shape
        
        # input-checking
        if err_threshold_conf_pred is None:
            # use default value
            err_threshold_conf_pred = self.default_err_threshold_conf_pred

        if isinstance(err_threshold_conf_pred, float):
            # reshape to (B, T, H, W, C)
            # map to the same shape as the timestep
            err_threshold_conf_pred = err_threshold_conf_pred * torch.ones_like(t_idx).to(x.device)

        sampling_noise_steps = torch.linspace(
            -1,
            self.timesteps - 1,
            steps=self.sampling_timesteps + 1,
            device=x.device,
            dtype=torch.long,
        )
        t = sampling_noise_steps[t_idx]
        t_next = sampling_noise_steps[t_next_idx]

        clipped_t = torch.where(t < 0, torch.full_like(t, self.stabilization_level - 1, dtype=torch.long), t)
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(x, clipped_t, torch.zeros_like(x))
        x = torch.where(t.reshape(B, T, 1, 1, 1) < 0, scaled_context, x)

        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        alphas_next_cumprod = torch.where(
            t_next < 0,
            torch.ones_like(t_next),
            self.alphas_cumprod[t_next.reshape(-1)].view(B, T),
        ).view(B, T, 1, 1, 1)
        c = (1 - alphas_next_cumprod).sqrt()

        v_pred, v_conf_pred = model(x, clipped_t, actions, err_threshold_conf_pred=err_threshold_conf_pred)
        x_start = alphas_cumprod.sqrt() * x - (1 - alphas_cumprod).sqrt() * v_pred
        pred_noise = ((1 / alphas_cumprod).sqrt() * x - x_start) / ((1 / alphas_cumprod) - 1).sqrt()
        x_pred = alphas_next_cumprod.sqrt() * x_start + c * pred_noise
        x_pred = torch.where(
            (t == t_next).view(B, T, 1, 1, 1),
            orig_x,
            x_pred,
        )
        
        if v_conf_pred is not None:
            # map to a probability distribution
            v_conf_pred = torch.sigmoid(v_conf_pred)
            
            # map to the latent space for conversion to RGB space later
            # TODO: Add support for better visualization
            conf_mask = (v_conf_pred <= 0.5)
            
            # B, T, C, H, W (in latent space)
            x_conf_pred = torch.zeros_like(v_conf_pred)
            
            # TODO: Refactor
            # map to the RGB space for values in [0, 0.5]
            v_conf_pred_shifted = v_conf_pred[conf_mask]
            
            # TODO: Revisit normalization
            if shift_conf_pred_lower_range is not None:
                v_conf_pred_shifted = torch.clamp(v_conf_pred_shifted - shift_conf_pred_lower_range, min=0)
                v_conf_pred_shifted = v_conf_pred_shifted / (0.5 - shift_conf_pred_lower_range)
            
            # x_conf_pred[conf_mask] = (
            #     (v_conf_pr ed_shifted / 0.5) * torch.broadcast_to(self.ref_latents["green"], x_conf_pred.shape).to(x_conf_pred.dtype)[conf_mask] 
            #     + ((0.5 - v_conf_pred_shifted) / 0.5) * torch.broadcast_to(self.ref_latents["red"], x_conf_pred.shape).to(x_conf_pred.dtype)[conf_mask]
            # )
            x_conf_pred[conf_mask] = (
                v_conf_pred_shifted * torch.broadcast_to(self.ref_latents["red"], x_conf_pred.shape).to(x_conf_pred.dtype)[conf_mask] 
                + (1 - v_conf_pred_shifted) * torch.broadcast_to(self.ref_latents["green"], x_conf_pred.shape).to(x_conf_pred.dtype)[conf_mask]
            )
            
            # TODO: Revisit normalization
            # map to the RGB space for values in (0.5, 1.0]
            v_conf_pred_shifted = v_conf_pred[~conf_mask] 
            
            if shift_conf_pred_lower_range is not None:
                v_conf_pred_shifted = torch.clamp(v_conf_pred_shifted - 0.5, min=0)
                v_conf_pred_shifted = v_conf_pred_shifted / 0.5
            
            x_conf_pred[~conf_mask] = (
                v_conf_pred_shifted * torch.broadcast_to(self.ref_latents["blue"], x_conf_pred.shape).to(x_conf_pred.dtype)[~conf_mask] 
                + (1 - v_conf_pred_shifted) * torch.broadcast_to(self.ref_latents["red"], x_conf_pred.shape).to(x_conf_pred.dtype)[~conf_mask]
            )
            
            # 
            # x_conf_pred = v_conf_pred * self.ref_latents["blue"] + (1 - v_conf_pred) * self.ref_latents["red"]
        else:
            x_conf_pred = None
            raw_conf_pred = None
        
        # outputs
        outputs = {
            "x_pred": x_pred,             # predicted RGB latents
            "x_conf_pred": x_conf_pred,   # predicted confidence in RGB latent space
            "raw_conf_pred": v_conf_pred,  # raw predicted confidence
        }
        
        return outputs

    def generate_pyramid_scheduling_matrix(self, horizon: int) -> torch.Tensor:
        height = self.sampling_timesteps + horizon
        scheduling_matrix = torch.zeros((height, horizon), dtype=torch.long)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + t - m
        return torch.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        n_context_frames: int = 1,
        n_frames: int = 1,
        horizon: int = 1,
        window_len: int | None = None,
        rgb_conf_pred_mask_threshold: float = 0.75,  # threshold for masking RGB and confidence pred
        err_threshold_conf_pred: torch.tensor = None,  # error threshold for accuracy function in confidence prediction
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        curr_frame = 0
        x_pred = x[:, :n_context_frames]
        curr_frame += n_context_frames
        
        if self.enable_confidence_pred:
            # raw confidence
            raw_conf_pred = torch.zeros_like(x_pred)
            
            # confidence in RGB latent space
            x_conf_pred = torch.zeros_like(x_pred)
        else:
            # raw confidence
            raw_conf_pred = None
            
            # confidence in RGB latent space
            x_conf_pred = None

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            horizon = min(n_frames - curr_frame, horizon)
            scheduling_matrix = self.generate_pyramid_scheduling_matrix(horizon)

            chunk = torch.randn((B, horizon, *x.shape[-3:]), device=self.device)
            x_pred = torch.cat([x_pred, chunk], dim=1)
            
            if self.enable_confidence_pred:
                raw_conf_pred = torch.cat([raw_conf_pred, torch.zeros_like(chunk)], dim=1)
                x_conf_pred = torch.cat([x_conf_pred, torch.zeros_like(chunk)], dim=1)

            # Adjust context length
            start_frame = max(0, curr_frame + horizon - (window_len or model.max_frames))

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1): # sampling_timesteps + horizon
                t, t_next = scheduling_matrix[m], scheduling_matrix[m + 1]
                t, t_next = map(lambda x: einops.repeat(x, "t -> b t", b=B), (t, t_next))
                t, t_next = map(
                    lambda x: torch.cat((torch.zeros((B, curr_frame), dtype=torch.long), x), dim=1),
                    (t, t_next),
                )

                ddim_outputs = self.ddim_sample_step(
                    model,
                    x_pred[:, start_frame:],
                    actions[:, start_frame : curr_frame + horizon],
                    t[:, start_frame:],
                    t_next[:, start_frame:],
                    err_threshold_conf_pred=err_threshold_conf_pred,
                )
                
                # unpack outputs
                x_pred[:, start_frame:] = ddim_outputs["x_pred"]
                x_conf_pred_batch = ddim_outputs["x_conf_pred"]
                raw_conf_pred_batch = ddim_outputs["raw_conf_pred"]
                
                if self.enable_confidence_pred and x_conf_pred_batch is not None:
                    # # Uses diffusion forcing
                    # x_conf_pred[:, start_frame:] = x_conf_pred_batch
                    
                    # TODO: Revisit, disable diffusion forcing
                    x_conf_pred[:, -1:] = x_conf_pred_batch[:, -1:]
                    raw_conf_pred[:, -1:] = raw_conf_pred_batch[:, -1:]

            curr_frame += horizon
            pbar.update(horizon)
            
        # TODO: refactor
        if self.enable_confidence_pred:
            # blend RGB and confidence predictions via mask
            conf_pred_mask = raw_conf_pred > rgb_conf_pred_mask_threshold
            
            # blend RGB and conf pred
            conf_pred_composited = x_conf_pred.detach().clone()
            conf_pred_composited[conf_pred_mask] = x_pred[conf_pred_mask]
        else:
            conf_pred_composited = None
        
        # outputs
        outputs = {
            "x_pred": x_pred,                               # predicted RGB latents
            "x_conf_pred": x_conf_pred,                     # predicted confidence in RGB latent space
            "raw_conf_pred": raw_conf_pred,                 # raw predicted confidence
            "conf_pred_composited": conf_pred_composited,   # composited predicted confidence
        }
            
        return outputs
