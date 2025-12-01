import einops
import torch
from torch import nn
from abc import ABC, abstractmethod
from diffusers.models import AutoencoderKL
from dynuq.tokenizers.cosmos_tokenizer import _video_vae


# Some of the code was built off open-source code in https://github.com/world-model-eval/world-model-eval


class VAE(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        pass

class StableDiffusionVAE(VAE):
    # Input:  (B, T, H,     W,     3)
    # Output: (B, T, H//8,  W//8,  16)
    def __init__(self):
        super().__init__()
        # self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium", subfolder="vae")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
        self.vae.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x_in = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x_in = x_in * 2 - 1 # [0, 1] -> [-1, 1]

        with torch.no_grad():
            z = self.vae.encode(x_in).latent_dist.sample()

        z = z * self.vae.config.scaling_factor
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = z.shape
        z_in = einops.rearrange(z, "b t h w c -> (b t) c h w")
        z_in = z_in / self.vae.config.scaling_factor

        with torch.no_grad():
            x = self.vae.decode(z_in, return_dict=False)[0]

        x = (x + 1) / 2
        x = einops.rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T)
        return x

class CosmosVAE(VAE):
    # Input:  (B, T,    H,     W,     3)
    # Output: (B, T//4, H//8,  W//8,  16)
    def __init__(
        self,
        dtype=torch.float,
        device="cuda",
        temporal_window: int = 4,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.temporal_window = temporal_window

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model, self.img_mean, self.img_std, self.video_mean, self.video_std = _video_vae(
            pretrained_path="/n/fs/tom-project/action_world_model/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
            device=device,
            temporal_window=4,
        )
        self.model = self.model.eval().requires_grad_(False)
        self.context = torch.amp.autocast("cuda", dtype=dtype)

    def count_param(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video tensor.
        Args:
            x: Input tensor with shape (B, T, H, W, C)
        Returns:
            Encoded latent tensor with shape (B, T//4, H//8, W//8, 16)
            Note: Temporal compression factor is 4, spatial compression factor is 8
        """
        B, T, H, W, C = x.shape
        # Convert from (B, T, H, W, C) to (B, C, T, H, W) for Cosmos model
        x_in = einops.rearrange(x, "b t h w c -> b c t h w")

        in_dtype = x_in.dtype
        with self.context:
            latent = self.model.encode(x_in, self.scale)
        latent = latent.to(in_dtype)

        # Convert back to (B, T//4, H//8, W//8, C) format
        # Note: temporal dimension is compressed by factor of 4
        latent = einops.rearrange(latent, "b c t h w -> b t h w c")
        return latent

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to video.
        Args:
            z: Input latent tensor with shape (B, T//4, H//8, W//8, 16)
        Returns:
            Decoded video tensor with shape (B, T, H, W, 3)
            Note: Temporal decompression factor is 4, spatial decompression factor is 8
        """
        B, T_compressed, H_compressed, W_compressed, C = z.shape
        # Convert from (B, T//4, H//8, W//8, C) to (B, C, T//4, H//8, W//8) for Cosmos model
        z_in = einops.rearrange(z, "b t h w c -> b c t h w")

        in_dtype = z_in.dtype
        with self.context:
            video_recon = self.model.decode(z_in, self.scale)
        video_recon = video_recon.to(in_dtype)

        # Convert back to (B, T, H, W, C) format
        # Note: temporal dimension is decompressed by factor of 4, spatial by factor of 8
        video_recon = einops.rearrange(video_recon, "b c t h w -> b t h w c")
        return video_recon

