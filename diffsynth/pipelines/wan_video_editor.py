"""
Flow Matching-based Video Editing Pipeline for WAN-I2V
Implements keyframe-guided video editing with coupled noise and velocity field correction
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from PIL import Image
import numpy as np
from einops import rearrange
from tqdm import tqdm

from .wan_video import WanVideoPipeline, model_fn_wan_video
from ..models import ModelManager
from ..models.wan_video_dit import sinusoidal_embedding_1d


class WanVideoEditorPipeline(WanVideoPipeline):
    """
    Video editing pipeline using Flow Matching with keyframe guidance.
    
    Key features:
    1. Coupled noise initialization for edited keyframes and original video
    2. Shared positional encoding (RoPE) for corresponding frames
    3. Velocity field correction to maintain consistency
    """
    
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        
    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        """Create pipeline from model manager"""
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoEditorPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward
            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    def prepare_coupled_noise(
        self,
        latent_shape: Tuple[int, ...],
        keyframe_indices: List[int],
        seed: Optional[int] = None,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate coupled noise for original video and edited keyframes.
        Ensures corresponding frames start from the same noise.
        
        Args:
            latent_shape: Shape of the main video latents (B, C, T, H, W)
            keyframe_indices: List of keyframe indices in the original video
            seed: Random seed for reproducibility
            device: Device for noise generation
            
        Returns:
            noise_main: Noise for original video sequence
            noise_edit: Noise for edited keyframes (extracted from noise_main)
        """
        # Generate noise for main video
        noise_main = self.generate_noise(latent_shape, seed=seed, device=device, dtype=torch.float32)
        
        # Extract noise for keyframes - CRITICAL: same noise for corresponding positions
        noise_edit = noise_main[:, :, keyframe_indices, :, :].clone()
        
        return noise_main, noise_edit
    
    def construct_rope_ids(
        self,
        total_frames: int,
        keyframe_indices: List[int],
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Construct RoPE position IDs ensuring edited keyframes have same position encoding
        as their corresponding original frames.
        
        Args:
            total_frames: Total number of frames in original video
            keyframe_indices: Indices of keyframes in original video
            device: Target device
            
        Returns:
            full_ids: Concatenated position IDs [T_main + T_edit]
        """
        # Original video temporal indices: [0, 1, 2, ..., T-1]
        ids_main = torch.arange(total_frames, device=device)
        
        # Edited keyframes use SAME indices as original positions
        # e.g., if keyframes=[0, 24], ids_edit=[0, 24] NOT [0, 1]
        ids_edit = torch.tensor(keyframe_indices, device=device)
        
        # Concatenate for joint processing
        full_ids = torch.cat([ids_main, ids_edit])
        
        return full_ids
    
    def compute_velocity_correction(
        self,
        z_main: torch.Tensor,
        z_edit: torch.Tensor,
        v_main: torch.Tensor,
        v_edit: torch.Tensor,
        keyframe_indices: List[int],
        dt: float,
        alpha: float = 10.0,
        beta: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity field correction based on consistency constraints.
        
        Implements the core algorithm from tech_design.md:
        Δv_k = (ε - z_k) / t - (ε - z_i_k) / t = (z_i_k - z_k) / t
        r_k = Δv_k - (z_k^t - z_i_k^t) / t
        
        Args:
            z_main: Current latents of main video (B, C, T, H, W)
            z_edit: Current latents of edited keyframes (B, C, K, H, W)
            v_main: Predicted velocity for main video
            v_edit: Predicted velocity for edited keyframes
            keyframe_indices: Keyframe positions
            dt: Time step size
            alpha: Correction strength for main video
            beta: Correction strength for edited keyframes (usually 0 to preserve DMI quality)
            
        Returns:
            v_main_corrected: Corrected velocity for main video
            v_edit_corrected: Corrected velocity for edited keyframes
        """
        # Extract velocities at keyframe positions
        v_main_at_keys = v_main[:, :, keyframe_indices, :, :]  # (B, C, K, H, W)
        
        # Current latent difference at keyframe positions
        z_diff = z_main[:, :, keyframe_indices, :, :] - z_edit  # (B, C, K, H, W)
        
        # Velocity difference
        v_diff = v_main_at_keys - v_edit  # (B, C, K, H, W)
        
        # Consistency residual r_k
        # r_k measures how much the denoising routes diverge
        r_k = z_diff - v_diff * dt  # (B, C, K, H, W)
        
        # Correction term
        # Force v_main to align with v_edit while compensating for r_k
        correction = alpha * r_k  # (B, C, K, H, W)
        
        # Apply correction to main video at keyframe positions
        v_main_corrected = v_main.clone()
        v_main_corrected[:, :, keyframe_indices, :, :] += correction
        
        # Optionally correct edited keyframes (usually beta=0 to preserve DMI quality)
        v_edit_corrected = v_edit
        if beta > 0:
            v_edit_corrected = v_edit - beta * correction
        
        return v_main_corrected, v_edit_corrected
    
    def compute_metrics(
        self,
        z_main: torch.Tensor,
        z_edit: torch.Tensor,
        v_main: torch.Tensor,
        v_edit: torch.Tensor,
        keyframe_indices: List[int],
        dt: float
    ) -> Dict[str, float]:
        """
        Compute monitoring metrics as described in tech_design.md
        
        Returns:
            metrics: Dictionary containing:
                - r_k_norm: Consistency residual magnitude
                - v_diff_norm: Velocity difference magnitude
                - delta_v_norm: Latent difference magnitude
        """
        v_main_at_keys = v_main[:, :, keyframe_indices, :, :]
        z_diff = z_main[:, :, keyframe_indices, :, :] - z_edit
        v_diff = v_main_at_keys - v_edit
        r_k = z_diff - v_diff * dt
        
        metrics = {
            "r_k_norm": torch.mean(torch.abs(r_k)).item(),
            "v_diff_norm": torch.mean(torch.abs(v_diff)).item(),
            "delta_v_norm": torch.mean(torch.abs(z_diff)).item(),
        }
        
        return metrics
    
    def encode_keyframes_independently(
        self,
        keyframes: List[Image.Image],
        tiler_kwargs: dict
    ) -> torch.Tensor:
        """
        Encode keyframes independently as single-frame videos.
        
        CRITICAL DESIGN DECISION:
        Keyframes are temporally distant (e.g., frames 0, 20, 40, 60, 80).
        Using 3D convolution on concatenated keyframes would incorrectly assume
        temporal continuity between them. Instead, we encode each keyframe as
        a separate 1-frame video.
        
        This ensures:
        1. No artificial temporal correlation between distant frames
        2. Each keyframe is encoded with proper 3D VAE context (even if T=1)
        3. Latent representations are independent and clean
        
        Args:
            keyframes: List of PIL Images (edited keyframes)
            tiler_kwargs: VAE tiling parameters
            
        Returns:
            z_edit: Concatenated latents (B, C, K, H, W)
        """
        z_edit_list = []
        for i, keyframe in enumerate(keyframes):
            # Preprocess single keyframe
            keyframe_tensor = self.preprocess_images([keyframe])
            # Add temporal dimension: (C, H, W) -> (C, 1, H, W)
            keyframe_tensor = torch.stack(keyframe_tensor, dim=2).to(dtype=self.torch_dtype, device=self.device)
            # Encode as single-frame video
            z_keyframe = self.encode_video(keyframe_tensor, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            z_edit_list.append(z_keyframe)
        
        # Concatenate along temporal dimension: List[(B,C,1,H,W)] -> (B,C,K,H,W)
        z_edit_clean = torch.cat(z_edit_list, dim=2)
        return z_edit_clean

    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        source_video: Optional[List[Image.Image]] = None,
        edited_keyframes: Optional[List[Image.Image]] = None,
        keyframe_indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        alpha: float = 10.0,
        beta: float = 0.0,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        progress_bar_cmd = tqdm,
        progress_bar_st = None,
        verbose: bool = True,
    ):
        """
        Edit video using Flow Matching with keyframe guidance.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            source_video: Original video frames (List of PIL Images)
            edited_keyframes: Edited keyframes (List of PIL Images)
            keyframe_indices: Indices of keyframes in source video
            seed: Random seed
            rand_device: Device for random number generation
            height: Video height (must be divisible by 16)
            width: Video width (must be divisible by 16)
            num_frames: Number of frames (must satisfy num_frames % 4 == 1)
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            sigma_shift: Scheduler shift parameter
            alpha: Velocity correction strength for main video
            beta: Velocity correction strength for edited keyframes
            tiled: Use tiled VAE encoding/decoding
            tile_size: Tile size for VAE
            tile_stride: Tile stride for VAE
            progress_bar_cmd: Progress bar function
            progress_bar_st: Streamlit progress bar
            verbose: Print monitoring metrics
            
        Returns:
            frames: List of edited video frames (PIL Images)
        """
        # Validate inputs
        if source_video is None or edited_keyframes is None or keyframe_indices is None:
            raise ValueError("source_video, edited_keyframes, and keyframe_indices are required")
        
        if len(edited_keyframes) != len(keyframe_indices):
            raise ValueError(f"Number of edited keyframes ({len(edited_keyframes)}) must match keyframe_indices ({len(keyframe_indices)})")
        
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 == 1` is acceptable. Rounded to {num_frames}.")
        
        if len(source_video) != num_frames:
            print(f"Warning: source_video has {len(source_video)} frames, expected {num_frames}")
            num_frames = len(source_video)
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)
        
        # ========== Encode source video ==========
        self.load_models_to_device(['vae'])
        source_video_tensor = self.preprocess_images(source_video)
        source_video_tensor = torch.stack(source_video_tensor, dim=2).to(dtype=self.torch_dtype, device=self.device)
        z_main_clean = self.encode_video(source_video_tensor, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        
        # ========== Encode edited keyframes ==========
        # IMPORTANT: Encode each keyframe independently as single-frame video
        # because keyframes are temporally distant and 3D convolution is not suitable
        z_edit_clean = self.encode_keyframes_independently(edited_keyframes, tiler_kwargs)
        
        # ========== Prepare coupled noise ==========
        noise_main, noise_edit = self.prepare_coupled_noise(
            z_main_clean.shape,
            keyframe_indices,
            seed=seed,
            device=rand_device
        )
        noise_main = noise_main.to(dtype=self.torch_dtype, device=self.device)
        noise_edit = noise_edit.to(dtype=self.torch_dtype, device=self.device)
        
        # Initialize latents from noise
        z_main = noise_main
        z_edit = noise_edit
        
        # ========== Encode prompts ==========
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        
        # ========== Construct RoPE IDs ==========
        # CRITICAL: Edited keyframes must have same position encoding as original frames
        rope_ids = self.construct_rope_ids(
            z_main_clean.shape[2],  # Total frames in temporal dimension
            keyframe_indices,
            device=self.device
        )
        
        # ========== Denoising loop ==========
        self.load_models_to_device(["dit"])
        
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_tensor = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Concatenate latents for joint processing
            # Shape: (B, C, T_main + T_edit, H, W)
            z_concat = torch.cat([z_main, z_edit], dim=2)
            
            # ========== Forward pass through DiT with custom RoPE ==========
            velocity_pred_posi = self.dit(
                x=z_concat,
                timestep=timestep_tensor,
                context=prompt_emb_posi["context"],
                rope_indices=rope_ids
            )
            
            # CFG
            if cfg_scale != 1.0:
                velocity_pred_nega = self.dit(
                    x=z_concat,
                    timestep=timestep_tensor,
                    context=prompt_emb_nega["context"],
                    rope_indices=rope_ids
                )
                velocity_pred = velocity_pred_nega + cfg_scale * (velocity_pred_posi - velocity_pred_nega)
            else:
                velocity_pred = velocity_pred_posi
            
            # Split velocity predictions
            v_main, v_edit = torch.split(velocity_pred, [z_main.shape[2], z_edit.shape[2]], dim=2)
            
            # ========== Velocity correction ==========
            dt = (self.scheduler.timesteps[progress_id] - self.scheduler.timesteps[progress_id + 1]).item() if progress_id < len(self.scheduler.timesteps) - 1 else 0
            
            v_main_corrected, v_edit_corrected = self.compute_velocity_correction(
                z_main, z_edit, v_main, v_edit,
                keyframe_indices, dt, alpha, beta
            )
            
            # Monitoring metrics
            if verbose and progress_id % 10 == 0:
                metrics = self.compute_metrics(z_main, z_edit, v_main, v_edit, keyframe_indices, dt)
                print(f"Step {progress_id}: r_k={metrics['r_k_norm']:.6f}, "
                      f"v_diff={metrics['v_diff_norm']:.6f}, "
                      f"Δv={metrics['delta_v_norm']:.6f}")
            
            # ========== Update latents (Euler step) ==========
            z_main = self.scheduler.step(v_main_corrected, self.scheduler.timesteps[progress_id], z_main)
            z_edit = self.scheduler.step(v_edit_corrected, self.scheduler.timesteps[progress_id], z_edit)
        
        # ========== Decode final video ==========
        self.load_models_to_device(['vae'])
        frames = self.decode_video(z_main, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        
        return frames
