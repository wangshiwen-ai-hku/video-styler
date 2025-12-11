"""
Integrated Video Editing Pipeline:
1. Use agent to generate styled keyframes (from src/agent/graph.py)
2. Apply Flow Matching-based video editing with WAN-I2V
"""
import sys
sys.path.insert(0, ".")

import asyncio
from pathlib import Path
from PIL import Image
import torch
from typing import List, Optional
import json

# Agent imports
from src.agent.graph import app as agent_app
from src.agent.schema import State

# DiffSynth imports
from diffsynth import ModelManager, WanVideoEditorPipeline


class IntegratedVideoEditor:
    """
    Integrated pipeline combining:
    1. Agent-based keyframe styling (using Gemini + image generation)
    2. Flow Matching video editing (using WAN-I2V)
    """
    
    def __init__(
        self,
        wan_model_size: str = "1.3B",
        device: str = "cuda",
        torch_dtype = torch.float16,
    ):
        self.wan_model_size = wan_model_size
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = None
        
    def load_wan_pipeline(self, model_path: Optional[str] = None):
        """Load WAN video editing pipeline"""
        print(f"Loading WAN-{self.wan_model_size} model...")
        
        if model_path is None:
            model_path = f"models/WAN-{self.wan_model_size}"
        
        model_manager = ModelManager(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_id_list=[model_path]
        )
        
        self.pipe = WanVideoEditorPipeline.from_model_manager(
            model_manager,
            torch_dtype=self.torch_dtype,
            device=self.device
        )
        
        # Enable VRAM management for 14B model
        if self.wan_model_size == "14B":
            print("Enabling VRAM management for 14B model...")
            self.pipe.enable_vram_management()
        
        print("WAN pipeline loaded successfully")
    
    async def generate_styled_keyframes(
        self,
        video_path: Path,
        style_reference: dict,
        output_dir: Path,
        max_frames: int = 10,
        target_fps: float = 1.0,
    ) -> tuple[List[Image.Image], List[int], dict]:
        """
        Use agent to generate styled keyframes
        
        Returns:
            styled_keyframes: List of styled PIL Images
            keyframe_indices: Indices of keyframes in original video
            metadata: Additional metadata from agent
        """
        print("\n" + "="*60)
        print("STEP 1: Generating Styled Keyframes with Agent")
        print("="*60)
        
        # Prepare agent state
        initial_state: State = {
            "video_url": str(video_path),
            "style_reference": style_reference,
            "output_dir": output_dir,
            "max_frames": max_frames,
            "target_fps": target_fps,
            "frames": [],
            "current_frame_index": 0,
            "processed_frames": [],
            "video_metadata": {},
            "style_analysis": {},
            "consistent_style_prompt": "",
            "current_node": "init_context",
        }
        
        # Run agent workflow
        print("Running agent workflow...")
        final_state = await agent_app.ainvoke(initial_state)
        
        # Extract styled keyframes
        frames = final_state["frames"]
        styled_keyframes = []
        keyframe_indices = []
        
        for frame_info in frames:
            if frame_info.get("stylized_image_path"):
                styled_img = Image.open(frame_info["stylized_image_path"]).convert('RGB')
                styled_keyframes.append(styled_img)
                keyframe_indices.append(frame_info["frame_index"])
        
        print(f"Generated {len(styled_keyframes)} styled keyframes")
        print(f"Keyframe indices: {keyframe_indices}")
        
        metadata = {
            "style_analysis": final_state.get("style_analysis", {}),
            "consistent_style_prompt": final_state.get("consistent_style_prompt", ""),
            "video_metadata": final_state.get("video_metadata", {}),
        }
        
        return styled_keyframes, keyframe_indices, metadata
    
    def load_source_video(self, video_path: Path, num_frames: int = 81) -> List[Image.Image]:
        """Load source video frames"""
        print("\nLoading source video frames...")
        
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        print(f"Loaded {len(frames)} source frames")
        return frames
    
    def run_video_editing(
        self,
        source_frames: List[Image.Image],
        styled_keyframes: List[Image.Image],
        keyframe_indices: List[int],
        prompt: str,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        alpha: float = 10.0,
        beta: float = 0.0,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Run Flow Matching-based video editing
        """
        print("\n" + "="*60)
        print("STEP 2: Running Flow Matching Video Editing")
        print("="*60)
        print(f"Parameters: alpha={alpha}, beta={beta}, cfg_scale={cfg_scale}")
        
        if self.pipe is None:
            raise RuntimeError("WAN pipeline not loaded. Call load_wan_pipeline() first.")
        
        edited_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            source_video=source_frames,
            edited_keyframes=styled_keyframes,
            keyframe_indices=keyframe_indices,
            seed=seed,
            height=height,
            width=width,
            num_frames=len(source_frames),
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            alpha=alpha,
            beta=beta,
            tiled=True,
            verbose=True,
        )
        
        return edited_frames
    
    def save_results(self, frames: List[Image.Image], output_dir: Path):
        """Save edited video frames"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame.save(output_dir / f"edited_{i:04d}.png")
        
        print(f"\nSaved {len(frames)} edited frames to {output_dir}")
        
        # Create video file
        try:
            import cv2
            import numpy as np
            
            video_path = output_dir / "edited_video.mp4"
            height, width = frames[0].size[1], frames[0].size[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 24.0, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video file: {e}")


async def main():
    """
    Main workflow:
    1. Agent generates styled keyframes
    2. WAN-I2V propagates style to all frames with Flow Matching
    """
    
    # ========== Configuration ==========
    video_path = Path("path/to/input/video.mp4")
    style_image_path = Path("path/to/style/reference.jpg")
    output_dir = Path("outputs/integrated_editing")
    
    # Style reference
    style_reference = {
        "image_path": str(style_image_path),
        "description": "Vibrant anime style with bold colors and dramatic lighting"
    }
    
    # Agent parameters (for keyframe generation)
    max_keyframes = 10  # Number of keyframes to style
    target_fps = 1.0    # Sample rate for keyframes
    
    # WAN parameters (for video editing)
    wan_model_size = "1.3B"  # or "14B"
    prompt = "A cinematic video with consistent anime style"
    negative_prompt = "blurry, low quality, inconsistent style"
    height = 480
    width = 832
    num_frames = 81
    cfg_scale = 5.0
    num_inference_steps = 50
    alpha = 10.0  # Velocity correction strength
    beta = 0.0
    seed = 42
    
    # ========== Initialize Editor ==========
    editor = IntegratedVideoEditor(
        wan_model_size=wan_model_size,
        device="cuda",
        torch_dtype=torch.float16
    )
    
    # ========== Step 1: Generate Styled Keyframes ==========
    styled_keyframes, keyframe_indices, metadata = await editor.generate_styled_keyframes(
        video_path=video_path,
        style_reference=style_reference,
        output_dir=output_dir / "agent_output",
        max_frames=max_keyframes,
        target_fps=target_fps,
    )
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # ========== Step 2: Load WAN Pipeline ==========
    editor.load_wan_pipeline()
    
    # ========== Step 3: Load Source Video ==========
    source_frames = editor.load_source_video(video_path, num_frames=num_frames)
    
    # Map keyframe indices to full video
    # Agent samples at target_fps, we need to map to num_frames
    scale_factor = num_frames / max_keyframes
    mapped_keyframe_indices = [int(idx * scale_factor) for idx in keyframe_indices]
    
    print(f"\nMapped keyframe indices: {mapped_keyframe_indices}")
    
    # ========== Step 4: Run Video Editing ==========
    edited_frames = editor.run_video_editing(
        source_frames=source_frames,
        styled_keyframes=styled_keyframes,
        keyframe_indices=mapped_keyframe_indices,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )
    
    # ========== Step 5: Save Results ==========
    editor.save_results(edited_frames, output_dir / "final_output")
    
    print("\n" + "="*60)
    print("COMPLETE! Integrated video editing finished successfully")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
