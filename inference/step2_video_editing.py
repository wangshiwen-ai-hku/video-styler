"""
Step 2: Video Editing with WAN-I2V
从第一步保存的结果中加载数据，运行 Flow Matching 视频编辑
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from PIL import Image
import torch
from typing import List, Optional
import json
import cv2
import numpy as np

# DiffSynth imports
from diffsynth import ModelManager, WanVideoEditorPipeline


class VideoEditor:
    """
    使用 WAN-I2V 进行 Flow Matching 视频编辑
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
        """加载 WAN video editing pipeline"""
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
    
    def load_source_video(self, video_path: Path, num_frames: int = 81) -> List[Image.Image]:
        """加载源视频帧"""
        print(f"\nLoading source video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {total_frames} frames, {fps} fps")
        
        # 均匀采样帧
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
    
    def load_styled_keyframes(self, keyframe_info: dict) -> tuple[List[Image.Image], List[int]]:
        """从保存的结果中加载风格化的关键帧"""
        print("\nLoading styled keyframes...")
        
        styled_keyframes = []
        keyframe_indices = []
        keyframes = keyframe_info["generated_frames"]
        keyframe_timestamps = keyframe_info["keyframe_timestamp"]
        keyframe_fps = keyframe_info["keyframe_fps"]
        source_fps = keyframe_info["source_fps"]
        
        for frame, timestamp in zip(keyframes, keyframe_timestamps):
            img = Image.open(frame).convert('RGB')
            styled_keyframes.append(img)
            keyframe_indices.append(int(timestamp * source_fps))
        
        return styled_keyframes, keyframe_indices
    
    def map_keyframe_indices(
        self,
        agent_keyframe_indices: List[int],
        max_agent_frames: int,
        num_video_frames: int
    ) -> List[int]:
        """
        将 agent 的关键帧索引映射到完整视频的帧索引
        
        agent 在 max_agent_frames 范围内采样
        需要映射到 num_video_frames 范围
        """
        scale_factor = num_video_frames / max_agent_frames
        mapped_indices = [int(idx * scale_factor) for idx in agent_keyframe_indices]
        
        print(f"\nMapping keyframe indices:")
        print(f"  Agent range: 0-{max_agent_frames}")
        print(f"  Video range: 0-{num_video_frames}")
        print(f"  Scale factor: {scale_factor:.2f}")
        print(f"  Original indices: {agent_keyframe_indices}")
        print(f"  Mapped indices: {mapped_indices}")
        
        return mapped_indices
    
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
        """运行 Flow Matching 视频编辑"""
        print("\n" + "="*60)
        print("STEP 2: Running Flow Matching Video Editing")
        print("="*60)
        print(f"Parameters:")
        print(f"  alpha={alpha}, beta={beta}")
        print(f"  cfg_scale={cfg_scale}")
        print(f"  num_inference_steps={num_inference_steps}")
        print(f"  height={height}, width={width}")
        print(f"  seed={seed}")
        
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
    
    def save_results(self, frames: List[Image.Image], output_dir: Path, fps: float = 24.0):
        """保存编辑后的视频帧和视频文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存帧图像
        print(f"\nSaving {len(frames)} frames...")
        for i, frame in enumerate(frames):
            frame.save(output_dir / f"edited_{i:04d}.png")
        
        print(f"Frames saved to {output_dir}")
        
        # 创建视频文件
        try:
            video_path = output_dir / "edited_video.mp4"
            height, width = frames[0].size[1], frames[0].size[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video file: {e}")


def load_keyframe_results(result_dir: Path) -> dict:
    """
    从第一步的结果目录中加载所有必要信息
    
    Returns:
        包含所有必要信息的字典
    """
    result_dir = Path(result_dir)
    
    # 加载 keyframe_info.json
    keyframe_info_path = result_dir / "keyframe_info.json"
    if not keyframe_info_path.exists():
        raise FileNotFoundError(f"keyframe_info.json not found in {result_dir}")
    
    with open(keyframe_info_path, "r") as f:
        keyframe_info = json.load(f)
    
    return {
        "keyframe_info": keyframe_info,
        "result_dir": result_dir,
    }


def main():
    """
    主函数：从第一步的结果中加载数据并运行视频编辑
    """
    
    # ========== 配置参数 ==========
    
    # 第一步的结果目录（自动查找最新的结果）
    result_dir = Path("outputs/video_editing/2/20251212_123456")

    print(f"Loading results from: {result_dir}")
    
    # 加载第一步的结果
    results = load_keyframe_results(result_dir)
    keyframe_info = results["keyframe_info"]
    
    
    print(f"\nLoaded keyframe info:")
    print(f"  Video: {keyframe_info['video_path']}")
    print(f"  Keyframes: {keyframe_info['num_keyframes']}")
    print(f"  Indices: {keyframe_info['keyframe_timestamp']}")
    
    # WAN 参数
    wan_model_size = "1.3B"  # 或 "14B"
    prompt = keyframe_info.get("consistent_edit_prompt", "A cinematic video with consistent style")
    negative_prompt = keyframe_info.get("negative_prompt", "blurry, low quality, inconsistent style")
    height = 480
    width = 832
    num_frames = 81  # 完整视频的帧数
    cfg_scale = 5.0
    num_inference_steps = 50
    alpha = 10.0  # Velocity correction strength
    beta = 0.0
    seed = 42
    
    print(f"\nVideo editing parameters:")
    print(f"  Model: WAN-{wan_model_size}")
    print(f"  Prompt: {prompt}")
    print(f"  Frames: {num_frames}")
    
    # ========== 初始化编辑器 ==========
    editor = VideoEditor(
        wan_model_size=wan_model_size,
        device="cuda",
        torch_dtype=torch.float16
    )
    
    # ========== 加载 WAN Pipeline ==========
    editor.load_wan_pipeline()
    
    # ========== 加载源视频 ==========
    video_path = Path(keyframe_info["video_path"])
    source_frames = editor.load_source_video(video_path, num_frames=num_frames)
    
    # ========== 加载风格化的关键帧 ==========
    styled_keyframes, keyframe_indices = editor.load_styled_keyframes(keyframe_info)
    source_frames = source_frames[:keyframe_indices[-1]]

    # ========== 运行视频编辑 ==========
    edited_frames = editor.run_video_editing(
        source_frames=source_frames,
        styled_keyframes=styled_keyframes,
        keyframe_indices=keyframe_indices,
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
    
    # ========== 保存结果 ==========
    output_dir = result_dir / "final_output"
    editor.save_results(edited_frames, output_dir, fps=24.0)
    
    print("\n" + "="*60)
    print("STEP 2 COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
