"""
Example: Video Editing with WAN-I2V using Flow Matching
Demonstrates keyframe-guided video editing with coupled noise and velocity correction
"""
import sys
sys.path.insert(0, ".")

from diffsynth import ModelManager, WanVideoEditorPipeline
from PIL import Image
import torch
import os

def load_video_frames(video_path, max_frames=81):
    """Load video frames from a directory or video file"""
    if os.path.isdir(video_path):
        # Load from directory
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        frames = [Image.open(os.path.join(video_path, f)).convert('RGB') for f in frame_files[:max_frames]]
    else:
        # Load from video file (requires cv2)
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
    return frames

def save_video_frames(frames, output_dir):
    """Save frames to directory"""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
    print(f"Saved {len(frames)} frames to {output_dir}")

def main():
    # ========== Configuration ==========
    # Model selection: "1.3B" or "14B"
    model_size = "1.3B"  # Change to "14B" for larger model
    
    # Paths
    source_video_path = "path/to/source/video"  # Directory or video file
    edited_keyframes_dir = "path/to/edited/keyframes"  # Directory with edited keyframes
    output_dir = "outputs/edited_video"
    
    # Keyframe indices (which frames were edited)
    keyframe_indices = [0, 20, 40, 60, 80]  # Example: edit every 20th frame
    
    # Generation parameters
    prompt = "A cinematic video with vibrant colors and dramatic lighting"
    negative_prompt = "blurry, low quality, distorted"
    height = 480
    width = 832
    num_frames = 81
    cfg_scale = 5.0
    num_inference_steps = 50
    seed = 42
    
    # Flow Matching parameters
    alpha = 10.0  # Velocity correction strength (higher = stronger consistency)
    beta = 0.0    # Keep 0 to preserve DMI quality
    
    # ========== Load Models ==========
    print(f"Loading WAN-{model_size} model...")
    model_manager = ModelManager(
        torch_dtype=torch.float16,
        device="cuda",
        model_id_list=[
            f"models/WAN-{model_size}",  # Adjust path as needed
        ]
    )
    
    pipe = WanVideoEditorPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.float16,
        device="cuda"
    )
    
    # Optional: Enable VRAM management for large models
    if model_size == "14B":
        pipe.enable_vram_management()
    
    # ========== Load Data ==========
    print("Loading source video...")
    source_frames = load_video_frames(source_video_path, max_frames=num_frames)
    print(f"Loaded {len(source_frames)} source frames")
    
    print("Loading edited keyframes...")
    edited_keyframes = []
    for idx in keyframe_indices:
        keyframe_path = os.path.join(edited_keyframes_dir, f"frame_{idx:04d}.png")
        if not os.path.exists(keyframe_path):
            raise FileNotFoundError(f"Edited keyframe not found: {keyframe_path}")
        edited_keyframes.append(Image.open(keyframe_path).convert('RGB'))
    print(f"Loaded {len(edited_keyframes)} edited keyframes")
    
    # ========== Run Video Editing ==========
    print("\nStarting video editing with Flow Matching...")
    print(f"Parameters: alpha={alpha}, beta={beta}, cfg_scale={cfg_scale}")
    print(f"Keyframe indices: {keyframe_indices}")
    
    edited_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        source_video=source_frames,
        edited_keyframes=edited_keyframes,
        keyframe_indices=keyframe_indices,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        alpha=alpha,
        beta=beta,
        tiled=True,
        verbose=True,  # Print monitoring metrics
    )
    
    # ========== Save Results ==========
    save_video_frames(edited_frames, output_dir)
    print(f"\nVideo editing complete! Results saved to {output_dir}")
    
    # Optional: Create video file
    try:
        import cv2
        video_path = os.path.join(output_dir, "edited_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 24.0, (width, height))
        for frame in edited_frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"Video saved to {video_path}")
    except ImportError:
        print("Install opencv-python to save video file")

if __name__ == "__main__":
    main()
