"""
Step 1: Generate Styled Keyframes with Agent
使用 agent 生成风格化的关键帧，并保存所有必要的中间结果
支持 Ctrl+C 中断并保存当前进度
"""
import sys
sys.path.insert(0, ".")

import asyncio
from pathlib import Path
from PIL import Image
import json
import datetime
import shutil
import signal
import glob

# Agent imports
from src.agent.graph import app as agent_app
import cv2

# 全局变量用于保存中断时的状态
_interrupted = False
_current_output_dir = None
_current_state = None

def _get_video_info(video_path: Path | str) -> dict[str, float | int]:
    """
    Extracts basic metadata from the video using OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = frame_count / fps if fps else 0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }

async def generate_styled_keyframes(
    video_path: Path,
    style_reference: dict,
    output_dir: Path,
    max_frames: int = 10,
    target_fps: float = 1.0,
    run_agent=False,
):
    """
    使用 agent 生成风格化的关键帧
    
    保存内容：
    - styled keyframes (PNG images)
    - keyframe_info.json (包含所有必要信息)
    - style_reference.png (风格参考图)
    - metadata.json (agent 输出的元数据)
    """
    print("\n" + "="*60)
    print("STEP 1: Generating Styled Keyframes with Agent")
    print("="*60)
    video_info = _get_video_info(video_path)
    source_fps = video_info["fps"]
    if run_agent:
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制风格参考图
        if style_reference.get("image_path"):
            shutil.copy(
                style_reference["image_path"],
                output_dir / "style_reference.png"
            )
            print(f"Style reference copied to {output_dir / 'style_reference.png'}")
        
        # 准备 agent state
        initial_state = {
            "video_url": str(video_path),
            "style_reference": style_reference,
            "output_dir": output_dir,
            "max_frames": max_frames,
            "target_fps": target_fps,
            "frames": [],
            "current_frame_index": 0,
            "processed_frames": [],
            "video_metadata": {},
            "edit_analysis": {},
            "consistent_edit_prompt": "",
            "current_node": "init_context",
        }
        
        # 运行 agent workflow
        print("Running agent workflow...")
        try:
            final_state = await agent_app.ainvoke(initial_state)
            print("Agent finished successfully")
        except Exception as e:
            print(f"Agent failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("Not run agent, just generated meta info.")
    
    state_last_path = output_dir / "state_last.json"
    with open(state_last_path, "r") as f:
        state_last = json.load(f)
    frames = state_last["frames"]
    generated_frames = []
    keyframe_idx = []
    for frame in frames:
        if frame["stylized_image_path"] is not None:
            generated_frames.append(frame["stylized_image_path"])
            keyframe_idx.append(frame["timestamp"])
    
    # 保存关键帧信息（用于第二步）
    keyframe_data = {
        "video_path": str(video_path),
        "style_reference": style_reference,
        "output_dir": str(output_dir),
        "keyframe_timestamp": keyframe_idx,
        "keyframe_fps": target_fps,
        "source_fps": source_fps,
        "generated_frames": generated_frames,
        "num_keyframes": len(generated_frames),
        "consistent_edit_prompt": state_last["consistent_edit_prompt"],
        "negative_prompt": state_last.get('edit_analysis', {}).get('negative_prompt', "blurry, low quality, inconsistent style"),
    }
    
    keyframe_info_path = output_dir / "keyframe_info.json"
    with open(keyframe_info_path, "w") as f:
        json.dump(keyframe_data, f, indent=2)
    print(f"Keyframe info saved to {keyframe_info_path}")
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    print("\nNext step: Run step2_video_editing.py with this output directory")
    
    return output_dir


async def main():
    """
    主函数：配置参数并运行关键帧生成
    """
    
    # ========== 配置参数 ==========
    video_path = Path("Ditto-1M/tests/youtube/2.mp4")
    style_image_path = Path("Ditto-1M/style/12.jpg")
    
    # 检查文件是否存在
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return
    if not style_image_path.exists():
        print(f"Error: Style image not found: {style_image_path}")
        return
    
    # 创建输出目录（带时间戳）
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = video_path.stem
    # output_dir = Path(f"outputs/video_editing/{video_name}/{timestamp}")
    output_dir = Path(f"outputs/video_editing/{video_name}/20251212_112926")
    
    print(f"Video: {video_path}")
    print(f"Style: {style_image_path}")
    print(f"Output: {output_dir}")
    

    # 风格参考配置
    style_reference = {
        "image_path": str(style_image_path),
        "description": "style transfer as the following image"
    }
    
    # Agent 参数
    max_keyframes = 10  # 生成的关键帧数量
    target_fps = 1.0    # 关键帧采样率
    
   
    # ========== 运行关键帧生成 ==========
    await generate_styled_keyframes(

        video_path=video_path,
        style_reference=style_reference,
        output_dir=output_dir,
        max_frames=max_keyframes,
        target_fps=target_fps,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",'-v', type=str, default="Ditto-1M/tests/youtube/2.mp4")
    parser.add_argument("--style_image_path", '-s', type=str, default="Ditto-1M/style/12.jpg")
    parser.add_argument("--output_dir", '-o', type=str, default="outputs/video_editing/2/20251212_112926")
    parser.add_argument("--max_keyframes", '-N', type=int, default=10)
    parser.add_argument("--target_fps", '-f', type=float, default=1.0)
    args = parser.parse_args()
    video_path = Path(args.video_path)
    style_image_path = Path(args.style_image_path)
    asyncio.run(main())
