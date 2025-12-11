from langgraph.graph import StateGraph, END
from .schema import State, FrameInfo, StyleAnalysis, KeyFrameStylingState
from src.config.manager import ConfigManager
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
import cv2
from PIL import Image
import os
from pathlib import Path
import json
from src.utils.colored_logger import init_default_logger, log_info, log_error, log_tool, log_save, log_warning
from src.utils.image_generation import image_generation_tool
from src.utils.multi_modal_utils import create_interleaved_multimodal_message, create_multimodal_message
from dotenv import load_dotenv
from src.config.model import AgentConfig, ModelConfig
from dataclasses import asdict
import subprocess
import glob

# Initialize logger
init_default_logger(__name__)

config_path = Path(__file__).parent / "config.yaml"
config = ConfigManager(config_path)
load_dotenv()

graph = StateGraph(State)

def get_agent_config(agent_name: str):
    return config.get_agent_config(agent_name)

def get_llm(agent_config: AgentConfig):
    model_config = asdict(agent_config.model)
    # Map 'gemini-2.5-flash' to provider format if needed, but init_chat_model handles many
    return init_chat_model(**model_config)

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

def _save_state_json(state: State, step_name: str):
    """Helper to save state to JSON for debugging/caching."""
    try:
        output_dir = state["output_dir"]
        if not output_dir:
            return

        # Ensure output directory exists
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Helper to serialize non-json objects
        def json_serial(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError (f"Type {type(obj)} not serializable")

        # Use step name for consistent naming
        file_path = output_dir / f"state_{step_name}.json"

        with open(file_path, 'w') as f:
            json.dump(state, f, default=json_serial, indent=2)

        log_save(f"Saved state to {file_path}")
        return file_path
    except Exception as e:
        log_error(f"Failed to save state json for {step_name}: {e}")

def _load_checkpoint(output_dir: Path, step_name: str) -> State | None:
    """Load checkpoint state from JSON file."""
    try:
        checkpoint_path = output_dir / f"state_{step_name}.json"
        with open(checkpoint_path, 'r') as f:
            state_dict = json.load(f)

        # Convert string paths back to Path objects
        if 'output_dir' in state_dict:
            state_dict['output_dir'] = Path(state_dict['output_dir'])

        log_info(f"Loaded checkpoint from {checkpoint_path}")
        return state_dict

    except Exception as e:
        log_error(f"Failed to load checkpoint {step_name}: {e}")
        return None

def _find_latest_checkpoint(output_dir: Path) -> tuple[str, State] | None:
    """Find the latest valid checkpoint in the output directory."""
    try:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.exists():
            return None

        # First check for error checkpoints (they might be more advanced than regular checkpoints)
        last_state = _load_checkpoint(output_dir, "last")
        if last_state:
            return last_state["current_node"], last_state

    except Exception as e:
        log_error(f"Error finding latest checkpoint: {e}")
        return None

async def init_context_node(state: State) -> State:
    """Initialize context: extract frames from video."""
    output_dir = state["output_dir"]

    # Check for existing checkpoints to resume from
    checkpoint_result = _find_latest_checkpoint(output_dir)
    if checkpoint_result:
        current_node, current_state = checkpoint_result
        log_info(f"Found checkpoint from {current_node}")
        return current_state

    video_path = state["video_url"]
    frames_dir = output_dir / "frames" / "source"
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Processing video: {video_path}")
    
    frames_info = []

    try:
        video_info = _get_video_info(video_path)
        fps = video_info["fps"]
        frame_count = video_info["frame_count"]
        width = video_info["width"]
        height = video_info["height"]
        duration = video_info["duration"]
        state['target_ratio'] = width / height
        state['target_fps'] = 1.0
        step = max(1, int(fps / state['target_fps']))

        log_info(f"Video FPS: {fps}, Extraction step: {step}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Unable to open video for frame extraction: {video_path}")

        saved_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(frame_rgb)
                frame_path = frames_dir / f"frame_{saved_count:04d}.png"
                frame_img.save(frame_path)
                log_save(f"Saved frame {saved_count:04d} to {frame_path}")
                frames_info.append(
                    {
                        "frame_index": saved_count,
                        "timestamp": frame_idx / fps,
                        "image_path": str(frame_path),
                        "stylized_image_path": None,
                    }
                )
                saved_count += 1

            frame_idx += 1
            if saved_count >= state.get("max_frames", 10):
                break

        cap.release()

        new_state = {
            **state,
            "video_metadata": {
                "fps": fps,
                "duration": duration,
                "width": width,
                "height": height,
                "frame_count": saved_count,
            },
            "frames": frames_info,
            "current_frame_index": 0,
            "processed_frames": [],
        }
        path = _save_state_json(new_state, "init_context")
        log_save(f"Save initial context state to {path}")
        new_state["current_node"] = "style_analysis"
        return new_state

    except Exception as e:
        log_error(f"Error initializing context: {e}")
        state['current_node'] = "init_context_node"
        return state

async def style_analysis_node(state: State) -> State:
    """Analyze the style reference."""
    # Check if style analysis already completed
    if (state.get("style_analysis") and
        state["style_analysis"].get("dominant_style_prefix") and
        state.get("consistent_style_prompt")):
        log_info("Style analysis already completed, skipping...")
        return state

    log_info("Analyzing style...")
    agent_config = get_agent_config("style_analysis_agent")
    llm = get_llm(agent_config)
    system_prompt = agent_config.prompt
    
    style_ref = state["style_reference"]
    content_parts = []
    
    # Construct message
    if style_ref.get("description"):
         content_parts.append({"type": "text", "text": f"Style Description: {style_ref['description']}"})
    
    if style_ref.get("image_path"):
         # Ensure image exists
         if os.path.exists(style_ref['image_path']):
            content_parts.append({"type": "image", "image": style_ref['image_path']})
         else:
             log_error(f"Style image not found: {style_ref['image_path']}")
    
    if not content_parts:
        # Fallback if no input
        fallback_state = {
             **state,
            "style_analysis": {"dominant_style_prefix": "Default Style", "specific_style_prefix": "Standard", "negative_prompt": ""},
            "consistent_style_prompt": "Standard style"
        }
        path = _save_state_json(fallback_state, "style_analysis")
        log_save(f"Save style analysis fallback state to {path}")
        return fallback_state

    message = create_interleaved_multimodal_message(content_parts)
    
    messages = [
        SystemMessage(content=system_prompt),
        message
    ]
    
    structured_llm = llm.with_structured_output(StyleAnalysis)
    
    try:
        # response should be the dictionary corresponding to StyleAnalysis
        analysis = await structured_llm.ainvoke(messages)
        
        # Construct consistent prompt
        consistent_prompt = f"{analysis['dominant_style_prefix']}. {analysis['specific_style_prefix']}"
        log_info(f"Style Analysis Result: {consistent_prompt}")
        
        new_state = {
            **state,
            "style_analysis": analysis,
            "consistent_style_prompt": consistent_prompt
        }
        path = _save_state_json(new_state, "style_analysis")
        log_save(f"Save style analysis state to {path}")
        new_state["current_node"] = "video_style"
        return new_state
        
    except Exception as e:
        log_error(f"Error in style analysis: {e}")
        # Fallback
        _save_state_json(state, "style_analysis_error")
        return state

async def video_style_node(state: State) -> State:
    """Style the current frame."""
    idx = state["current_frame_index"]
    frames = state["frames"]

    # Check if all frames are already processed
    if idx >= len(frames):
        log_info("All frames already processed, skipping video_style")
        return state

    # Validate that we have required style analysis
    if not state.get("style_analysis", {}).get("dominant_style_prefix"):
        log_error("Style analysis not completed, cannot proceed with video styling")
        raise ValueError("Missing style analysis results")
        
    current_frame = frames[idx]
    log_info(f"Styling frame {idx+1}/{len(frames)}: {Path(current_frame['image_path']).name}")
    
    agent_config = get_agent_config("video_style_agent")
    llm = get_llm(agent_config)
    system_prompt = agent_config.prompt
    
    # Inputs
    style_desc = state["consistent_style_prompt"]
    curr_image_path = current_frame["image_path"]
    inputs = [
        {"type": "text", "text": f"Target Style: {style_desc}"},
        {"type": "text", "text": "Current Content Frame:"},
        {"type": "image", "image": curr_image_path}
    ]
    
    # Add previous frame context if available
    if idx > 0:
        prev_frame = frames[idx-1]
        if prev_frame["stylized_image_path"] and os.path.exists(prev_frame["stylized_image_path"]):
            inputs.insert(1, {"type": "image", "image": prev_frame["stylized_image_path"]})
            inputs.insert(1, {"type": "text", "text": "Previous Stylized Frame:"})
        if prev_frame["styling_prompt"]:
            inputs.insert(1, {"type": "text", "text": "Previous Styling Prompt:"})
            inputs.insert(1, {"type": "text", "text": prev_frame["styling_prompt"]})

    if state.get("style_reference", {}).get("image_path"):
        style_image_path = state["style_reference"]["image_path"]
        if os.path.exists(style_image_path):
            inputs.insert(1, {"type": "image", "image": style_image_path})
            inputs.insert(1, {"type": "text", "text": "Style Reference Image:"})
        else:
            log_error(f"Style reference image not found: {style_image_path}")
    else:
        log_info("No style reference image provided")
    log_info(f"Inputs: {inputs}")
    message = create_interleaved_multimodal_message(inputs)
    
    # Get prompt from Agent
    messages = [
        SystemMessage(content=system_prompt),
        message
    ]
    
    try:
        response = await llm.ainvoke(messages)

        generated_prompt = response.content.strip()
        log_info(f"Generated prompt: {generated_prompt[:100]}...")
        
        # Call Image Generation Tool
        stylized_dir = state["output_dir"] / "frames" / "stylized"
        prompts_dir = state["output_dir"] / "prompts"
        stylized_dir.mkdir(parents=True, exist_ok=True)
        prompts_dir.mkdir(parents=True, exist_ok=True)
        # We combine the consistent style prompt with the specific frame prompt
        full_prompt = f"Stylize prompt: {generated_prompt} Negative prompt: {state['style_analysis']['negative_prompt']}"
        
        open(prompts_dir / f"frame_{idx:04d}.txt", "w").write(full_prompt)
        log_save(f"[{idx:04d}] Saved prompt to {prompts_dir / f'frame_{idx:04d}.txt'}")

        result_image = image_generation_tool(
            text_prompt=full_prompt,
            image_paths=[curr_image_path],
            model="gemini-2.5-flash-image" ,
            target_ratio=state["target_ratio"]
        )
        
        # Save result
        stylized_path = stylized_dir / f"stylized_{idx:04d}.png"
        result_image.save(stylized_path)
        log_save(f"[{idx:04d}] Saved stylized image to {stylized_path}")
        # Update state - need to create a copy of list to ensure immutability/correct updates in some contexts, but simple assign works for dict ref
        frames[idx]["stylized_image_path"] = str(stylized_path)
        frames[idx]["styling_prompt"] = generated_prompt

        new_state = {
            **state,
            "frames": frames,
            "current_frame_index": idx + 1,
            "processed_frames": state["processed_frames"] + [idx]
        }
        if idx + 1 < len(frames):
            new_state["current_node"] = "video_style"
        else:
            new_state["current_node"] = "combine_video"
        return new_state
        
    except Exception as e:
        log_error(f"Error styling frame {idx}: {e}")
        # Skip frame but continue
        _save_state_json(state, f"video_style_error_{idx:04d}")
        return state

# def should_continue(state: State):
#     if state["current_frame_index"] < len(state["frames"]):
#         return "continue"
#     return "end"

def combine_video_node(state: State) -> State:
    """把 stylized_0000.png, stylized_0001.png ... 合成高清视频（带音轨完美同步）"""
    # Check if video already exists
    try:
        output_dir = state["output_dir"]
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        final_video_path = output_dir / "video_with_audio.mp4"
        if final_video_path.exists() and state.get("final_video_path"):
            log_info(f"Final video already exists at {final_video_path}, skipping combine_video")
            return state

        log_info("正在合成最终视频...")

        output_dir = state["output_dir"]
        stylized_dir = output_dir / "frames" / "stylized"
        prompts_dir = output_dir / "prompts"          # 里面应该有一个 audio.wav 或 original_audio.wav
        final_video_path = output_dir / "video_with_audio.mp4"

        # 1. 先检查原始音频文件（常见几种可能名字）
        possible_audio_files = ["audio.wav", "original_audio.wav", "sound.wav", "prompt_audio.wav"]
        audio_path = None
        for name in possible_audio_files:
            candidate = prompts_dir / name
            if candidate.exists():
                audio_path = candidate
                break
        
        if audio_path is None:
            log_warning("未在 prompts 文件夹找到音频文件，将生成无声视频")
        
        frame_rate = state.get("target_fps", 1)  # 如果你前面存了 fps 就用，没存默认 24

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(frame_rate),
            "-i", str(stylized_dir / "stylized_%04d.png"),   # 关键！%04d 表示 0000、0001...
            "-vf", "format=yuv420p,fps=24",  # 强制输出 24fps + 兼容格式（可改成变量）
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",                  # 18~23 画质极好，推荐 18
            "-pix_fmt", "yuv420p",
        ]

        if audio_path:
            cmd += [
                "-i", str(audio_path),
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",                   # 自动按较短轨道结束（避免黑屏尾巴）
            ]
        else:
            cmd += ["-an"]  # 无音频

        cmd += [str(final_video_path)]

        # 执行
        log_info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log_error("ffmpeg failed to combine video!")
            log_error(result.stderr)
            raise RuntimeError("Failed to combine video!")
        else:
            log_save(f"Success! Final video saved to {final_video_path}")
            # 可选：把路径写回 state，方便后续节点使用
            state["final_video_path"] = str(final_video_path)
        state["current_node"] = "end"
        return state
    except Exception as e:
        log_error(f"Error combining video: {e}")
        return state

def router_node(state: State) -> State:
    """Router node to determine the next node to execute."""
    _save_state_json(state, "last")
    return state

def router_logic(state: State) -> State:
    """Router logic to determine the next node to execute."""
    log_info(f"Router to {state['current_node']}")
    if state["current_node"] == "init_context":
        return "init_context"
    elif state["current_node"] == "style_analysis":
        return "style_analysis"
    elif state["current_node"] == "video_style":
        return "video_style"
    elif state["current_node"] == "combine_video":
        return "combine_video"
    else:
        return "end"

# Build Graph
graph.add_node("init_context", init_context_node)
graph.add_node("style_analysis", style_analysis_node)
graph.add_node("router", router_node)
graph.add_node("video_style", video_style_node)
graph.add_node("combine_video", combine_video_node)
graph.set_entry_point("init_context")


graph.add_edge("init_context", "router")
graph.add_edge("style_analysis", "router")
graph.add_edge("video_style", "router")
graph.add_edge("combine_video", "router")

graph.add_conditional_edges("router", 
router_logic,
{
    "init_context": "init_context",
    "style_analysis": "style_analysis",
    "video_style": "video_style",
    "combine_video": "combine_video",
    "end": END
})

app = graph.compile()
