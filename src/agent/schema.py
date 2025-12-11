from typing import List, TypedDict, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field

class VideoMetadata(TypedDict):
    fps: float
    duration: float
    width: int
    height: int
    frame_count: int

class FrameInfo(TypedDict):
    frame_index: int
    timestamp: float
    image_path: str
    stylized_image_path: Optional[str]
    styling_prompt: Optional[str]

class StyleReference(TypedDict):
    image_path: Optional[str]
    description: Optional[str]

class KeyFrameStylingState(TypedDict):
    keyframe_index: int
    keyframe_style_prompt: str
    # generated_image_path: str # Added to track result

class StyleAnalysis(TypedDict):
    dominant_style_prefix: str
    specific_style_prefix: str
    negative_prompt: str

class State(TypedDict):
    video_url: str # Or local path
    output_dir: Path
    
    # Video info
    video_metadata: VideoMetadata
    frames: List[FrameInfo]
    
    # Style input
    style_reference: StyleReference
    target_ratio: float
    # Analysis result
    style_analysis: StyleAnalysis
    consistent_style_prompt: str
    
    # Processing state
    keyframe_styling_states: List[KeyFrameStylingState]
    current_frame_index: int 
    processed_frames: List[int] # Indices of processed frames
    max_frames: int = 10
    current_node: str
