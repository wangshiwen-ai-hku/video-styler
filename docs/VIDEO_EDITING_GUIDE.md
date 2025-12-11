# Flow Matching-based Video Editing with WAN-I2V

## Overview

This implementation provides a **free** video editing framework based on Flow Matching, combining:

1. **Agent-based Keyframe Styling**: Uses multimodal LLM (Gemini) + image generation to create consistently styled keyframes
2. **Flow Matching Video Editing**: Propagates keyframe edits to all frames using WAN-I2V with velocity field correction

## Key Features

### 🎯 Core Algorithm

The system implements the mathematical framework from `tech_design.md`:

1. **Coupled Noise Initialization**: Edited keyframes and original video share identical noise at corresponding positions
2. **Shared Positional Encoding**: Keyframes maintain same RoPE indices as their original positions
3. **Velocity Field Correction**: Monitors and corrects denoising trajectories to ensure consistency

### 📐 Mathematical Foundation

```python
# Velocity difference at keyframes
Δv_k = (ε - z_k) / t - (ε - z_i_k) / t = (z_i_k - z_k) / t

# Consistency residual
r_k = Δv_k - (z_k^t - z_i_k^t) / t

# Velocity correction
v_k^corrected = v_k + α * r_k
```

See `math_verification.py` for a toy example demonstrating convergence.

## Architecture

### Pipeline Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Video + Style Reference             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Agent-based Keyframe Generation                    │
│  - Extract keyframes from video                              │
│  - Analyze style reference (Gemini)                          │
│  - Generate styled keyframes (Image Generation Tool)         │
│  - Maintain temporal consistency                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Flow Matching Video Editing (WAN-I2V)              │
│  - Encode original video → z_main                            │
│  - Encode styled keyframes → z_edit                          │
│  - Initialize with coupled noise                             │
│  - Denoise with velocity correction                          │
│  - Decode final video                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Edited Video Output                       │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
diffsynth/
├── models/
│   └── wan_video_dit.py          # Modified to support custom RoPE indices
├── pipelines/
│   ├── wan_video.py               # Base WAN pipeline
│   └── wan_video_editor.py        # NEW: Flow Matching editor
│
src/
├── agent/
│   ├── graph.py                   # Agent workflow for keyframe styling
│   └── schema.py                  # State definitions
│
inference/
└── video_editing_with_agent.py   # Integrated pipeline
│
examples/
└── wan_video_editing_example.py  # Standalone editing example
│
math_verification.py               # Mathematical proof of concept
tech_design.md                     # Technical design document
```

## Installation

```bash
# Install dependencies
pip install torch torchvision
pip install diffsynth-studio
pip install langgraph langchain-google-genai
pip install opencv-python pillow einops

# Set up API keys
export GOOGLE_API_KEY="your-gemini-api-key"
```

## Usage

### Option 1: Integrated Pipeline (Agent + WAN)

```python
from inference.video_editing_with_agent import IntegratedVideoEditor
import asyncio

async def main():
    editor = IntegratedVideoEditor(wan_model_size="1.3B")
    
    # Step 1: Generate styled keyframes with agent
    styled_keyframes, keyframe_indices, metadata = await editor.generate_styled_keyframes(
        video_path="input.mp4",
        style_reference={
            "image_path": "style.jpg",
            "description": "Anime style with vibrant colors"
        },
        output_dir="outputs/agent",
        max_frames=10,
    )
    
    # Step 2: Load WAN pipeline
    editor.load_wan_pipeline()
    
    # Step 3: Run video editing
    source_frames = editor.load_source_video("input.mp4", num_frames=81)
    edited_frames = editor.run_video_editing(
        source_frames=source_frames,
        styled_keyframes=styled_keyframes,
        keyframe_indices=keyframe_indices,
        prompt="Consistent anime style video",
        alpha=10.0,  # Correction strength
    )
    
    # Step 4: Save results
    editor.save_results(edited_frames, "outputs/final")

asyncio.run(main())
```

### Option 2: Standalone WAN Editing

If you already have styled keyframes:

```python
from diffsynth import ModelManager, WanVideoEditorPipeline
from PIL import Image

# Load pipeline
model_manager = ModelManager(model_id_list=["models/WAN-1.3B"])
pipe = WanVideoEditorPipeline.from_model_manager(model_manager)

# Load data
source_frames = [Image.open(f"source/frame_{i:04d}.png") for i in range(81)]
edited_keyframes = [Image.open(f"edited/frame_{i:04d}.png") for i in [0, 20, 40, 60, 80]]
keyframe_indices = [0, 20, 40, 60, 80]

# Run editing
edited_frames = pipe(
    prompt="Cinematic video with consistent style",
    source_video=source_frames,
    edited_keyframes=edited_keyframes,
    keyframe_indices=keyframe_indices,
    alpha=10.0,
    num_inference_steps=50,
)
```

## Parameters

### Agent Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_frames` | Number of keyframes to style | 10 |
| `target_fps` | Sampling rate for keyframes | 1.0 |
| `style_reference` | Dict with `image_path` and `description` | Required |

### WAN Editing Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `alpha` | Velocity correction strength | 10.0 | 1.0-20.0 |
| `beta` | Correction for edited keyframes | 0.0 | 0.0-1.0 |
| `cfg_scale` | Classifier-free guidance | 5.0 | 1.0-10.0 |
| `num_inference_steps` | Denoising steps | 50 | 20-100 |

### Parameter Tuning Guide

- **`alpha`**: Controls consistency enforcement
  - Higher (15-20): Stronger consistency, may reduce quality
  - Lower (5-10): Weaker consistency, better quality
  - Recommended: 10.0

- **`beta`**: Usually keep at 0.0 to preserve DMI quality
  - Only increase if keyframes need adjustment

- **`cfg_scale`**: Standard guidance parameter
  - Higher: Stronger prompt adherence
  - Lower: More creative freedom

## Model Support

### Supported WAN Models

| Model | Size | Resolution | Speed | VRAM |
|-------|------|------------|-------|------|
| WAN-1.3B | 1.3B | 480p-720p | Fast | ~8GB |
| WAN-14B | 14B | 720p-1080p | Slow | ~24GB |

Both models support the same API and features.

### Model Loading

```python
# 1.3B model (faster, lower VRAM)
editor = IntegratedVideoEditor(wan_model_size="1.3B")

# 14B model (better quality, higher VRAM)
editor = IntegratedVideoEditor(wan_model_size="14B")
editor.load_wan_pipeline()
editor.pipe.enable_vram_management()  # Enable for 14B
```

## Monitoring Metrics

The pipeline outputs three key metrics during denoising:

1. **`r_k_norm`**: Consistency residual magnitude
   - Measures trajectory divergence
   - Lower is better (< 0.01 is good)

2. **`v_diff_norm`**: Velocity difference
   - Measures velocity field alignment
   - Should decrease over time

3. **`delta_v_norm`**: Latent difference
   - Measures current state divergence
   - Should converge to small value

Example output:
```
Step 0: r_k=0.123456, v_diff=0.234567, Δv=0.345678
Step 10: r_k=0.045678, v_diff=0.123456, Δv=0.234567
Step 20: r_k=0.012345, v_diff=0.056789, Δv=0.123456
...
```

## Technical Details

### Keyframe Encoding Strategy

**Critical Design Decision**: Keyframes are encoded **independently** as single-frame videos:

```python
# WRONG: Encoding all keyframes together with 3D convolution
keyframes_tensor = torch.stack(keyframes, dim=2)  # (C, K, H, W)
z_edit = vae.encode(keyframes_tensor)  # ❌ 3D conv assumes temporal continuity

# CORRECT: Encode each keyframe independently
z_edit_list = []
for keyframe in keyframes:
    keyframe_tensor = keyframe.unsqueeze(2)  # (C, 1, H, W)
    z_keyframe = vae.encode(keyframe_tensor)  # ✓ Treat as 1-frame video
    z_edit_list.append(z_keyframe)
z_edit = torch.cat(z_edit_list, dim=2)  # (B, C, K, H, W)
```

**Rationale**: Keyframes are temporally distant (e.g., frames 0, 20, 40, 60, 80). Using 3D convolution would incorrectly assume temporal continuity between them.

### Coupled Noise Implementation

```python
def prepare_coupled_noise(latent_shape, keyframe_indices, seed):
    # Generate noise for full video
    noise_main = torch.randn(latent_shape, generator=torch.Generator().manual_seed(seed))
    
    # Extract SAME noise for keyframes
    noise_edit = noise_main[:, :, keyframe_indices, :, :].clone()
    
    return noise_main, noise_edit
```

### RoPE Index Mapping

```python
def construct_rope_ids(total_frames, keyframe_indices):
    # Original: [0, 1, 2, ..., T-1]
    ids_main = torch.arange(total_frames)
    
    # Keyframes keep original indices: [0, 24, 48, ...] NOT [0, 1, 2, ...]
    ids_edit = torch.tensor(keyframe_indices)
    
    # Concatenate: [0,1,2,...,T-1, 0,24,48,...]
    return torch.cat([ids_main, ids_edit])
```

### Velocity Correction

```python
def compute_velocity_correction(z_main, z_edit, v_main, v_edit, keyframe_indices, dt, alpha):
    # Extract at keyframe positions
    v_main_at_keys = v_main[:, :, keyframe_indices, :, :]
    z_diff = z_main[:, :, keyframe_indices, :, :] - z_edit
    v_diff = v_main_at_keys - v_edit
    
    # Consistency residual
    r_k = z_diff - v_diff * dt
    
    # Apply correction
    v_main[:, :, keyframe_indices, :, :] += alpha * r_k
    
    return v_main, v_edit
```

## Extending to Other I2V Models

The framework is designed to be model-agnostic. To adapt to other I2V models:

1. **Implement coupled noise initialization**
2. **Support custom positional encoding**
3. **Add velocity correction in denoising loop**

Example for a new model:

```python
class NewModelEditorPipeline(BaseI2VPipeline):
    def prepare_coupled_noise(self, ...):
        # Same logic as WAN
        pass
    
    def construct_rope_ids(self, ...):
        # Adapt to model's positional encoding
        pass
    
    def compute_velocity_correction(self, ...):
        # Same mathematical formula
        pass
```

## Troubleshooting

### Common Issues

1. **High `r_k` values (> 0.1)**
   - Increase `alpha` (try 15-20)
   - Reduce `cfg_scale`
   - Check keyframe quality

2. **Inconsistent style**
   - Increase number of keyframes
   - Improve keyframe styling quality
   - Adjust agent prompts

3. **CUDA OOM**
   - Use 1.3B model instead of 14B
   - Enable `tiled=True`
   - Reduce `num_frames`
   - Enable VRAM management

4. **Slow generation**
   - Reduce `num_inference_steps` (try 30)
   - Use smaller model (1.3B)
   - Enable mixed precision

## Citation

If you use this work, please cite:

```bibtex
@software{wan_video_editor_2024,
  title={Flow Matching-based Video Editing with WAN-I2V},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

This project is licensed under the same license as the base WAN-I2V model.

## Acknowledgments

- WAN-I2V team for the base model
- Flow Matching framework
- LangGraph for agent orchestration
