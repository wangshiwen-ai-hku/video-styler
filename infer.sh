#!/bin/bash
python inference/infer_ditto.py \
--lora_path ./models/ditto_global.safetensors \
--num_frames 73 \
--device_id 0 \
--input_video ./Ditto-1M/tests/mini_test_videos/0fb4e6607c7061b57fe4396b5872675a.mp4 \
--output_video results/scene_02_lego.mp4 \
--prompt "Make it the LEGO style."