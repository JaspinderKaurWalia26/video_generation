import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32)

# Force the model to use the CPU
pipe.to("cpu")

# Optimize memory usage for CPU
pipe.enable_attention_slicing()

prompt = "A majestic sunflower swaying in the morning mist, cinematic lighting"
video_frames = pipe(prompt, num_inference_steps=25).frames[0]

# Save the result
export_to_video(video_frames, "output_video.mp4")