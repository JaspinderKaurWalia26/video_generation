from diffusers import StableDiffusionPipeline
import torch
import cv2
import numpy as np

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to("cpu")

prompt = "An astronaut riding a horse on the moon, cinematic lighting, ultra realistic"

frames = []

print("Generating smooth frames...")

# Fix seed for consistency
generator = torch.manual_seed(42)

for i in range(8):
    image = pipe(
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator
    ).images[0]

    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frames.append(frame)

# Add smooth transition (frame blending)
smooth_frames = []

for i in range(len(frames)-1):
    smooth_frames.append(frames[i])
    
    # create intermediate frame
    blend = cv2.addWeighted(frames[i], 0.5, frames[i+1], 0.5, 0)
    smooth_frames.append(blend)

smooth_frames.append(frames[-1])

# Save video
height, width, _ = smooth_frames[0].shape

out = cv2.VideoWriter(
    "smooth_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    3,
    (width, height)
)

for frame in smooth_frames:
    out.write(frame)

out.release()

print(" Better quality video saved!")