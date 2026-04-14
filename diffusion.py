from diffusers import StableDiffusionPipeline
import torch
import cv2
import numpy as np

# Load model 
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe = pipe.to("cpu")
prompt = "A futuristic cityscape at sunset"
frames = []
#Generate frames
print("Generating frames")
for i in range(10):
    image = pipe(prompt, num_inference_steps=20).images[0]
    frames.append(image)

print("Frames generated!")


# Save frames and convert to OpenCV format
frame_list = []
for i, frame in enumerate(frames):
    img = np.array(frame) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

    cv2.imwrite(f"frame_{i}.png", img)
    frame_list.append(img)

print("Frames saved!")

#  Create video
height, width, layers = frame_list[0].shape
frame_size = (width, height)

out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    5,  # FPS
    frame_size
)

for frame in frame_list:
    out.write(frame)

out.release()

print("Video created: output_video.mp4")