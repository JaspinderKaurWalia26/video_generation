import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="replicate",
    api_key=os.getenv("HF_API_KEY")
)

video = client.text_to_video(
    "A young man walking on the street",
    model="Wan-AI/Wan2.2-TI2V-5B",
)

with open("newtestoutput_video.mp4", "wb") as f:
    f.write(video)

print("Video saved as newtestoutput_video.mp4")