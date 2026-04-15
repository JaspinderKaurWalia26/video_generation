from langchain_tavily import TavilySearch
import torch
import requests
from datetime import datetime
from diffusers.utils import export_to_video

# tavily api key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# initialize tavily
search_tool = TavilySearch(
    max_results=8,
    topic="general",
    tavily_api_key=TAVILY_API_KEY
)

# input section
print("\n" + "=" * 50)
print("Corporate Instagram Reel Content Generator")
print("=" * 50)

company_name = input("\nEnter your company name: ").strip() or "Our Company"
user_topic = input("Enter your topic (e.g. Employee birthday): ").strip()

if not user_topic:
    print("No topic entered. Exiting.")
    exit(1)

extra_details = input("Enter extra details: ").strip()

print(f"\nGenerating content for: {company_name} | {user_topic}")

# search trends
query = f"Instagram reels corporate company {user_topic} India 2026"
print(f"\nSearching trends: {query}")

data = search_tool.invoke(query)

all_text = ""
for result in data.get('results', []):
    content = result.get('content', '').strip()
    title = result.get('title', '')
    if content:
        all_text += f"{title}\n{content}\n\n"

all_text = all_text[:3000]

if not all_text.strip():
    print("No data found. Exiting.")
    exit(1)

# check ollama
print("\nChecking Ollama...")

try:
    requests.get("http://localhost:11434", timeout=5)
    print("Ollama running")
except:
    print("Ollama NOT running")
    print("Run: ollama serve")
    print("Then: ollama run qwen2.5:3b")
    exit(1)

# ollama function
def call_ollama(prompt, temperature=0.7):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        return response.json().get("response", "")
    except Exception as e:
        return f"Error: {str(e)}"

# script generation
print("\nGenerating Reel Script...")

script_prompt = f"""
Company: {company_name}
Topic: {user_topic}
Details: {extra_details}

Trending:
{all_text[:1500]}

Create a 30-second professional Instagram reel script.
Include a VIDEO PROMPT line.
"""

script_output = call_ollama(script_prompt)

print("\nSCRIPT:\n", script_output)

if script_output.startswith("Error"):
    exit(1)

# extract video prompt
video_prompt = ""
for line in script_output.split("\n"):
    if "VIDEO PROMPT" in line.upper():
        video_prompt = line.split(":", 1)[1].strip()
        break

if not video_prompt:
    video_prompt = "corporate office celebration, professional employees smiling, cinematic lighting"

video_prompt = video_prompt[:200]

print("\nVideo Prompt:", video_prompt)

# caption generation
print("\nGenerating Caption...")

caption_prompt = f"""
Write a professional Instagram caption for {company_name} about {user_topic}.
No emojis.
"""

caption_output = call_ollama(caption_prompt)

print("\nCAPTION:\n", caption_output)

# cpu video generation
def generate_video_t2v(prompt, num_steps=25):
    try:
        print("\nLoading CPU video model (first time slow)...")

        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float32
        )

        pipe.to("cpu")
        pipe.enable_attention_slicing()

        print("Generating video... (this may take several minutes)")

        output = pipe(prompt, num_inference_steps=num_steps)
        video_frames = output.frames[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = user_topic.replace(" ", "_")
        filename = f"reel_{safe_topic}_{timestamp}.mp4"

        export_to_video(video_frames, filename, fps=8)

        return filename

    except Exception as e:
        print("Video generation failed:", str(e))
        return None

# generate video
video_file = generate_video_t2v(video_prompt)

# save file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"reels_{company_name}_{user_topic}_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write("# Corporate Reels Content Plan\n\n")
    f.write(f"Company: {company_name}\n")
    f.write(f"Topic: {user_topic}\n\n")

    f.write("## Script\n\n")
    f.write(script_output + "\n\n")

    f.write("## Video Prompt\n\n")
    f.write(video_prompt + "\n\n")

    f.write("## Caption\n\n")
    f.write(caption_output + "\n\n")

    if video_file:
        f.write(f"## Video File\n{video_file}\n")


print("\nDONE")
print("Saved:", filename)

if video_file:
    print("Video saved:", video_file)
else:
    print("Video generation failed")