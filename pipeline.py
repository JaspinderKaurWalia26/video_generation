from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
import torch
import requests
from datetime import datetime
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Tavily API key for web search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Tavily search tool
search_tool = TavilySearch(
    max_results=8,
    topic="general",
    tavily_api_key=TAVILY_API_KEY
)

# Take user inputs
print("\n" + "=" * 50)
print("Corporate Instagram Reel Content Generator")
print("=" * 50)

company_name = input("\nEnter your company name: ").strip()
if not company_name:
    company_name = "Our Company"

user_topic = input("Enter your topic (e.g. Employee birthday): ").strip()
if not user_topic:
    print("No topic entered. Exiting.")
    exit(1)

extra_details = input("Enter extra details (e.g. employee name, date): ").strip()

print(f"\nGenerating content for: {company_name} | {user_topic} | {extra_details}")

# Fetch trending Instagram-related data using Tavily
query = f"Instagram reels corporate company {user_topic} India 2026"
print(f"\nSearching trends: {query}")

data = search_tool.invoke(query)

# Combine search results into a single text context
all_text = ""
for result in data.get('results', []):
    content = result.get('content', '').strip()
    title = result.get('title', '')
    if content:
        all_text += f"SOURCE: {title}\n{content}\n\n"

# Limit context size for LLM processing
all_text = all_text[:3000]

print("\nRAW DATA OUTPUT:\n")
print(all_text if all_text else "No data found!")

if not all_text.strip():
    print("No data found. Exiting.")
    exit(1)

# Check if Ollama server is running
print("\nChecking Ollama connection...")
try:
    requests.get("http://localhost:11434", timeout=5)
    print("Ollama is running!")
except Exception:
    print("Ollama is NOT running!")
    print("Run: ollama serve")
    print("Then: ollama run qwen2.5:3b")
    exit(1)

# Function to call Ollama model
def call_ollama(prompt, temperature=0.7, timeout=300):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 600,
            "top_p": 0.9,
            "num_ctx": 2048
        }
    }

    try:
        print("Waiting for Ollama response...")
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "")

    except requests.exceptions.Timeout:
        return "Error: Timeout"
    except requests.exceptions.ConnectionError:
        return "Error: Connection refused"
    except Exception as e:
        return f"Error: {str(e)}"

# Generate Instagram reel script using LLM
print("\nStep 1: Generating Reel script...")

script_prompt = f"""You are writing a real Instagram Reel script.

Company: {company_name}
Topic: {user_topic}
Details: {extra_details}

Trending data:
{all_text[:1500]}

Create a professional 30-second reel script with proper format.
"""

script_output = call_ollama(script_prompt)

print("\nREEL SCRIPT:\n")
print(script_output)

if script_output.startswith("Error:"):
    exit(1)

# Extract video prompt from generated script
video_prompt = ""
for line in script_output.split("\n"):
    if "VIDEO PROMPT:" in line.upper():
        video_prompt = line.split(":", 1)[1].strip()
        break

# Fallback prompt if missing
if not video_prompt:
    video_prompt = "Professional corporate office celebration, employees smiling, cinematic lighting"

# Limit prompt length for stability
if len(video_prompt) > 200:
    video_prompt = video_prompt[:200].rsplit(" ", 1)[0]

print("\nVideo Prompt:", video_prompt)

# Generate Instagram caption
print("\nStep 2: Generating caption...")

caption_prompt = f"""Write an Instagram caption for {company_name} about {user_topic}.
Details: {extra_details}

Rules:
No emojis
No placeholders
Professional tone
"""

caption_output = call_ollama(caption_prompt)

print("\nCAPTION:\n")
print(caption_output)

# Generate video using text-to-video model
print("\nStep 3: Generating video...")

def generate_video_t2v(prompt, num_frames=16, num_steps=25):
    try:
        # Load pretrained text-to-video model
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Use optimized scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # Enable memory optimization
        pipe.enable_model_cpu_offload()

        # Optional xformers acceleration
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # Generate frames
        output = pipe(
            prompt,
            num_inference_steps=num_steps,
            num_frames=num_frames,
        )

        raw_frames = output.frames

        # Handle different diffusers formats
        if isinstance(raw_frames[0], list):
            video_frames = raw_frames[0]
        else:
            video_frames = raw_frames

        # Save video with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"reel_{user_topic}_{timestamp}.mp4"

        return export_to_video(video_frames, video_filename, fps=8)

    except Exception as e:
        print("Video generation failed:", str(e))
        return None


video_file = generate_video_t2v(
    prompt=video_prompt,
    num_frames=16,
    num_steps=25
)

# Save final report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"reels_{company_name}_{user_topic}_{timestamp}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write("# Corporate Reels Content Plan\n\n")
    f.write(f"Company: {company_name}\n")
    f.write(f"Topic: {user_topic}\n")
    f.write(f"Details: {extra_details}\n\n")
    f.write("## Reel Script\n\n")
    f.write(script_output + "\n\n")
    f.write("## Video Prompt\n\n")
    f.write(video_prompt + "\n\n")
    f.write("## Caption\n\n")
    f.write(caption_output + "\n\n")

    if video_file:
        f.write(f"## Video File\n\n{video_file}\n")

# Final output
print("\nDone!")
print("Saved file:", filename)

if video_file:
    print("Video saved:", video_file)
else:
    print("Video generation failed.")