Agentic Video Generator (Groq + LangGraph + Streamlit)
=========================================================

A Streamlit app that turns a prompt into a short multi-scene video script, generates TTS narration, and renders an MP4 slideshow using placeholder visuals.

Important: Groq is an LLM API. It generates text (script + prompts), but it cannot generate images or video frames.This repo uses placeholder images so you still get a playable MP4. You can later plug in any image/video model as a tool node.

✨ Features
----------

*   Agentic flow via LangGraph: Generate → Validate → Retry (if needed) → Build Assets → Render MP4
    
*   Structured script output:
    
    *   Title + description
        
    *   5–7 scenes
        
    *   Each scene: narration (2–3 sentences) + image prompt
        
*   Narration audio using gTTS
    
*   MP4 slideshow rendering using MoviePy
    
*   Streamlit UI:
    
    *   Tabs: Video / Script / Prompts / Debug
        
    *   Recent runs list + downloads
        
    *   Optional debug view (raw JSON)
        

🧱 Repo Structure
-----------------

groq-agentic-video/app.pyrequirements.txtREADME.md.gitignore.env.examplevideo\_run/ # created at runtime (ignored)

✅ Prerequisites
---------------

*   Python 3.10+ recommended
    
*   A Groq API key
    
*   Internet access (Groq + gTTS)
    

If you see: 403 Access denied. Please check your network settings.Your network/VPN/proxy may be blocking Groq.

🚀 Quickstart
-------------

### 1) Clone

git clone https://github.com/amey-ghate/Agentic-Video-Generator.git

cd groq-agentic-video

### 2) Create & activate a virtual environment

Windows (PowerShell)python -m venv venvvenv\\Scripts\\Activate.ps1

Windows (CMD)python -m venv venvvenv\\Scripts\\activate.bat

macOS / Linuxpython3 -m venv venvsource venv/bin/activate

### 3) Install dependencies

pip install -r requirements.txt

### 4) Set environment variables (.env)

GROQ\_API\_KEY=YOUR\_GROQ\_KEY\_HEREGROQ\_MODEL=llama-3.3-70b-versatile

Never commit .env (already in .gitignore).

### 5) Run the Streamlit app

streamlit run app.py

🛠️ Configuration
-----------------

### Environment variables

GROQ\_API\_KEY (required): your Groq API keyGROQ\_MODEL (required): model name (default: llama-3.3-70b-versatile)

### Script constraints (in code)

*   Scenes: 5–7
    
*   Narration: 2–3 sentences per scene
    
*   Target: < ~60 seconds (best effort)
    

🧩 How the “Agentic” part works
-------------------------------

LangGraph runs a state machine with conditional routing:

1.  Generate Script (Groq) → returns JSON
    
2.  Validate (Pydantic + business rules)
    
3.  If validation fails: Retry up to MAX\_RETRIES\_SCRIPT
    
4.  Make Assets
    
    *   placeholder images (PIL)
        
    *   narration mp3 (gTTS)
        
5.  Create Video
    
    *   stitch images + audio → MP4 (MoviePy)
        
6.  Cleanup temp files (optional)
    

🖼️ Upgrading to real AI visuals (recommended)
----------------------------------------------

Right now frames are placeholders. To generate real images, replace the placeholder step with any of:

*   Local: ComfyUI / Automatic1111 (Stable Diffusion) API endpoint
    
*   Hosted: Replicate / Stability / RunPod / etc.
    

Where to integrate:

*   Replace make\_placeholder\_image(...) in the asset stage with an image generation function.
