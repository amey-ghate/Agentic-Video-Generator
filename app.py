import os
import json
import time
import uuid
from dataclasses import asdict
from typing import List, Optional, TypedDict, Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from groq import Groq

from gtts import gTTS
from PIL import Image, ImageDraw

from langgraph.graph import StateGraph, START, END
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="Agentic Video Generator",
    page_icon="🎬",
    layout="wide",
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MIN_SCENES = 5
MAX_SCENES = 7
MAX_RETRIES_SCRIPT = 2

WORKDIR = os.path.abspath("./video_run")
os.makedirs(WORKDIR, exist_ok=True)


# -------------------------
# Schema
# -------------------------
class Scene(BaseModel):
    narration: str = Field(..., description="2-3 short sentences. Clear spoken narration.")
    image_prompt: str = Field(..., description="Visual, descriptive prompt for imagery.")


class VideoScript(BaseModel):
    title: str
    description: str
    scenes: List[Scene]


# -------------------------
# LangGraph State
# -------------------------
class GraphState(TypedDict, total=False):
    user_prompt: str

    script_raw: str
    script: VideoScript
    validation_errors: List[str]
    script_attempt: int

    image_paths: List[str]
    audio_paths: List[str]
    video_path: str

    run_id: str


# -------------------------
# Helpers
# -------------------------
def groq_chat(messages, temperature=0.4) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY in .env")

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def make_placeholder_image(text: str, out_path: str, size=(1280, 720)) -> None:
    img = Image.new("RGB", size, color=(25, 25, 28))
    draw = ImageDraw.Draw(img)

    def wrap(s: str, width: int = 52, max_lines: int = 11) -> List[str]:
        words = s.split()
        lines, line = [], ""
        for w in words:
            if len(line) + len(w) + 1 <= width:
                line = (line + " " + w).strip()
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines[:max_lines]

    header = "PLACEHOLDER VISUAL (plug in an image model later)"
    lines = [header, ""] + wrap(text)

    # Simple “card” look
    pad = 48
    card_w = size[0] - pad * 2
    card_h = size[1] - pad * 2
    draw.rounded_rectangle(
        (pad, pad, pad + card_w, pad + card_h),
        radius=24,
        fill=(35, 35, 40),
        outline=(70, 70, 80),
        width=2,
    )

    y = pad + 28
    x = pad + 28
    for i, ln in enumerate(lines):
        fill = (240, 240, 240) if i == 0 else (220, 220, 220)
        draw.text((x, y), ln, fill=fill)
        y += 52 if i == 0 else 48

    img.save(out_path)


def estimate_total_duration_sec(script: VideoScript) -> float:
    # very rough speaking rate: ~150 wpm = 2.5 wps
    total_words = sum(len(s.narration.split()) for s in script.scenes)
    return max(5.0, total_words / 2.5)


def script_to_json(script: VideoScript) -> str:
    data = {
        "title": script.title,
        "description": script.description,
        "scenes": [{"narration": s.narration, "image_prompt": s.image_prompt} for s in script.scenes],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# -------------------------
# LangGraph nodes
# -------------------------
def node_init(state: GraphState) -> GraphState:
    state["run_id"] = state.get("run_id") or str(uuid.uuid4())[:8]
    state["script_attempt"] = state.get("script_attempt", 0)
    state["validation_errors"] = []
    return state


def node_generate_script(state: GraphState) -> GraphState:
    user_prompt = state["user_prompt"]
    attempt = state.get("script_attempt", 0)

    system = "You are a screenwriter assistant. Output ONLY valid JSON. No markdown. No extra text."
    user = f"""
Create a short video script based on this prompt:
"{user_prompt}"

Constraints:
- Total video under 60 seconds.
- {MIN_SCENES}-{MAX_SCENES} scenes.
- Each scene narration: 2-3 short sentences, easy to speak.
- Each image_prompt: very visual, descriptive, not overly long.

Return JSON exactly in this schema:
{{
  "title": "...",
  "description": "...",
  "scenes": [
    {{
      "narration": "...",
      "image_prompt": "..."
    }}
  ]
}}

Retry attempt number: {attempt}
Prior validation errors (if any): {state.get("validation_errors", [])}
""".strip()

    raw = groq_chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.5,
    )
    state["script_raw"] = raw
    state["script_attempt"] = attempt + 1
    return state


def node_validate_script(state: GraphState) -> GraphState:
    raw = state.get("script_raw", "")
    errors: List[str] = []

    try:
        data = json.loads(raw)
    except Exception as e:
        errors.append(f"JSON parse error: {e}")
        state["validation_errors"] = errors
        return state

    try:
        script = VideoScript.model_validate(data)
    except ValidationError as e:
        errors.append(f"Schema validation error: {e}")
        state["validation_errors"] = errors
        return state

    n = len(script.scenes)
    if not (MIN_SCENES <= n <= MAX_SCENES):
        errors.append(f"Scene count must be {MIN_SCENES}-{MAX_SCENES}, got {n}.")

    for i, sc in enumerate(script.scenes, start=1):
        if len(sc.narration.strip()) < 20:
            errors.append(f"Scene {i} narration too short.")
        if len(sc.image_prompt.strip()) < 20:
            errors.append(f"Scene {i} image_prompt too short.")

    state["validation_errors"] = errors
    if not errors:
        state["script"] = script
    return state


def route_after_validation(state: GraphState) -> Literal["retry_script", "make_assets"]:
    if state.get("validation_errors"):
        if state.get("script_attempt", 0) <= MAX_RETRIES_SCRIPT:
            return "retry_script"
        return "make_assets"
    return "make_assets"


def node_make_assets(state: GraphState) -> GraphState:
    script: Optional[VideoScript] = state.get("script")
    if not script:
        script = VideoScript(
            title="Untitled",
            description="Fallback script (validation failed).",
            scenes=[Scene(narration=state["user_prompt"], image_prompt=state["user_prompt"])],
        )
        state["script"] = script

    image_paths: List[str] = []
    audio_paths: List[str] = []

    for i, sc in enumerate(script.scenes, start=1):
        img_path = os.path.join(WORKDIR, f"{state['run_id']}_scene_{i}.png")
        make_placeholder_image(f"Scene {i}\n\n{sc.image_prompt}", img_path)
        image_paths.append(img_path)

        aud_path = os.path.join(WORKDIR, f"{state['run_id']}_narration_{i}.mp3")
        try:
            gTTS(sc.narration).save(aud_path)
        except Exception:
            pass
        audio_paths.append(aud_path)
        time.sleep(0.05)

    state["image_paths"] = image_paths
    state["audio_paths"] = audio_paths
    return state


def node_create_video(state: GraphState) -> GraphState:
    image_paths = state.get("image_paths", [])
    audio_paths = state.get("audio_paths", [])

    clips = []
    for img, aud in zip(image_paths, audio_paths):
        duration = 5.0
        audio_clip = None

        if aud and os.path.exists(aud):
            audio_clip = AudioFileClip(aud)
            duration = max(2.0, float(audio_clip.duration) + 0.25)

        img_clip = ImageClip(img).set_duration(duration)
        if audio_clip is not None:
            img_clip = img_clip.set_audio(audio_clip)

        clips.append(img_clip)

    if not clips:
        raise RuntimeError("No clips generated.")

    final = concatenate_videoclips(clips, method="compose")
    out_path = os.path.join(WORKDIR, f"{state['run_id']}_final.mp4")

    final.write_videofile(
        out_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="ultrafast",
        verbose=False,
        logger=None,
    )

    state["video_path"] = out_path
    return state


def node_cleanup(state: GraphState) -> GraphState:
    # Keep final MP4; delete intermediates
    for p in state.get("image_paths", []):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    for p in state.get("audio_paths", []):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    return state


@st.cache_resource(show_spinner=False)
def build_graph():
    g = StateGraph(GraphState)

    g.add_node("init", node_init)
    g.add_node("generate_script", node_generate_script)
    g.add_node("validate_script", node_validate_script)
    g.add_node("make_assets", node_make_assets)
    g.add_node("create_video", node_create_video)
    g.add_node("cleanup", node_cleanup)

    g.add_edge(START, "init")
    g.add_edge("init", "generate_script")
    g.add_edge("generate_script", "validate_script")

    g.add_conditional_edges(
        "validate_script",
        route_after_validation,
        {"retry_script": "generate_script", "make_assets": "make_assets"},
    )

    g.add_edge("make_assets", "create_video")
    g.add_edge("create_video", "cleanup")
    g.add_edge("cleanup", END)

    return g.compile()


# -------------------------
# UI
# -------------------------
st.markdown(
    """
<style>
/* Make it feel more "app-like" */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); padding: 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); }
div[data-testid="stExpander"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎬 Agentic Video Generator")
st.caption("Groq generates the script + prompts. Visuals are placeholders (no image model hooked up yet).")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts with run metadata


with st.sidebar:
    st.subheader("⚙️ Settings")

    st.text_input("Model", value=GROQ_MODEL, disabled=True)

    if GROQ_API_KEY:
        st.success("GROQ_API_KEY detected")
    else:
        st.error("Missing GROQ_API_KEY")

    st.divider()
    st.subheader("ℹ️ What this app does")
    st.write(
        "- Agentic script generation (validate + retry)\n"
        "- Generates narration with gTTS\n"
        "- Creates a slideshow MP4\n\n"
        "To get real AI visuals, plug in an image model API later."
    )

    st.divider()
    keep_video_files = st.checkbox("Keep generated MP4 files", value=True)
    show_debug = st.checkbox("Show debug (raw JSON)", value=False)


left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("🧠 Prompt")
    prompt = st.text_area(
        "Describe your story / idea",
        height=160,
        placeholder="e.g., A cyberpunk detective solves a mystery in a neon-lit city...",
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)
    with c2:
        clear_btn = st.button("🧹 Clear", use_container_width=True)
    with c3:
        demo_btn = st.button("✨ Demo prompt", use_container_width=True)

    if clear_btn:
        st.session_state.history = []
        st.rerun()

    if demo_btn and not prompt.strip():
        st.session_state["demo_prompt"] = "A short uplifting story about a lonely astronaut who finds a glowing garden on a distant moon."
        st.rerun()

    if "demo_prompt" in st.session_state and not prompt.strip():
        prompt = st.session_state.pop("demo_prompt")

    st.subheader("📦 Output")
    tabs = st.tabs(["Video", "Script", "Prompts", "Debug"])

with right:
    st.subheader("📊 Run summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Scenes", f"{MIN_SCENES}-{MAX_SCENES}")
    m2.metric("Max retries", str(MAX_RETRIES_SCRIPT))
    m3.metric("Workspace", os.path.basename(WORKDIR))

    st.subheader("🕘 Recent runs")
    if not st.session_state.history:
        st.info("No runs yet.")
    else:
        for item in st.session_state.history[:5]:
            with st.container(border=True):
                st.write(f"**{item['title']}**")
                st.caption(f"Run: `{item['run_id']}` · Scenes: {item['scene_count']} · ~{int(item['est_sec'])}s")
                if item.get("video_path") and os.path.exists(item["video_path"]):
                    st.download_button(
                        "Download MP4",
                        data=open(item["video_path"], "rb"),
                        file_name=os.path.basename(item["video_path"]),
                        mime="video/mp4",
                        key=f"dl_{item['run_id']}",
                    )


if generate_btn:
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY in `.env`.")
        st.stop()
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    graph = build_graph()

    progress = st.progress(0, text="Starting…")
    status_box = st.empty()

    try:
        status_box.info("Step 1/4: Generating script (Groq)…")
        progress.progress(15)

        # Run the agentic graph
        final_state = graph.invoke({"user_prompt": prompt.strip()})

        progress.progress(85, text="Finalizing…")
        status_box.success("Done ✅")
        progress.progress(100, text="Completed")

    except Exception as e:
        status_box.error("Failed ❌")
        st.exception(e)
        st.stop()

    script: Optional[VideoScript] = final_state.get("script")
    errors = final_state.get("validation_errors") or []
    video_path = final_state.get("video_path")

    if script:
        est_sec = estimate_total_duration_sec(script)
        record = {
            "run_id": final_state.get("run_id", "unknown"),
            "title": script.title,
            "scene_count": len(script.scenes),
            "est_sec": est_sec,
            "video_path": video_path if keep_video_files else None,
        }
        st.session_state.history.insert(0, record)

    # If user wants to delete video files after run
    if not keep_video_files and video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass

    # ---- Tabs rendering ----
    with tabs[0]:
        st.subheader("🎞️ Video preview")
        if video_path and os.path.exists(video_path):
            st.video(video_path)
            with open(video_path, "rb") as f:
                st.download_button(
                    "⬇️ Download MP4",
                    data=f,
                    file_name=os.path.basename(video_path),
                    mime="video/mp4",
                )
        else:
            st.warning("Video not available (file missing).")

    with tabs[1]:
        st.subheader("🧾 Script")
        if script:
            st.markdown(f"### {script.title}")
            st.write(script.description)

            if errors:
                st.warning("Validation issues encountered (pipeline continued):")
                for err in errors:
                    st.write(f"- {err}")

            for i, sc in enumerate(script.scenes, start=1):
                with st.expander(f"Scene {i}", expanded=(i == 1)):
                    st.markdown("**Narration**")
                    st.write(sc.narration)
                    st.markdown("**Image prompt**")
                    st.write(sc.image_prompt)

            st.download_button(
                "Download script.json",
                data=script_to_json(script),
                file_name=f"{final_state.get('run_id','run')}_script.json",
                mime="application/json",
            )
        else:
            st.error("No script returned.")

    with tabs[2]:
        st.subheader("🖼️ Image prompts (copy/paste)")
        if script:
            prompts_text = "\n\n".join([f"Scene {i}: {s.image_prompt}" for i, s in enumerate(script.scenes, start=1)])
            st.text_area("All prompts", value=prompts_text, height=260, label_visibility="collapsed")
        else:
            st.info("Run generation to see prompts.")

    with tabs[3]:
        st.subheader("🪲 Debug")
        if show_debug:
            st.markdown("**Raw JSON from Groq**")
            st.code(final_state.get("script_raw", ""), language="json")
        else:
            st.info("Enable **Show debug** in the sidebar to display raw model output.")
