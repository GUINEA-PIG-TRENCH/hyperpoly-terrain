"""
LS04 LIGHTING AI SERVER — Deploy on Lightning AI GPU Studio
FastAPI endpoint serving Gemma 4 dynamic lighting for HYPERPOLY terrain engine.

DEPLOYMENT (paste into Lightning Studio terminal after creation):
  pip install fastapi uvicorn llama-cpp-python
  wget https://huggingface.co/bartowski/gemma-4-2b-it-GGUF/resolve/main/gemma-4-2b-it-Q4_K_M.gguf
  python server.py

ENDPOINT:
  POST /lighting
  Body: {"position": [128, 64, 200], "depth": "surface", "biome": "block"}
  Response: {"sunDirection": [0.3, 0.8, -0.4], "ambientColor": [...], ...}

COST ESTIMATE (T4 GPU):
  - T4: ~$0.40/hr → $10 buys 25 hours
  - Model download: ~2 min ($0.01)
  - Per-request: <2s inference, negligible
  - Auto-shutdown after 30min idle recommended

MANUAL SETUP STEPS:
  1. Go to https://lightning.ai/dashawnspacem-org/my-studios
  2. Click "New Studio" → name it "ls04"
  3. Select GPU: T4 (cheapest, $0.40/hr)
  4. Start the studio
  5. Paste this entire file into the terminal or upload it
  6. Run: python server.py
  7. Click "Add Port" → expose port 8766
  8. Copy the public URL → update your game's lighting_ai.js Ollama URL to this endpoint
"""

import os
import json
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Configuration ──
MODEL_PATH = Path("gemma-4-2b-it-Q4_K_M.gguf")
PORT = 8766

# ── FastAPI app ──
app = FastAPI(title="LS04 Lighting AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ──
class LightingRequest(BaseModel):
    position: list[float]  # [x, y, z]
    depth: str = "surface"  # surface, underground, deep_underground, high
    biome: str = "between"  # block, vault, between

class LightingResponse(BaseModel):
    sunDirection: list[float]
    ambientColor: list[float]
    sunColor: list[float]
    fogDensity: float
    fogColor: list[float]

# ── Llama model (lazy load) ──
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        print(f"[LS04] Loading {MODEL_PATH}...")
        t0 = time.time()
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=512,
            n_threads=4,
            n_gpu_layers=20,  # offload 20 layers to GPU
            verbose=False,
        )
        print(f"[LS04] Model loaded in {time.time() - t0:.1f}s")
    return _llm

# ── Procedural fallback (when model loading or too many concurrent requests) ──
def procedural_lighting(position, depth):
    """Fast fallback — no LLM needed."""
    x, y, z = position
    import math
    t = time.time() * 0.3

    # Sun orbits
    sun_dir = [
        math.cos(t) * 0.7,
        max(0.05, math.sin(t) * 0.8),
        -0.5,
    ]
    length = math.sqrt(sum(v**2 for v in sun_dir))
    sun_dir = [v / length for v in sun_dir]

    sun_height = max(0, sun_dir[1])
    warmth = 1.0 - sun_height * 0.6

    sun_color = [1.0, 0.7 + sun_height * 0.3, 0.4 + sun_height * 0.5]
    day_factor = max(0.05, sun_height)

    # Depth adjustment
    depth_factor = max(0, 1 - y / 128)
    ambient = [
        0.12 * day_factor * (1 - depth_factor * 0.7) + 0.01,
        0.16 * day_factor * (1 - depth_factor * 0.6) + 0.01,
        0.30 * day_factor * (1 + depth_factor * 0.3) + 0.02,
    ]

    fog_density = 0.006 + (1 - day_factor) * 0.015 + depth_factor * 0.01
    fog_color = [0.08 + depth_factor * 0.05, 0.10 + depth_factor * 0.04, 0.20 + depth_factor * 0.05]

    return LightingResponse(
        sunDirection=sun_dir,
        ambientColor=ambient,
        sunColor=sun_color,
        fogDensity=round(fog_density, 4),
        fogColor=fog_color,
    )

def ai_lighting(position, depth, biome):
    """Gemma 4-driven lighting."""
    llm = get_llm()
    x, y, z = position

    prompt = f"""You are a lighting director. Player at ({x:.0f},{y:.0f},{z:.0f}), depth={depth}, biome={biome}.
Return ONLY JSON: {{"sunDir":[x,y,z],"ambient":[r,g,b],"sunColor":[r,g,b],"fogDensity":float,"fogColor":[r,g,b]}}.
Underground: cool ambient + warm artificial. Surface: natural daylight. High: thin air, bright, desaturated."""

    try:
        output = llm(
            prompt,
            max_tokens=120,
            temperature=0.6,
            stop=["\n\n"],
        )
        text = output["choices"][0]["text"]

        # Extract JSON
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            data = json.loads(match.group(0))
            return LightingResponse(
                sunDirection=data.get("sunDir", [0.3, 0.8, -0.4]),
                ambientColor=data.get("ambient", [0.12, 0.2, 0.35]),
                sunColor=data.get("sunColor", [1.0, 0.85, 0.6]),
                fogDensity=data.get("fogDensity", 0.018),
                fogColor=data.get("fogColor", [0.2, 0.15, 0.4]),
            )
    except Exception as e:
        print(f"[LS04] AI error: {e}")

    return procedural_lighting(position, depth)

# ── Routes ──
@app.get("/")
def root():
    return {"status": "LS04 Lighting AI", "model": str(MODEL_PATH), "endpoints": ["POST /lighting", "GET /health"]}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _llm is not None}

@app.post("/lighting", response_model=LightingResponse)
def lighting(req: LightingRequest):
    # Use AI for first request, fall back to procedural for speed
    # In production: use AI every Nth request, procedural otherwise
    return ai_lighting(req.position, req.depth, req.biome)

# ── Main ──
if __name__ == "__main__":
    print(f"[LS04] Starting Lighting AI Server on port {PORT}")
    print(f"[LS04] Model path: {MODEL_PATH} (exists: {MODEL_PATH.exists()})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
