/**
 * LIGHTING AI — Gemma 4-driven dynamic lighting for HYPERPOLY
 * 
 * Calls Ollama (gemma3:1b) with compact game state to produce
 * lighting parameters that evolve the scene based on player context.
 * 
 * Architecture:
 *   - Periodic async calls to localhost:11434 (Ollama)
 *   - JSON-structured prompt → JSON response with lighting params
 *   - Smooth interpolation toward AI targets (no pop-in)
 *   - Procedural fallback day/night cycle if AI unavailable
 * 
 * Usage:
 *   const ai = new LightingAI();
 *   ai.init();
 *   // In render loop:
 *   const params = ai.update(dt, cameraPos, playerY);
 *   // Apply params to terrainRenderer.setViewProj(...)
 */

export class LightingAI {
  constructor() {
    // Current lighting state (always available, smoothly interpolated)
    this.sunDirection = [-0.5, 0.7, -0.5];
    this.ambientColor = [0.15, 0.18, 0.25];
    this.sunColor = [1.0, 0.95, 0.85];
    this.fogDensity = 0.012;
    this.fogColor = [0.15, 0.18, 0.25];

    // AI target (what we're interpolating toward)
    this._target = null;
    this._interpSpeed = 0.5; // blend per second toward target

    // Timing
    this._aiInterval = 8.0;   // seconds between AI calls
    this._aiTimer = 0;
    this._pendingRequest = false;
    this._lastCameraPos = null;
    this._significantMove = 50; // units — only call AI if moved this far

    // Procedural fallback
    this._timeOfDay = 0;      // radians, 0=dawn, π/2=noon, π=dusk
    this._daySpeed = 0.03;    // rad/s (~2 min full cycle)

    // Stats
    this.aiCalls = 0;
    this.aiFailures = 0;
    this.usingAI = false;
  }

  /**
   * Initialize — verify Ollama is reachable.
   */
  async init() {
    try {
      const r = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(2000) });
      if (r.ok) {
        this.usingAI = true;
        console.log('[LIGHTING AI] Ollama reachable — AI lighting active');
        // Immediate first call
        this._aiTimer = this._aiInterval;
        return true;
      }
    } catch (e) {
      console.warn('[LIGHTING AI] Ollama unreachable — using procedural fallback');
    }
    this.usingAI = false;
    return false;
  }

  /**
   * Call once per frame. Returns current lighting params.
   * @param {number} dt — frame delta in seconds
   * @param {number[]} cameraPos — [x, y, z]
   * @param {number} playerY — player height (for depth context)
   * @returns {{sunDirection, ambientColor, sunColor, fogDensity, fogColor}}
   */
  update(dt, cameraPos, playerY) {
    this._aiTimer += dt;

    // ── Trigger AI call ──
    const moved = this._lastCameraPos
      ? Math.sqrt(
          (cameraPos[0] - this._lastCameraPos[0])**2 +
          (cameraPos[2] - this._lastCameraPos[2])**2
        )
      : Infinity;

    if (this.usingAI && this._aiTimer >= this._aiInterval && !this._pendingRequest) {
      if (moved > this._significantMove || this._aiTimer >= this._aiInterval * 3) {
        this._requestLighting(cameraPos, playerY);
        this._aiTimer = 0;
        this._lastCameraPos = [...cameraPos];
      }
    }

    // ── Interpolate toward AI target ──
    if (this._target) {
      const t = Math.min(1.0, this._interpSpeed * dt);
      this._lerp3(this.sunDirection, this._target.sunDirection, t);
      this._lerp3(this.ambientColor, this._target.ambientColor, t);
      this._lerp3(this.sunColor, this._target.sunColor, t);
      this.fogDensity += (this._target.fogDensity - this.fogDensity) * t;
      this._lerp3(this.fogColor, this._target.fogColor, t);
    } else {
      // ── Procedural fallback day/night ──
      this._timeOfDay += this._daySpeed * dt;
      if (this._timeOfDay > Math.PI * 2) this._timeOfDay -= Math.PI * 2;
      this._applyProceduralDayNight(playerY);
    }

    return {
      sunDirection: [...this.sunDirection],
      ambientColor: [...this.ambientColor],
      sunColor: [...this.sunColor],
      fogDensity: this.fogDensity,
      fogColor: [...this.fogColor],
    };
  }

  /**
   * Send compact game state to Gemma 4, get lighting params back.
   */
  async _requestLighting(cameraPos, playerY) {
    this._pendingRequest = true;
    const depth = this._classifyDepth(playerY);

    // Compact prompt — under 200 chars, expects JSON back
    const prompt = `You are a lighting director for a WebGPU terrain engine.
Player at (${cameraPos[0].toFixed(0)},${cameraPos[1].toFixed(0)},${cameraPos[2].toFixed(0)}), depth=${depth}, world is 256^3 voxel field with density/cohesion/porosity/water channels.
Return ONLY JSON with these keys: sunDir[x,y,z], ambient[r,g,b], sunColor[r,g,b], fogDensity, fogColor[r,g,b].
sunDir should point toward player for dramatic lighting. Underground=cool ambient+warm artificial. Surface=natural daylight. Underwater=blue-green shift, high fog.
Example: {"sunDir":[0.3,0.8,-0.4],"ambient":[0.12,0.2,0.35],"sunColor":[1,0.85,0.6],"fogDensity":0.018,"fogColor":[0.2,0.15,0.4]}`;

    try {
      const r = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3:1b',
          prompt: prompt,
          stream: false,
          options: { temperature: 0.3, num_predict: 120 }
        }),
        signal: AbortSignal.timeout(15000),
      });

      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const response = data.response || '';

      // Extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        this._target = this._validateAndApply(parsed);
        this.aiCalls++;
        console.log('[LIGHTING AI] New target:', this._target);
      }
    } catch (e) {
      this.aiFailures++;
      if (this.aiFailures > 3) {
        console.warn('[LIGHTING AI] Too many failures — switching to procedural');
        this.usingAI = false;
      }
    } finally {
      this._pendingRequest = false;
    }
  }

  /**
   * Validate AI response, clamp values, return clean target.
   */
  _validateAndApply(parsed) {
    return {
      sunDirection: this._clampVec3(parsed.sunDir || parsed.sunDirection, -1, 1, [-0.5, 0.7, -0.5]),
      ambientColor: this._clampVec3(parsed.ambient, 0, 0.6, [0.15, 0.18, 0.25]),
      sunColor: this._clampVec3(parsed.sunColor, 0, 2.5, [1.0, 0.95, 0.85]),
      fogDensity: Math.max(0.002, Math.min(0.05, parsed.fogDensity || 0.012)),
      fogColor: this._clampVec3(parsed.fogColor, 0, 1, [0.15, 0.18, 0.25]),
    };
  }

  _clampVec3(arr, min, max, fallback) {
    if (!arr || arr.length < 3) return [...fallback];
    return [
      Math.max(min, Math.min(max, Number(arr[0]) || fallback[0])),
      Math.max(min, Math.min(max, Number(arr[1]) || fallback[1])),
      Math.max(min, Math.min(max, Number(arr[2]) || fallback[2])),
    ];
  }

  _lerp3(a, b, t) {
    a[0] += (b[0] - a[0]) * t;
    a[1] += (b[1] - a[1]) * t;
    a[2] += (b[2] - a[2]) * t;
  }

  _classifyDepth(y) {
    if (y < 10) return 'deep_underground';
    if (y < 50) return 'underground';
    if (y < 100) return 'surface';
    if (y < 200) return 'high';
    return 'extreme_height';
  }

  /**
   * Procedural day/night cycle — runs when AI is unavailable.
   */
  _applyProceduralDayNight(playerY) {
    const t = this._timeOfDay;
    const sinT = Math.sin(t);
    const cosT = Math.cos(t);

    // Sun orbits in a tilted circle
    const sunAngle = t;
    this.sunDirection = [
      Math.cos(sunAngle) * 0.7,
      Math.max(0.05, Math.sin(sunAngle) * 0.8),
      -0.5,
    ];

    // Normalize
    const len = Math.sqrt(this.sunDirection[0]**2 + this.sunDirection[1]**2 + this.sunDirection[2]**2);
    this.sunDirection = this.sunDirection.map(v => v / len);

    // Sun color: warm at dawn/dusk, white at noon
    const sunHeight = Math.max(0, this.sunDirection[1]);
    const warmth = 1.0 - sunHeight * 0.6; // 1.0 at horizon, 0.4 at noon
    this.sunColor = [
      1.0,
      0.7 + sunHeight * 0.3,
      0.4 + sunHeight * 0.5,
    ];

    // Ambient: brighter at noon, darker at night
    const dayFactor = Math.max(0.05, sunHeight);
    const baseAmbient = [
      0.12 * dayFactor + 0.01,
      0.16 * dayFactor + 0.01,
      0.30 * dayFactor + 0.02,
    ];

    // Depth adjustment: underground = darker blue ambient
    const depth = playerY / 128;
    const underground = Math.max(0, 1 - depth);
    this.ambientColor = [
      baseAmbient[0] * (1 - underground * 0.7),
      baseAmbient[1] * (1 - underground * 0.6),
      baseAmbient[2] * (1 + underground * 0.3),
    ];

    // Fog: heavier at night and underground
    this.fogDensity = 0.006 + (1 - dayFactor) * 0.015 + underground * 0.01;
    this.fogColor = [
      0.08 + underground * 0.05,
      0.10 + underground * 0.04,
      0.20 + underground * 0.05,
    ];
  }

  /**
   * Get current sun direction as a normalized vec3 (for JS uniform upload).
   */
  getSunDirection() {
    const len = Math.sqrt(this.sunDirection[0]**2 + this.sunDirection[1]**2 + this.sunDirection[2]**2) || 1;
    return this.sunDirection.map(v => v / len);
  }
}
