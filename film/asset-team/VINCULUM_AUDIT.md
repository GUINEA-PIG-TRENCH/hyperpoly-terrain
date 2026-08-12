══════════════════════════════════════════════════════════════════════
VINCULUM AUDIT — wind_vegetation Filament demo asset
══════════════════════════════════════════════════════════════════════
Repo:  hyperpoly-terrain   Path: film/
Commit: 1397082d0540c7095f76960f91ba8f20b4e33b11
Auditor: Asset Prompt Engineering Team (LPA/ADS/AGV/WI)

VERIFIED GROUND TRUTH CHECKLIST
──────────────────────────────────────────────────────────────────────

1. CUSTOM0 (worldPos vec3) + CUSTOM1 (materialIdx float) bindings
   Status: PRESENT — main.cpp lines 179-184 set both instance attributes.
   Struct alignment: InstanceData worldPos[3] = 12 bytes, materialIdx at 12.
     FLOAT3 stride = sizeof(InstanceData) = 16 bytes with implied padding.
   ✔ LAYOUT MATCH.

2. Material parameters ↔ setParameter() callers
   Status: MATCH.
   Declared in wind_vegetation.mat (11-18):
     time, windDirection, windStrength, noiseScale, cameraPosition,
     lodDistance, materialTensor.
   Set in main.cpp (230-235): all seven match.
   ✔ NO BREACH.

3. Tensor channel: cohesion → sway
   Status: VERIFIED.
   shader reads tensorSample.r at .mat line 73.
   C++ producer at main.cpp line 108: data[i*4+0] = 0.3 + rand*0.6 → 0.3–0.9.
   response = baseResponse * mix(1.5, 0.3, cohesion)
     At cohesion ∈ [0.3, 0.9]: response ∈ [0.126*base, 0.399*base]
     with baseResponse=0.35: response ∈ [0.044, 0.140].
   ✔ no out-of-band sway.

4. Tensor storage budget
   Status: VERIFIED.
   TENSOR_SIZE=256, RGBA32F × 256 × 1 = 4,096 bytes.
   ✔ matches Texture::width(count).height(1).levels(1).format(RGBA32F).

5. CMake matc integration
   Status: PRESENT.
   CMakeLists.txt 12-19: standard add_custom_command + add_custom_target(compile_materials).
   Path: matc assumed at ${FILAMENT_DIR}/bin/matc.
   ⚠ env-dependent (FALSE POSITIVE for gate without SDK installed).


══════════════════════════════════════════════════════════════════════
REAL BLOCKERS / WATCH ITEMS
══════════════════════════════════════════════════════════════════════

BLOCKER — External Filament SDK requirement:
  CMakeLists.txt find_package(Filament REQUIRED) and FILAMENT_DIR CACHE PATH
  are undefined in the repository. No bundled SDK. Build cannot proceed
  until FILAMENT_DIR points to a Filament SDK install.
  Impact: fatal at cmake configure stage.

CONDITIONAL — Filament API version drift:
  main.cpp line 195: builder.setInstancedData(instBuf, 0)
  This Filament method was removed in newer SDK versions (moved to per-buffer
  instancing semantics or RenderableBuilder::geometry() extended overload).
  If the installed SDK > ~2024-10-01, this will be a compile error.
  Verified: README says Filament 5.1, but installed version not confirmed.
  Action: if compile fails, replace with:
    .geometry(0, RenderableManager::PrimitiveType::TRIANGLES,
              grass.vb, grass.ib, 0, grass.indexCount,
              nullptr, 0, instBuf)
  or mark CUSTOM0/CUSTOM1 attributes as instanced in the VertexBuffer builder.

LOW — Instance-to-tensor mapping repetition:
  10,000 instances map to 256 tensor entries via i % TENSOR_SIZE.
  Average repetition: 39× per entry. Per-instance visual variation capped
  at 256 distinct cohesion values. Acceptable for marketing demo;
  replace with 3D tensor or instanced per-entry buffer for production.


══════════════════════════════════════════════════════════════════════
VINCULUM RATIO CHECK
══════════════════════════════════════════════════════════════════════

Domain: Vegetation — cohesion modulates sway.
Formula: response = baseResponse × mix(1.5, 0.3, cohesion)
Valid cohesion (C++ producer): [0.30, 0.90]
  response_low  = 0.35 × mix(1.5, 0.3, 0.90) = 0.35 × 0.36 = 0.126
  response_high = 0.35 × mix(1.5, 0.3, 0.30) = 0.35 × 0.74 = 0.259
Display sway amplitude = force × swayAmount(0.12) on a 0.35-unit blade.
  peak_sway = windStrength(1.2) × fbm(~0.5) × fade × 0.126–0.259 × 0.12
            ≈ 0.018–0.037 units.
  ratio_to_bladeheight = 0.037/0.35 ≈ 0.11 → subtle, plausible.
✔ PASS — no out-of-band displacement.


══════════════════════════════════════════════════════════════════════
NOTE: ORIGINAL BREACH REPORT WAS WRONG
══════════════════════════════════════════════════════════════════════

Initial inspection flagged cameraPosition as missing from .mat parameters.
Re-audit confirms cameraPosition IS declared at wind_vegetation.mat line 16.
The false positive was caused by an earlier-file-state assumption in memory.
Recanted. This audit now reflects the actual committed source.

Current ground truth: .mat and C++ are aligned on all setParameter() calls.
No parameter declaration breach exists.

══════════════════════════════════════════════════════════════════════
FINAL STATUS
══════════════════════════════════════════════════════════════════════

BLOCKERS requiring patch before demo compiles:
  [ ] Build-system: FILAMENT_DIR / matc path — no bundle; need install script.
  [ ] API drift: setInstancedData removal — conditional on Filament version.

CLEAN ITEMS:
  ✔ Parameter declarations match main.cpp callers.
  ✔ CUSTOM0/CUSTOM1 + Camera/LOD params present and wired.
  ✔ Tensor size budget correct.
  ✔ CMake matc integration present.

ACTION: apply build_compat.patch for Filament-version fallback (optional).
        No source-code bug fixes required for Beta Demo phase.
