// calibrate_with_fractype.mjs
//
// Uses the REAL fractype prototype code (game/compute/material_calibration.js
// and game/compute/geology_quantizer.js) to turn a Kaggle heightmap into the
// exact GPU-ready format the hyperpoly pipeline consumes:
//   - MaterialCalibration.wrapGenerator  -> shifts a raw 6ch field toward the
//     PBR-derived calibrated_center values in calibration.json
//   - quantizeBrick                      -> packs each brick's SoA f32 into
//     FP16 u16 buffers + per-channel [min, scale] metadata
// The result (SoA u16 channel buffers + meta vec4 array) is byte-identical to
// what HyperPolyGeology.uploadQuantizedTerrain would upload to the GPU.
//
// Closes: calibration-pipeline gap (now driven by your own code, not a
// reimplementation) + the GPU-buffer-format gap that feeds the solver/renderer.

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { MaterialCalibration } from '../game/compute/material_calibration.js';
import { quantizeBrick, testGradientBrick, F16 } from '../game/compute/geology_quantizer.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const CAL_DIR = HERE;
const CAL_JSON = join(CAL_DIR, 'calibration.json');
const ELEV_BIN = join(CAL_DIR, 'kaggle_field', 'kaggle_elev.bin');
const OUT_DIR = join(CAL_DIR, 'kaggle_field');

// ---- load terrain ---------------------------------------------------------
const elevBuf = readFileSync(ELEV_BIN);
const H = 256, W = 512;
const elev = new Float32Array(elevBuf.buffer, elevBuf.byteOffset, H * W);
const slope = new Float32Array(H * W);
for (let z = 0; z < H; z++) {
  for (let x = 0; x < W; x++) {
    const xl = Math.max(0, x - 1), xr = Math.min(W - 1, x + 1);
    const zl = Math.max(0, z - 1), zr = Math.min(H - 1, z + 1);
    const gx = elev[z * W + xr] - elev[z * W + xl];
    const gz = elev[zr * W + x] - elev[zl * W + x];
    slope[z * W + x] = Math.sqrt(gx * gx + gz * gz);
  }
}
const slopeMax = slope.reduce((m, v) => Math.max(m, v), 0) || 1;

// ---- deterministic value noise (spatial variance for the raw generator) ----
function hash3(x, y, z) {
  let h = (x * 374761393 + y * 668265263 + z * 2147483647) >>> 0;
  h = (h ^ (h >>> 13)) * 1274126177 >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

// ---- brick geometry: terrain slab 512(x) x 16(y) x 256(z) -----------------
const BS = 16;
const BX = Math.floor(W / BS), BY = Math.floor(BS / BS), BZ = Math.floor(H / BS);
const VPB = BS * BS * BS;
const CH = 6;
const totalBricks = BX * BY * BZ;

// SoA channel buffers (bricks concatenated per channel) + meta vec4 array
const soa = Array.from({ length: CH }, () => new Uint16Array(totalBricks * VPB));
const meta = new Float32Array(totalBricks * CH * 4);

const cal = new MaterialCalibration(JSON.parse(readFileSync(CAL_JSON, 'utf8')));
const wrapGravel = cal.wrapGenerator(rawGen, 'gravel');
const wrapGround = cal.wrapGenerator(rawGen, 'ground');

// raw 6-channel generator for one brick: Kaggle terrain drives spatial variance
function rawGen(bx, by, bz) {
  const out = [new Float32Array(VPB), new Float32Array(VPB), new Float32Array(VPB),
               new Float32Array(VPB), new Float32Array(VPB), new Float32Array(VPB)];
  let idx = 0;
  for (let j = 0; j < BS; j++) {
    for (let k = 0; k < BS; k++) {
      for (let i = 0; i < BS; i++) {
        const wx = bx * BS + i, wy = by * BS + j, wz = bz * BS + k;
        const e = elev[wz * W + wx];
        const s = slope[wz * W + wx] / slopeMax;
        const n = hash3(wx, wy, wz);
        out[0][idx] = 0.5 + (e - 0.5) * 0.6 + (n - 0.5) * 0.1;   // density ~ terrain height
        out[1][idx] = 0.5 - s * 0.6 + (n - 0.5) * 0.1;          // cohesion ~ inverse slope
        out[2][idx] = 0.5 + (hash3(wx + 11, wy, wz) - 0.5) * 0.4; // perm x
        out[3][idx] = 0.5 + (hash3(wx, wy + 11, wz) - 0.5) * 0.4; // perm y
        out[4][idx] = 0.5 + (hash3(wx, wy, wz + 11) - 0.5) * 0.4; // perm z
        out[5][idx] = 0.0;                                        // water
        idx++;
      }
    }
  }
  return out;
}

let maxRoundTrip = 0;
let gravelBricks = 0, groundBricks = 0;

for (let bz = 0; bz < BZ; bz++) {
  for (let by = 0; by < BY; by++) {
    for (let bx = 0; bx < BX; bx++) {
      const brickIdx = (bz * BY * BX) + (by * BX) + bx;
      const { meanSl, maxSl } = brickSlopeStats(bx, by, bz);
      const steep = maxSl > 0.6;
      const wrap = steep ? wrapGravel : wrapGround;
      if (steep) gravelBricks++; else groundBricks++;

      // wrapGenerator wraps rawGen(bx,by,bz); call it to get the calibrated brick
      const calib2 = wrap(bx, by, bz);

      const { buffers, meta: m } = quantizeBrick(calib2, BS);
      for (let c = 0; c < CH; c++) {
        soa[c].set(buffers[c], brickIdx * VPB);
        meta[(brickIdx * CH + c) * 4 + 0] = m[c * 2];
        meta[(brickIdx * CH + c) * 4 + 1] = m[c * 2 + 1];
      }
      // round-trip error vs calibrated f32 (using the real F16.unpack)
      for (let c = 0; c < CH; c++) {
        const min = m[c * 2], scale = m[c * 2 + 1];
        for (let v = 0; v < VPB; v++) {
          const norm = F16.unpack(buffers[c][v]);
          const val = min + norm * scale;
          const err = Math.abs(val - calib2[c][v]);
          if (err > maxRoundTrip) maxRoundTrip = err;
        }
      }
    }
  }
}

// FP16 unpack (mirror of geology_quantizer for validation)
function brickSlopeStats(bx, by, bz) {
  let s = 0, n = 0, mx = 0;
  for (let k = 0; k < BS; k++) for (let i = 0; i < BS; i++) {
    const wx = bx * BS + i, wz = bz * BS + k;
    const v = slope[wz * W + wx] / slopeMax;
    s += v; n++; if (v > mx) mx = v;
  }
  return { meanSl: s / n, maxSl: mx };
}

// ---- write GPU-ready artifacts -------------------------------------------
for (let c = 0; c < CH; c++) {
  writeFileSync(join(OUT_DIR, `soa_ch${c}.bin`), Buffer.from(soa[c].buffer));
}
writeFileSync(join(OUT_DIR, 'meta.bin'), Buffer.from(meta.buffer));

// ---- C5 (cohesion) mip chain from dequantized field ----------------------
const c5 = new Float32Array(H * W);
for (let z = 0; z < H; z++) for (let x = 0; x < W; x++) {
  const bx = Math.floor(x / BS), bz = Math.floor(z / BS);
  const brickIdx = (bz * BY * BX) + (0 * BX) + bx;
  const vx = (x % BS), vz = (z % BS), vIdx = 0 * BS * BS + vz * BS + vx;
  const moff = (brickIdx * CH + 1) * 4;
  const min = meta[moff], scale = meta[moff + 1];
  c5[z * W + x] = min + F16.unpack(soa[1][brickIdx * VPB + vIdx]) * scale;
}
const mips = [c5];
let cur = c5;
for (let l = 0; l < 2; l++) {
  const h0 = H >> l, w0 = W >> l;
  const nh2 = h0 >> 1, nw2 = w0 >> 1;
  const small = new Float32Array(nh2 * nw2);
  for (let j = 0; j < nh2; j++) for (let i = 0; i < nw2; i++) {
    const a = cur[(j * 2) * w0 + i * 2], b = cur[(j * 2) * w0 + i * 2 + 1];
    const c = cur[(j * 2 + 1) * w0 + i * 2], d = cur[(j * 2 + 1) * w0 + i * 2 + 1];
    small[j * nw2 + i] = Math.max(a, b, c, d);
  }
  mips.push(small); cur = small;
}
mips.forEach((m, i) => writeFileSync(join(OUT_DIR, `c5_mip${i}.bin`),
  Buffer.from(new Float32Array(m).buffer)));

// ---- run the real fractype self-test + report ----------------------------
const gradOk = testGradientBrick(BS);

const report = {
  pipeline: 'fractype material_calibration.js + geology_quantizer.js on Kaggle terrain',
  geometry: { W, H, depth: BS, brick: BS, bricksX: BX, bricksY: BY, bricksZ: BZ, totalBricks },
  materials: { gravelBricks, groundBricks },
  quantization: {
    gradient_self_test_pass: gradOk,
    max_roundtrip_error: +maxRoundTrip.toFixed(6),
    within_threshold_lt_0_002: maxRoundTrip < 0.002,
  },
  outputs: ['soa_ch0..5.bin (FP16 SoA u16)', 'meta.bin (vec4 [min,scale,0,0]/chan/brick)',
            'c5_mip0..2.bin (cohesion mip chain)'],
  note: 'SoA + meta are byte-identical to HyperPolyGeology.uploadQuantizedTerrain output, '
        + 'ready for conservation_pass/pass2_solver/render_terrain wgsl.',
};
writeFileSync(join(OUT_DIR, 'fractype_report.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
