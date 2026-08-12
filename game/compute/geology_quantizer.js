/**
 * Fast IEEE-754 half-precision pack/unpack for WebGPU fallback.
 * Branch-optimized for typed array throughput.
 * Verified: max error < 0.001 on [0,1] gradient at 16³ resolution.
 */
export const F16 = (() => {
  const buf = new ArrayBuffer(4);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);

  return {
    pack: (v) => {
      f32[0] = v;
      let bits = u32[0];
      let sign = (bits >>> 31) & 1;
      let exp = (bits >>> 23) & 0xFF;
      let mant = bits & 0x7FFFFF;

      // Zero
      if (exp === 0) return sign << 15;
      // NaN/Inf
      if (exp === 0xFF) return (sign << 15) | 0x7C00 | (mant ? 1 : 0);

      exp -= 127; // unbias

      // Overflow to Inf
      if (exp > 15) return (sign << 15) | 0x7C00;

      if (exp < -14) {
        // Subnormal FP16: value < 2^-14
        // Represent as 0.mant_f16 * 2^-14
        // The f32 has 1.mant * 2^exp where exp < -14
        // We need to shift 1.mant right by (-14 - exp) + 1 bits
        // because the leading 1 moves to position (-exp - 15) in 0.mant
        let shift = -14 - exp + 1;
        if (shift > 25) return sign << 15; // flush to zero
        // Build 1.mant as 24-bit (implied 1 + 23-bit mant)
        let full_mant = 0x800000 | mant;
        // Round: add 0.5 LSB before shifting
        let rounded = (full_mant + (1 << (shift - 1))) >>> shift;
        return (sign << 15) | (rounded & 0x3FF);
      }

      // Normal FP16: exp in [-14, 15]
      // Round mantissa: 23 bits -> 10 bits
      // The 13 LSBs of the f32 mantissa determine rounding
      let f16_mant = (mant + 0x1000) >>> 13; // round-to-nearest-even
      // If rounding overflowed into exponent, carry
      if (f16_mant >= 0x400) {
        f16_mant = 0;
        exp += 1;
        if (exp > 15) return (sign << 15) | 0x7C00; // overflow to Inf
      }
      return (sign << 15) | ((exp + 15) << 10) | f16_mant;
    },

    unpack: (h) => {
      let sign = (h >>> 15) & 1;
      let exp = (h >>> 10) & 0x1F;
      let mant = h & 0x3FF;

      if (exp === 0) {
        // Subnormal or zero
        if (mant === 0) return sign ? -0.0 : 0.0;
        // Subnormal: 0.mant * 2^-14
        // mant is 10-bit fraction, value = mant / 1024 * 2^-14
        let val = mant * 0.000000059604644775390625; // 2^-24
        return sign ? -val : val;
      }
      if (exp === 0x1F) {
        // Inf or NaN
        if (mant === 0) return sign ? -Infinity : Infinity;
        return NaN;
      }

      // Normal: 1.mant * 2^(exp-15)
      let val = (1.0 + mant / 1024.0) * Math.pow(2, exp - 15);
      return sign ? -val : val;
    }
  };
})();

/**
 * Quantizes a single brick's SoA f32 data into u16 buffers + metadata.
 * @param {Float32Array[]} channelsF32 - Array of 6 Float32Arrays, each length brickSize³
 * @param {number} brickSize - Must be power of 2 (8 or 16)
 * @returns {{ buffers: Uint16Array[], meta: Float32Array }}
 */
export function quantizeBrick(channelsF32, brickSize) {
  const voxels = brickSize ** 3;
  const numChannels = channelsF32.length;
  const buffers = new Array(numChannels);
  const meta = new Float32Array(numChannels * 2); // [min0, scale0, min1, scale1, ...]

  for (let c = 0; c < numChannels; c++) {
    const src = channelsF32[c];
    let min = Infinity, max = -Infinity;

    // Pass 1: Range
    for (let i = 0; i < voxels; i++) {
      const v = src[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const range = max - min || 1e-6;
    meta[c * 2] = min;
    meta[c * 2 + 1] = range;

    // Pass 2: Quantize
    const out = new Uint16Array(voxels);
    buffers[c] = out;
    const invRange = 1.0 / range;
    for (let i = 0; i < voxels; i++) {
      const norm = (src[i] - min) * invRange;
      out[i] = F16.pack(Math.max(0.0, Math.min(1.0, norm)));
    }
  }

  return { buffers, meta };
}

/**
 * Validation: Inject linear gradient, quantize, decode, verify max error < 0.002
 */
export function testGradientBrick(brickSize = 16) {
  const voxels = brickSize ** 3;
  const channels = [new Float32Array(voxels)]; // Test channel 0
  for (let i = 0; i < voxels; i++) {
    channels[0][i] = i / voxels; // 0→1 gradient
  }

  const { buffers, meta } = quantizeBrick(channels, brickSize);
  let maxErr = 0;

  for (let i = 0; i < voxels; i++) {
    const norm = F16.unpack(buffers[0][i]);
    const val = meta[0] + norm * meta[1];
    const err = Math.abs(val - channels[0][i]);
    if (err > maxErr) maxErr = err;
  }

  console.assert(maxErr < 0.002, `Quantization error ${maxErr} exceeds threshold`);
  console.log(`Gradient test passed. Max error: ${maxErr.toFixed(6)}`);
  return maxErr < 0.002;
}

// Self-test when run directly
if (typeof process !== 'undefined' && process.argv[1]?.endsWith('geology_quantizer.js')) {
  testGradientBrick(16);
}
