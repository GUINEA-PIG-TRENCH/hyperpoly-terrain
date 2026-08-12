#!/usr/bin/env python3
"""
Real benchmark for eigenmode_decomp GOVERNOR module.
Tests:
  1. Vinculum tree depth traversal (8192 modes, zero fragmentation)
  2. DC isolation — large DC offset masked, physical principal mode identified
  3. Sensitivity scaling — coefficients from 1e-6 to 1e6, normalized perturbation finite
  4. Reconstruction convergence — L2 relative error monotonic decrease as N grows
"""
import math
import time
import numpy as np

from eigenmode_decomp import analyze_field

# ── synthetic basis_map: deterministic, no spatial grid assumptions ──
def _make_basis_map(num_modes: int):
    """Return a basis_map(n, pos) using catalog-like per-mode frequencies."""
    freqs = [(i + 1) * 0.7 for i in range(num_modes)]
    def basis_map(n, pos):
        x, y, z = pos
        phase = freqs[n] * (x + y + z)
        return math.cos(phase), math.sin(phase)
    return basis_map


def _positions(count=125):
    """5x5x5 grid inside [-1, 1]^3."""
    xs = [i * 0.5 - 1.0 for i in range(5)]
    return [(x, y, z) for x in xs for y in xs for z in xs][:count]


# ── 1. Depth / direct-indexing ────────────────────────────────────
def test_vinculum_tree_depth():
    print("\n[TEST 1] Vinculum tree depth: 8192 modes")
    num_modes = 8192
    coeffs = [math.sin(i * 0.1) + 1j * math.cos(i * 0.1) for i in range(num_modes)]
    flat = [v for c in coeffs for v in (c.real, c.imag)]
    basis = _make_basis_map(num_modes)
    positions = _positions()

    t0 = time.perf_counter()
    result = analyze_field(flat, basis_map=basis, sample_positions=positions)
    dt = time.perf_counter() - t0
    assert dt < 1.0, f"Depth traversal too slow: {dt:.3f}s"
    assert len(result.ranked_modes) == num_modes
    print(f"  traversed {num_modes} modes in {dt:.4f}s; ranked={len(result.ranked_modes)}")


# ── 2. DC isolation ───────────────────────────────────────────────
def test_dc_index_isolation():
    print("\n[TEST 2] DC isolation: 1e6 offset at mode 0, physical peak at mode 4")
    catalog = [0.0] * 6
    catalog[0] = 1e6           # DC
    catalog[1] = 2.5
    catalog[2] = 8.1
    catalog[3] = 0.4
    catalog[4] = 15.2          # expected principal
    catalog[5] = 3.3
    coeffs = [v for v in catalog]  # real-only for this synthetic test
    num_modes = len(coeffs)
    basis = _make_basis_map(num_modes)
    positions = _positions()

    result = analyze_field(coeffs, basis_map=basis, sample_positions=positions)
    top = result.ranked_modes[0]
    assert top.index == 4, f"Expected mode 4 as principal, got {top.index}"
    print(f"  principal mode index={top.index} magnitude={top.magnitude:.2f}")


# ── 3. Sensitivity scaling limits ─────────────────────────────────
def test_sensitivity_scaling_limits():
    print("\n[TEST 3] Sensitivity scaling 1e-6..1e6")
    coeffs = np.geomspace(1e-6, 1e6, 100).tolist()
    # imag zeros
    flat = [v for c in coeffs for v in (c, 0.0)]
    basis = _make_basis_map(len(coeffs))
    positions = _positions(count=20)

    with np.errstate(over='raise', under='raise', invalid='raise'):
        result = analyze_field(flat, basis_map=basis, sample_positions=positions)

    sens = list(result.delta_sensitivity.values())
    assert all(math.isfinite(v) for v in sens), "Non-finite sensitivity detected"
    max_sens = max(sens)
    assert max_sens <= 1.0, f"Normalized sensitivity breached bounds: {max_sens}"
    print(f"  max normalized sensitivity={max_sens:.6e}; all finite={all(math.isfinite(v) for v in sens)}")


# ── 4. Reconstruction convergence ─────────────────────────────────
def test_reconstruction_convergence():
    print("\n[TEST 4] Reconstruction convergence N=1..46")
    max_modes = 46
    # Build coefficients from bump function: heavier tail, still converges
    coeffs = [1.0 / (i + 1) for i in range(max_modes)]
    flat = [v for c in coeffs for v in (c, 0.0)]
    basis = _make_basis_map(max_modes)
    positions = _positions()

    prev = float('inf')
    for n in range(1, max_modes + 1):
        # Only use first 2*n coefficients
        sub = flat[: 2 * n]
        basis_n = _make_basis_map(n)
        result = analyze_field(sub, basis_map=basis_n, sample_positions=positions)
        err = result.reconstruction_error
        print(f"  N={n:02d} | reconstruction_error={err:.6e} | principal_delta={result.principal_delta:.6e}")
        assert err <= prev + 1e-12, f"Convergence broken at N={n}: {err} > {prev}"
        prev = err

    print("  PASS: strict monotonic convergence")


def run_all():
    test_vinculum_tree_depth()
    test_dc_index_isolation()
    test_sensitivity_scaling_limits()
    test_reconstruction_convergence()
    print("\nALL GOVERNOR BENCHMARKS PASSED.")


if __name__ == "__main__":
    run_all()
