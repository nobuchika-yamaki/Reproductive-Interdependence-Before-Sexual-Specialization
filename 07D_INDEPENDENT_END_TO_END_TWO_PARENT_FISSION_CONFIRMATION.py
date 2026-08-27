#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07D_INDEPENDENT_END_TO_END_TWO_PARENT_FISSION_CONFIRMATION.py

Independent confirmatory replication fixed after completion of 07C and before any
07D scientific data are observed.

Primary causal chain
--------------------
Fresh environmental-history field formation -> differentiated catalytic states ->
passive convergence at the single pre-specified tau_conv/tau_C ratio 1.0 -> the
chronologically first natural exact two-parent fusion -> explicit serial chemical
closure -> finite membrane-material production -> conserved-area diffuse-interface
shape instability -> spontaneous topological fission.

Independence
------------
The 07A/07C discovery seeds 70000-70031 are never reused. Full mode uses fresh
07D seeds 71000-71031 and regenerates the physical release state for every seed.
No old release ZIP is accepted. Ratio 1.0 is the only confirmatory operating point.
It was chosen prospectively for 07D from the completed 07C secondary discovery and
is fixed before the new seed block is generated.

Success rule
------------
07D confirms the end-to-end chain only if BOTH are satisfied:
  1) the fresh-seed 07A chemical interaction at ratio 1.0 is positive with the
     inherited two-sided exact Wilcoxon signed-rank test at p < 0.05; and
  2) AB_POOL exceeds AA_EQUAL_MASS, BB_EQUAL_MASS, and AB_NO_POOL in fresh-seed
     spontaneous fission outcomes, using exact paired McNemar tests with Holm
     correction across the three pre-specified physical comparisons (all p_Holm<0.05).

NO_Y is a mechanistic negative control and is reported but is not an additional
post-hoc success gate. Missing natural fusion events remain missing and are never
converted to failures or zeros.

No genes, mutation, fitness, selection, partner sensing, pair attraction, division
trigger, J threshold, division axis, daughter placement rule, or favorable-event
selection is present.

Dependencies: Python >= 3.10, numpy, scipy. matplotlib is optional in inherited code.

Typical full run on macOS:
    cd ~/Downloads
    caffeinate -i python3 07D_INDEPENDENT_END_TO_END_TWO_PARENT_FISSION_CONFIRMATION.py \
      --mode full --workers 8 --resume

Default output:
    ~/Desktop/INDEPENDENT_END_TO_END_TWO_PARENT_FISSION_V07D_FULL
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
    SCIPY_AVAILABLE = True
except Exception:
    ndi = None
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False


VERSION = "2026-06-05-holecount-sweep-fixed-total-numpy2-fixed"


# =============================================================================
# Utility
# =============================================================================

class ProgressLogger:
    def __init__(self, outdir: Path):
        self.t0 = time.time()
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log_path = outdir / "run_progress.log"
        self.fp = open(self.log_path, "w", encoding="utf-8")

    def log(self, msg: str) -> None:
        elapsed = time.time() - self.t0
        line = f"[{elapsed:9.1f}s] {msg}"
        print(line, flush=True)
        self.fp.write(line + "\n")
        self.fp.flush()

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass


def json_safe(value: Any) -> Any:
    """Recursively convert Path/NumPy values so json.dump never fails."""
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: json_safe(row.get(k, "")) for k in fieldnames})


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def laplacian6(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1)
        + np.roll(a, 1, axis=2) + np.roll(a, -1, axis=2)
        - 6.0 * a
    )


def mean6(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1)
        + np.roll(a, 1, axis=2) + np.roll(a, -1, axis=2)
    ) / 6.0


def grad3(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = 0.5 * (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0))
    gy = 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))
    gz = 0.5 * (np.roll(a, -1, axis=2) - np.roll(a, 1, axis=2))
    return gx, gy, gz


def gradmag3(a: np.ndarray) -> np.ndarray:
    gx, gy, gz = grad3(a)
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def advect_upwind(a: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    # Stable, low-order upwind advection.
    ax_back = a - np.roll(a, 1, axis=0)
    ax_forw = np.roll(a, -1, axis=0) - a
    ay_back = a - np.roll(a, 1, axis=1)
    ay_forw = np.roll(a, -1, axis=1) - a
    az_back = a - np.roll(a, 1, axis=2)
    az_forw = np.roll(a, -1, axis=2) - a
    return -(
        np.where(vx >= 0, vx * ax_back, vx * ax_forw)
        + np.where(vy >= 0, vy * ay_back, vy * ay_forw)
        + np.where(vz >= 0, vz * az_back, vz * az_forw)
    )


def advect_conservative_upwind(a: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """Mass-conservative first-order upwind advection, grid spacing = 1.

    Returns -div(v a). Face velocities are arithmetic means of neighboring
    cell-centered velocities. With zero normal boundary velocity the wrapped
    boundary flux is zero, and the discrete domain sum is conserved to
    roundoff before reactions/diffusion/sponge terms.
    """
    def axis_term(arr: np.ndarray, vel: np.ndarray, axis: int) -> np.ndarray:
        arr_f = np.roll(arr, -1, axis=axis)
        vel_f = 0.5 * (vel + np.roll(vel, -1, axis=axis))
        flux_f = np.where(vel_f >= 0.0, vel_f * arr, vel_f * arr_f)
        flux_b = np.roll(flux_f, 1, axis=axis)
        return -(flux_f - flux_b)

    return axis_term(a, vx, 0) + axis_term(a, vy, 1) + axis_term(a, vz, 2)


def discrete_divergence_centered(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    return (
        0.5 * (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0))
        + 0.5 * (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))
        + 0.5 * (np.roll(vz, -1, axis=2) - np.roll(vz, 1, axis=2))
    )


def sponge_mask(shape: Tuple[int, int, int], width: int = 5) -> np.ndarray:
    n = shape[0]
    grid = np.indices(shape)
    dist = np.minimum.reduce([
        grid[0], grid[1], grid[2],
        n - 1 - grid[0], n - 1 - grid[1], n - 1 - grid[2]
    ]).astype(np.float32)
    s = np.clip((width - dist) / max(width, 1), 0.0, 1.0)
    return s


def normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < eps:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


# =============================================================================
# Environment and simulation
# =============================================================================

@dataclass
class SimConfig:
    seed: int
    hole_count: int
    fixed_total_pore: bool
    reference_hole_count: int
    confined_n: int
    open_n: int
    pre_steps: int
    gen_steps: int
    open_steps: int
    sample_every: int
    dt: float
    b_threshold: float
    t_threshold: float
    min_component_voxels: int
    min_lumen_voxels: int
    source_regime: str
    recenter: bool
    save_float16: bool
    release_min_components: int
    release_check_every: int
    release_min_gen_step: int
    release_max_largest_fraction: float
    release_fallback: str


def gaussian_mixture_3d(n: int, rng: np.random.Generator, count: int, sigma_range: Tuple[float, float]) -> np.ndarray:
    x, y, z = np.indices((n, n, n), dtype=np.float32)
    field = np.zeros((n, n, n), dtype=np.float32)
    for _ in range(count):
        cx, cy, cz = rng.uniform(0.15 * n, 0.85 * n, size=3)
        sx = rng.uniform(sigma_range[0] * n, sigma_range[1] * n)
        sy = rng.uniform(sigma_range[0] * n, sigma_range[1] * n)
        sz = rng.uniform(sigma_range[0] * n, sigma_range[1] * n)
        amp = rng.uniform(0.6, 1.2)
        d = ((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2
        field += amp * np.exp(-0.5 * d).astype(np.float32)
    return normalize(field)


@dataclass
class Environment3D:
    pore: np.ndarray
    mineral: np.ndarray
    residence: np.ndarray
    vent: np.ndarray
    pressure: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    shear: np.ndarray
    source_shape: np.ndarray


def multipore_centers(n: int, hole_count: int) -> List[Tuple[float, float, float]]:
    """Deterministic, physically separated pore centers.

    Centers are selected by farthest-point sampling from a 3 x 3 x 3 candidate grid.
    This keeps the placement rule fixed across 4, 6, and 8-hole conditions and avoids
    post hoc visual tuning.
    """
    if hole_count < 1:
        raise ValueError("hole_count must be >= 1")
    if hole_count > 12:
        raise ValueError("hole_count > 12 is intentionally disallowed for this assay")
    levels = np.array([0.24, 0.50, 0.76], dtype=np.float64) * float(n)
    candidates = np.array([(x, y, z) for x in levels for y in levels for z in levels], dtype=np.float64)
    # Prefer separated corner-like basins first; then add farthest remaining points.
    chosen = [np.array([0.24*n, 0.24*n, 0.24*n], dtype=np.float64)]
    while len(chosen) < hole_count:
        best_i = None
        best_d = -1.0
        for i, c in enumerate(candidates):
            if any(np.linalg.norm(c - q) < 1e-9 for q in chosen):
                continue
            d = min(float(np.linalg.norm(c - q)) for q in chosen)
            # mild center penalty prevents early central bridging when alternatives are equivalent
            center_penalty = 0.03 * float(np.linalg.norm(c - np.array([0.5*n, 0.5*n, 0.5*n])))
            score = d + center_penalty
            if score > best_d:
                best_d = score
                best_i = i
        chosen.append(candidates[best_i])
    return [tuple(float(v) for v in c) for c in chosen]


def separated_gaussian_pores(
    n: int,
    rng: np.random.Generator,
    hole_count: int,
    fixed_total_pore: bool = True,
    reference_hole_count: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pore field, mineral field, and strict pore mask for separated holes.

    If fixed_total_pore=True, per-hole radius is scaled by
    (reference_hole_count / hole_count)^(1/3). This approximately fixes total pore/
    resource volume while varying the number of holes.
    """
    x, y, z = np.indices((n, n, n), dtype=np.float32)
    pore = np.zeros((n, n, n), dtype=np.float32)
    mineral = np.zeros_like(pore)
    strict_mask = np.zeros_like(pore)
    centers = multipore_centers(n, hole_count)
    radius_scale = (float(reference_hole_count) / float(hole_count)) ** (1.0 / 3.0) if fixed_total_pore else 1.0
    # Clamp prevents tiny holes becoming numerically unstable at high hole counts.
    radius_scale = float(np.clip(radius_scale, 0.68, 1.18))
    for j, (cx, cy, cz) in enumerate(centers):
        sx = rng.uniform(0.060 * n, 0.080 * n) * radius_scale
        sy = rng.uniform(0.060 * n, 0.080 * n) * radius_scale
        sz = rng.uniform(0.060 * n, 0.080 * n) * radius_scale
        d = ((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2
        g = np.exp(-0.5 * d).astype(np.float32)
        pore = np.maximum(pore, g)
        mineral = np.maximum(mineral, (0.85 + 0.10 * rng.random()) * g)
        strict_mask = np.maximum(strict_mask, (g > 0.16).astype(np.float32))
    pore = normalize(pore)
    mineral = normalize(mineral)
    return pore.astype(np.float32), mineral.astype(np.float32), strict_mask.astype(np.float32)


def make_confined_environment(n: int, rng: np.random.Generator, cfg: SimConfig) -> Environment3D:
    # Multi-pore environment: separated microtopographic holes rather than one
    # continuous fusing basin. Hole count is an experimental condition.
    pore, mineral, pore_mask = separated_gaussian_pores(
        n,
        rng,
        hole_count=cfg.hole_count,
        fixed_total_pore=cfg.fixed_total_pore,
        reference_hole_count=cfg.reference_hole_count,
    )
    rough = normalize(gradmag3(pore))

    # High residence only inside the separated holes. Outside the holes, residence
    # stays low to prevent the whole lattice from becoming one membrane sheet.
    residence = clip01(0.025 + pore_mask * (0.72 * pore + 0.28 * mineral) - 0.10 * rough)
    vent = clip01(pore_mask * (0.08 + 0.70 * pore + 0.22 * mineral) + 0.01 * rng.random((n, n, n), dtype=np.float32))

    # Pressure/shear barrier outside holes reduces bridging between basins.
    outside = 1.0 - pore_mask
    pressure = clip01(0.20 * normalize(gradmag3(pore)) + 0.55 * outside + 0.05 * rng.random((n, n, n), dtype=np.float32))

    base = rng.normal(0, 1, size=3)
    base = base / (np.linalg.norm(base) + 1e-12)
    flow_speed = 0.014
    atten = np.clip(1.0 - 0.80 * residence + 0.35 * outside, 0.15, 1.25)
    vx = (flow_speed * base[0] * atten + 0.003 * rng.normal(size=(n, n, n))).astype(np.float32)
    vy = (flow_speed * base[1] * atten + 0.003 * rng.normal(size=(n, n, n))).astype(np.float32)
    vz = (flow_speed * base[2] * atten + 0.003 * rng.normal(size=(n, n, n))).astype(np.float32)
    shear = clip01(normalize(gradmag3(vx) + gradmag3(vy) + gradmag3(vz)) + 0.35 * outside)

    source_shape = clip01(pore_mask * normalize(0.55 * vent + 0.35 * pore + 0.10 * mineral))
    return Environment3D(pore, mineral, residence, vent, pressure, vx, vy, vz, shear, source_shape)


def make_open_environment(n: int, rng: np.random.Generator, source_regime: str) -> Environment3D:
    # Open sea: weak spatial structure, directional flow, peripheral sponge handled separately.
    pore = np.zeros((n, n, n), dtype=np.float32)
    mineral = np.zeros_like(pore)
    residence = np.full_like(pore, 0.04, dtype=np.float32)
    vent = np.zeros_like(pore)
    pressure = normalize(gaussian_mixture_3d(n, rng, max(2, n // 18), (0.15, 0.30)))

    base = rng.normal(0, 1, size=3)
    base = base / (np.linalg.norm(base) + 1e-12)
    speed = 0.030
    vx = (speed * base[0] + 0.006 * rng.normal(size=(n, n, n))).astype(np.float32)
    vy = (speed * base[1] + 0.006 * rng.normal(size=(n, n, n))).astype(np.float32)
    vz = (speed * base[2] + 0.006 * rng.normal(size=(n, n, n))).astype(np.float32)
    shear = normalize(gradmag3(vx) + gradmag3(vy) + gradmag3(vz))

    if source_regime == "continuous":
        source_shape = normalize(gaussian_mixture_3d(n, rng, max(3, n // 16), (0.12, 0.25)))
    elif source_regime == "low":
        source_shape = 0.15 * normalize(gaussian_mixture_3d(n, rng, max(3, n // 16), (0.12, 0.25)))
    elif source_regime == "none":
        source_shape = np.zeros((n, n, n), dtype=np.float32)
    else:
        raise ValueError(f"unknown source_regime: {source_regime}")

    return Environment3D(pore, mineral, residence, vent, pressure, vx, vy, vz, shear, source_shape.astype(np.float32))


@dataclass
class Fields:
    R: np.ndarray
    L: np.ndarray
    H: np.ndarray
    X: np.ndarray
    M: np.ndarray
    B: np.ndarray
    T: np.ndarray


def init_fields(n: int, rng: np.random.Generator, env: Environment3D) -> Fields:
    noise = lambda scale: (scale * rng.random((n, n, n), dtype=np.float32)).astype(np.float32)
    # Keep background lipid/resource low. Otherwise the outside space slowly fills
    # and separated pores bridge into one component.
    local = np.clip(env.source_shape + 0.50 * env.mineral + 0.35 * env.residence, 0.0, 1.0).astype(np.float32)
    R = clip01(0.035 + 0.60 * env.source_shape + 0.06 * noise(1.0)).astype(np.float32)
    H = clip01(0.025 + 0.52 * env.vent + 0.04 * noise(1.0)).astype(np.float32)
    X = clip01(0.015 + 0.018 * noise(1.0)).astype(np.float32)
    L = clip01(0.018 + 0.62 * env.mineral + 0.30 * env.residence + 0.035 * noise(1.0)).astype(np.float32)
    M = (0.035 * rng.normal(size=(n, n, n)) * (0.25 + local)).astype(np.float32)
    B = clip01(0.004 + 0.035 * env.mineral + 0.010 * noise(1.0)).astype(np.float32)
    T = clip01(0.002 + 0.014 * env.mineral + 0.005 * noise(1.0)).astype(np.float32)
    return Fields(R=R, L=L, H=H, X=X, M=M, B=B, T=T)


def permeability(B: np.ndarray, T: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    P = np.exp(-3.0 * T - 1.8 * B - 0.55 * pressure)
    return np.clip(P, 0.04, 1.0).astype(np.float32)


def update_fields(f: Fields, env: Environment3D, rng: np.random.Generator, cfg: SimConfig, phase: str, advect_fn=advect_upwind) -> None:
    dt = cfg.dt
    B, T, M, R, L, H, X = f.B, f.T, f.M, f.R, f.L, f.H, f.X
    P = permeability(B, T, env.pressure)

    # Effective diffusion.
    interface = gradmag3(B + 0.5 * T)
    diff_mod = np.clip(P * (1.0 - 0.35 * B) + 0.10 * env.residence, 0.02, 1.0).astype(np.float32)
    D_R = 0.065 * diff_mod
    D_L = 0.035 * diff_mod
    D_H = 0.052 * diff_mod
    D_X = 0.060 * diff_mod
    D_M = 0.024 * diff_mod

    if phase == "confined_pre":
        SR = 0.018 * env.source_shape
        SH = 0.015 * env.vent
        SX = np.zeros_like(R)
        flow_strength = 0.30
    elif phase == "confined_gen":
        SR = 0.040 * env.source_shape
        SH = 0.028 * env.vent
        SX = 0.010 * normalize(gradmag3(env.source_shape))
        flow_strength = 0.70
    elif phase == "open":
        SR = 0.035 * env.source_shape
        SH = 0.008 * env.source_shape
        SX = np.zeros_like(R)
        flow_strength = 1.00
    else:
        raise ValueError(phase)

    adv_R = advect_fn(R, env.vx, env.vy, env.vz)
    adv_L = advect_fn(L, env.vx, env.vy, env.vz)
    adv_H = advect_fn(H, env.vx, env.vy, env.vz)
    adv_X = advect_fn(X, env.vx, env.vy, env.vz)
    adv_M = advect_fn(M, env.vx, env.vy, env.vz)
    adv_B = advect_fn(B, env.vx, env.vy, env.vz)
    adv_T = advect_fn(T, env.vx, env.vy, env.vz)

    # Resource/hydrothermal/waste/lipid.
    lipid_consumption = 0.055 * L * (0.35 * B + 0.65 * T + 0.05)
    lipid_release = 0.026 * X * (B + 0.5 * T)

    dR = D_R * laplacian6(R) + flow_strength * 0.018 * adv_R + SR - 0.010 * R - 0.020 * R * (B + 0.25 * np.abs(M))
    dH = D_H * laplacian6(H) + flow_strength * 0.018 * adv_H + SH - 0.009 * H - 0.010 * H * B
    dX = D_X * laplacian6(X) + flow_strength * 0.018 * adv_X + SX + 0.018 * (R * B + H * T) - 0.028 * X
    dL = D_L * laplacian6(L) + flow_strength * 0.018 * adv_L - lipid_consumption + lipid_release - 0.009 * L

    # Delayed-retention field surrogate: local M recurrence plus field forcing.
    # This is not an evolutionary or reward term. It only preserves delayed local-field dynamics.
    gamma_i = 1.0 + 0.40 * T + 0.15 * X + 0.25 * env.shear
    beta_i = 2.0 + 0.50 * T - 0.45 * P - 0.20 * env.shear
    M_neighbor = mean6(M)
    noise = 0.045 * rng.normal(size=M.shape).astype(np.float32)
    dM = (
        D_M * laplacian6(M)
        + flow_strength * 0.018 * adv_M
        - gamma_i * M
        + beta_i * np.tanh(M_neighbor)
        + 0.024 * R
        - 0.017 * X
        + noise
    )

    # Membrane density and thickness.
    chem = 0.45 * R + 0.35 * H - 0.30 * X + 0.25 * L
    chem_grad = gradmag3(chem)
    basal_agg = 0.018 * L * (R + 0.2)
    residence_agg = 0.085 * L * env.residence
    mineral_agg = 0.040 * L * env.mineral
    interface_agg = 0.055 * L * interface
    self_agg = 0.050 * B * (1.0 - B)
    cohesion = 0.045 * (mean6(B) - B)
    m_feedback = 0.018 * np.tanh(np.abs(M)) * L
    frag = 0.060 * B * (0.60 * X + 0.50 * env.shear + 0.35 * gradmag3(env.pressure))
    turnover_B = 0.009 * B
    water_carry_B = flow_strength * 0.010 * adv_B

    dB = basal_agg + residence_agg + mineral_agg + interface_agg + self_agg + cohesion + m_feedback - frag - turnover_B + water_carry_B

    thick_growth = (
        0.055 * L * B
        + 0.080 * B * np.maximum(mean6(B) - 0.10, 0.0)
        + 0.016 * np.abs(M) * B
        + 0.020 * env.pressure * B
    )
    thick_loss = (
        0.058 * env.shear * T
        + 0.038 * gradmag3(env.pressure) * T
        + 0.090 * sigmoid(18.0 * (T - 0.34)) * T
        + 0.024 * interface * T
        + 0.016 * T
    )
    smooth_T = 0.014 * (mean6(T) - T)
    water_carry_T = flow_strength * 0.010 * adv_T
    dT = thick_growth - thick_loss + smooth_T + water_carry_T

    f.R = clip01(R + dt * dR).astype(np.float32)
    f.H = clip01(H + dt * dH).astype(np.float32)
    f.X = clip01(X + dt * dX).astype(np.float32)
    f.L = clip01(L + dt * dL).astype(np.float32)
    f.M = np.clip(M + dt * dM, -8.0, 8.0).astype(np.float32)
    f.B = clip01(B + dt * dB).astype(np.float32)
    f.T = clip01(T + dt * dT).astype(np.float32)

    # Absorbing peripheral sponge for open sea.
    if phase == "open":
        s = sponge_mask(f.B.shape, width=5)
        absorption = 0.12 * s
        for name in ["R", "L", "H", "X", "B", "T"]:
            arr = getattr(f, name)
            setattr(f, name, (arr * (1.0 - absorption)).astype(np.float32))
        f.M = (f.M * (1.0 - absorption)).astype(np.float32)


def detect_labels(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    if SCIPY_AVAILABLE:
        structure = np.ones((3, 3, 3), dtype=np.int8)
        labels, nlab = ndi.label(mask, structure=structure)
        return labels.astype(np.int32, copy=False), int(nlab)

    # Fallback connected components. Slower but dependency-free.
    labels = np.zeros(mask.shape, dtype=np.int32)
    n = mask.shape[0]
    current = 0
    coords = np.argwhere(mask)
    neigh = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == dy == dz == 0)]
    for x, y, z in coords:
        if labels[x, y, z] != 0:
            continue
        current += 1
        stack = [(int(x), int(y), int(z))]
        labels[x, y, z] = current
        while stack:
            cx, cy, cz = stack.pop()
            for dx, dy, dz in neigh:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if 0 <= nx < n and 0 <= ny < n and 0 <= nz < n and mask[nx, ny, nz] and labels[nx, ny, nz] == 0:
                    labels[nx, ny, nz] = current
                    stack.append((nx, ny, nz))
    return labels, current


def filter_small_labels(labels: np.ndarray, min_size: int) -> Tuple[np.ndarray, Dict[int, int]]:
    if labels.max() == 0:
        return labels.astype(np.int32), {}
    counts = np.bincount(labels.ravel())
    keep_old = np.where(counts >= min_size)[0]
    keep_old = keep_old[keep_old != 0]
    out = np.zeros_like(labels, dtype=np.int32)
    size_map = {}
    new_id = 0
    for old in keep_old:
        new_id += 1
        out[labels == old] = new_id
        size_map[new_id] = int(counts[old])
    return out, size_map


def summarize_components_for_release(B: np.ndarray, T: np.ndarray, cfg: SimConfig) -> Dict[str, Any]:
    """Return component count and largest fraction under the fixed membrane definition."""
    mask = (B >= cfg.b_threshold) & (T >= cfg.t_threshold)
    labels, _ = detect_labels(mask)
    labels, sizes = filter_small_labels(labels, cfg.min_component_voxels)
    total = int(sum(sizes.values()))
    count = int(len(sizes))
    largest = int(max(sizes.values())) if sizes else 0
    frac = float(largest / total) if total > 0 else 0.0
    return {
        "component_count": count,
        "total_membrane_voxels": total,
        "largest_component_voxels": largest,
        "largest_component_fraction": frac,
    }


def release_trigger_met(summary: Dict[str, Any], cfg: SimConfig) -> bool:
    """Fixed trigger: release only when multiple components exist before saturation."""
    if int(summary["component_count"]) < int(cfg.release_min_components):
        return False
    max_frac = float(cfg.release_max_largest_fraction)
    if max_frac > 0 and float(summary["largest_component_fraction"]) > max_frac:
        return False
    return True


def largest_component_centroid(B: np.ndarray, T: np.ndarray, cfg: SimConfig) -> Optional[np.ndarray]:
    mask = (B >= cfg.b_threshold) & (T >= cfg.t_threshold)
    labels, _ = detect_labels(mask)
    labels, sizes = filter_small_labels(labels, cfg.min_component_voxels)
    if not sizes:
        return None
    lab = max(sizes, key=sizes.get)
    pts = np.argwhere(labels == lab)
    if pts.size == 0:
        return None
    w = (B[labels == lab] + T[labels == lab] + 1e-6).astype(np.float64)
    c = np.average(pts.astype(np.float64), axis=0, weights=w)
    return c


def recenter_fields_if_needed(f: Fields, cfg: SimConfig, logger: ProgressLogger, step: int, margin: int = 14) -> Tuple[int, int, int]:
    if not cfg.recenter:
        return (0, 0, 0)
    c = largest_component_centroid(f.B, f.T, cfg)
    if c is None:
        return (0, 0, 0)
    n = f.B.shape[0]
    center = np.array([n // 2, n // 2, n // 2], dtype=np.float64)
    low_margin = np.min(c)
    high_margin = np.min((n - 1) - c)
    if low_margin >= margin and high_margin >= margin:
        return (0, 0, 0)
    shift = np.round(center - c).astype(int)
    if np.linalg.norm(shift) < 6:
        return (0, 0, 0)
    for name in ["R", "L", "H", "X", "M", "B", "T"]:
        arr = getattr(f, name)
        setattr(f, name, np.roll(arr, shift=tuple(shift), axis=(0, 1, 2)).astype(np.float32))
    logger.log(f"open step {step}: recentered fields by shift={tuple(int(x) for x in shift)}")
    return tuple(int(x) for x in shift)


def embed_center(f: Fields, open_n: int) -> Fields:
    old_n = f.B.shape[0]
    if open_n < old_n:
        raise ValueError("open_n must be >= confined_n")
    start = (open_n - old_n) // 2
    sl = slice(start, start + old_n)

    def embed(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
        out = np.full((open_n, open_n, open_n), fill, dtype=np.float32)
        out[sl, sl, sl] = arr.astype(np.float32)
        return out

    return Fields(
        R=embed(f.R, 0.04),
        L=embed(f.L, 0.00),
        H=embed(f.H, 0.02),
        X=embed(f.X, 0.00),
        M=embed(f.M, 0.00),
        B=embed(f.B, 0.00),
        T=embed(f.T, 0.00),
    )


def save_snapshot(snapdir: Path, step: int, f: Fields, cfg: SimConfig, recenter_shift: Tuple[int, int, int] = (0, 0, 0)) -> None:
    snapdir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if cfg.save_float16 else np.float32
    np.savez_compressed(
        snapdir / f"open_sea_step_{step:06d}.npz",
        step=np.array(step, dtype=np.int32),
        recenter_shift=np.array(recenter_shift, dtype=np.int32),
        B=f.B.astype(dtype),
        T=f.T.astype(dtype),
        M=f.M.astype(dtype),
        R=f.R.astype(dtype),
        X=f.X.astype(dtype),
    )


# =============================================================================
# Component trait extraction and tracking
# =============================================================================

TRAIT_COLUMNS = [
    "volume",
    "surface_area",
    "compactness",
    "elongation",
    "planarity",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "mean_B",
    "mean_T",
    "var_T",
    "mean_abs_M",
    "std_abs_M",
    "mean_R",
    "mean_X",
    "lumen_count",
    "lumen_volume",
    "lumen_volume_fraction",
    "step_displacement",
    "cumulative_path",
    "net_displacement",
    "persistence_ratio",
]


def component_surface_area(mask: np.ndarray) -> int:
    # 6-neighbor exposed faces.
    vol = int(mask.sum())
    if vol == 0:
        return 0
    area = 0
    for axis in range(3):
        area += int(np.sum(mask & (~np.roll(mask, 1, axis=axis))))
        area += int(np.sum(mask & (~np.roll(mask, -1, axis=axis))))
    return area


def component_shape_traits(points: np.ndarray, max_points: int = 6000) -> Tuple[float, float, Tuple[int, int, int]]:
    if points.shape[0] < 3:
        return 0.0, 0.0, (0, 0, 0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    bbox = tuple(int(x) for x in (maxs - mins + 1))

    pts = points
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[idx]
    centered = pts.astype(np.float64) - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(pts) - 1, 1)
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 0.0))[::-1]
    l1, l2, l3 = vals + 1e-12
    elongation = float((l1 - l2) / l1)
    planarity = float((l2 - l3) / l1)
    return elongation, planarity, bbox


def lumen_traits(component_mask: np.ndarray, min_lumen_voxels: int) -> Tuple[int, int]:
    # Internal lumens inside component bounding box. Non-membrane regions not connected to bbox border.
    if component_mask.sum() == 0:
        return 0, 0
    coords = np.argwhere(component_mask)
    mins = np.maximum(coords.min(axis=0) - 1, 0)
    maxs = np.minimum(coords.max(axis=0) + 2, np.array(component_mask.shape))
    sub = component_mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    non = ~sub

    labels, nlab = detect_labels(non)
    if nlab == 0:
        return 0, 0

    border_labels = set()
    for face in [
        labels[0, :, :], labels[-1, :, :],
        labels[:, 0, :], labels[:, -1, :],
        labels[:, :, 0], labels[:, :, -1],
    ]:
        border_labels.update(int(x) for x in np.unique(face) if x != 0)

    counts = np.bincount(labels.ravel())
    lumen_count = 0
    lumen_volume = 0
    for lab in range(1, len(counts)):
        if lab in border_labels:
            continue
        c = int(counts[lab])
        if c >= min_lumen_voxels:
            lumen_count += 1
            lumen_volume += c
    return lumen_count, lumen_volume


@dataclass
class TrackState:
    track_id: int
    first_step: int
    last_step: int
    first_centroid_x: float
    first_centroid_y: float
    first_centroid_z: float
    prev_centroid_x: float
    prev_centroid_y: float
    prev_centroid_z: float
    cumulative_path: float
    observations: int


class ComponentTracker:
    def __init__(self, min_overlap_voxels: int = 5):
        self.next_track_id = 1
        self.prev_labels: Optional[np.ndarray] = None
        self.prev_label_to_track: Dict[int, int] = {}
        self.tracks: Dict[int, TrackState] = {}
        self.min_overlap_voxels = int(min_overlap_voxels)

    def assign_tracks(
        self,
        step: int,
        labels: np.ndarray,
        centroids: Dict[int, Tuple[float, float, float]],
    ) -> Tuple[Dict[int, int], List[Dict[str, Any]], List[Dict[str, Any]]]:
        events: List[Dict[str, Any]] = []
        overlap_rows: List[Dict[str, Any]] = []
        curr_labels = [int(x) for x in np.unique(labels) if x != 0]
        curr_to_track: Dict[int, int] = {}

        if self.prev_labels is None:
            for lab in curr_labels:
                tid = self._new_track(step, centroids[lab])
                curr_to_track[lab] = tid
                events.append({"step": step, "event_type": "emergence", "track_id": tid, "component_label": lab})
            self.prev_labels = labels.copy()
            self.prev_label_to_track = curr_to_track
            return curr_to_track, events, overlap_rows

        prev = self.prev_labels
        curr = labels
        n_curr = int(curr.max()) + 1
        both = (prev > 0) & (curr > 0)
        pairs: List[Tuple[int, int, int]] = []
        if np.any(both):
            code = prev[both].astype(np.int64) * n_curr + curr[both].astype(np.int64)
            uniq, counts = np.unique(code, return_counts=True)
            for u, c in zip(uniq, counts):
                pl = int(u // n_curr)
                cl = int(u % n_curr)
                if int(c) >= self.min_overlap_voxels:
                    pairs.append((pl, cl, int(c)))
                    overlap_rows.append({
                        "step": step,
                        "prev_label": pl,
                        "curr_label": cl,
                        "prev_track_id": self.prev_label_to_track.get(pl, -1),
                        "overlap_voxels": int(c),
                    })

        by_curr: Dict[int, List[Tuple[int, int]]] = {}
        by_prev: Dict[int, List[Tuple[int, int]]] = {}
        for pl, cl, c in pairs:
            by_curr.setdefault(cl, []).append((pl, c))
            by_prev.setdefault(pl, []).append((cl, c))

        assigned_prev_for_curr: Dict[int, int] = {}
        for cl in curr_labels:
            candidates = by_curr.get(cl, [])
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_prev, best_overlap = candidates[0]
                prev_tid = self.prev_label_to_track.get(best_prev)
                if prev_tid is not None:
                    curr_to_track[cl] = prev_tid
                    assigned_prev_for_curr[cl] = best_prev
                    self._update_track(prev_tid, step, centroids[cl])
                    events.append({
                        "step": step,
                        "event_type": "continuation",
                        "track_id": prev_tid,
                        "component_label": cl,
                        "parent_prev_label": best_prev,
                        "overlap_voxels": best_overlap,
                    })
                    if len(candidates) > 1:
                        events.append({
                            "step": step,
                            "event_type": "fusion_into_component",
                            "track_id": prev_tid,
                            "component_label": cl,
                            "n_previous_components": len(candidates),
                        })
                else:
                    tid = self._new_track(step, centroids[cl])
                    curr_to_track[cl] = tid
                    events.append({"step": step, "event_type": "emergence", "track_id": tid, "component_label": cl})
            else:
                tid = self._new_track(step, centroids[cl])
                curr_to_track[cl] = tid
                events.append({"step": step, "event_type": "emergence", "track_id": tid, "component_label": cl})

        # Fission: one previous component overlapping multiple current components.
        for pl, currs in by_prev.items():
            if len(currs) >= 2:
                prev_tid = self.prev_label_to_track.get(pl, -1)
                events.append({
                    "step": step,
                    "event_type": "fission_from_previous",
                    "parent_track_id": prev_tid,
                    "prev_label": pl,
                    "n_current_components": len(currs),
                    "child_track_ids": ";".join(str(curr_to_track.get(cl, -1)) for cl, _ in currs),
                })

        # Extinction: previous labels with no overlap.
        prev_labels = [int(x) for x in np.unique(prev) if x != 0]
        prev_with_child = set(by_prev.keys())
        for pl in prev_labels:
            if pl not in prev_with_child:
                tid = self.prev_label_to_track.get(pl, -1)
                events.append({
                    "step": step,
                    "event_type": "extinction_or_disappearance",
                    "track_id": tid,
                    "prev_label": pl,
                })

        self.prev_labels = labels.copy()
        self.prev_label_to_track = curr_to_track
        return curr_to_track, events, overlap_rows

    def _new_track(self, step: int, centroid: Tuple[float, float, float]) -> int:
        tid = self.next_track_id
        self.next_track_id += 1
        cx, cy, cz = centroid
        self.tracks[tid] = TrackState(
            track_id=tid,
            first_step=step,
            last_step=step,
            first_centroid_x=float(cx),
            first_centroid_y=float(cy),
            first_centroid_z=float(cz),
            prev_centroid_x=float(cx),
            prev_centroid_y=float(cy),
            prev_centroid_z=float(cz),
            cumulative_path=0.0,
            observations=1,
        )
        return tid

    def _update_track(self, tid: int, step: int, centroid: Tuple[float, float, float]) -> None:
        st = self.tracks[tid]
        cx, cy, cz = centroid
        d = math.sqrt(
            (float(cx) - st.prev_centroid_x) ** 2
            + (float(cy) - st.prev_centroid_y) ** 2
            + (float(cz) - st.prev_centroid_z) ** 2
        )
        st.cumulative_path += d
        st.prev_centroid_x = float(cx)
        st.prev_centroid_y = float(cy)
        st.prev_centroid_z = float(cz)
        st.last_step = step
        st.observations += 1


def extract_component_traits_from_snapshot(
    snapshot_path: Path,
    cfg: SimConfig,
    tracker: ComponentTracker,
    logger: Optional[ProgressLogger] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = read_npz(snapshot_path)
    step = int(np.array(data["step"]).item()) if "step" in data else int(snapshot_path.stem.split("_")[-1])
    B = data["B"].astype(np.float32)
    T = data["T"].astype(np.float32)
    M = data["M"].astype(np.float32)
    R = data.get("R", np.zeros_like(B)).astype(np.float32)
    X = data.get("X", np.zeros_like(B)).astype(np.float32)

    mask = (B >= cfg.b_threshold) & (T >= cfg.t_threshold)
    labels, _ = detect_labels(mask)
    labels, sizes = filter_small_labels(labels, cfg.min_component_voxels)

    centroids: Dict[int, Tuple[float, float, float]] = {}
    prelim: Dict[int, Dict[str, Any]] = {}

    labs = [int(x) for x in np.unique(labels) if x != 0]
    for lab in labs:
        cmask = labels == lab
        pts = np.argwhere(cmask)
        vol = int(pts.shape[0])
        w = (B[cmask] + T[cmask] + 1e-6).astype(np.float64)
        centroid = np.average(pts.astype(np.float64), axis=0, weights=w)
        cx, cy, cz = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
        centroids[lab] = (cx, cy, cz)

        surface = component_surface_area(cmask)
        compactness = float((vol ** (2.0 / 3.0)) / (surface + 1e-12))
        elongation, planarity, bbox = component_shape_traits(pts)
        lumen_count, lumen_volume = lumen_traits(cmask, cfg.min_lumen_voxels)
        prelim[lab] = {
            "step": step,
            "component_label": lab,
            "volume": vol,
            "surface_area": surface,
            "compactness": compactness,
            "elongation": elongation,
            "planarity": planarity,
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_z": bbox[2],
            "centroid_x": cx,
            "centroid_y": cy,
            "centroid_z": cz,
            "mean_B": float(np.mean(B[cmask])),
            "mean_T": float(np.mean(T[cmask])),
            "var_T": float(np.var(T[cmask])),
            "mean_abs_M": float(np.mean(np.abs(M[cmask]))),
            "std_abs_M": float(np.std(np.abs(M[cmask]))),
            "mean_R": float(np.mean(R[cmask])),
            "mean_X": float(np.mean(X[cmask])),
            "lumen_count": int(lumen_count),
            "lumen_volume": int(lumen_volume),
            "lumen_volume_fraction": float(lumen_volume / max(vol, 1)),
        }

    curr_to_track, events, overlap_rows = tracker.assign_tracks(step, labels, centroids)

    rows: List[Dict[str, Any]] = []
    for lab, row in prelim.items():
        tid = curr_to_track[lab]
        st = tracker.tracks[tid]
        cx, cy, cz = centroids[lab]
        prev_d = math.sqrt(
            (cx - st.prev_centroid_x) ** 2
            + (cy - st.prev_centroid_y) ** 2
            + (cz - st.prev_centroid_z) ** 2
        )
        # After tracker update, prev_centroid is current. The step displacement is recovered from event if possible.
        # For first observation it is zero. For continued tracks, use current cumulative minus previous path is unavailable;
        # therefore compute step displacement from current and previous event not stored. Keep a conservative value:
        step_displacement = 0.0
        if st.observations > 1:
            # Use final event overlap as continuation marker; exact step distance is reconstructed later in change summary.
            step_displacement = np.nan

        net = math.sqrt(
            (cx - st.first_centroid_x) ** 2
            + (cy - st.first_centroid_y) ** 2
            + (cz - st.first_centroid_z) ** 2
        )
        row["track_id"] = tid
        row["track_first_step"] = st.first_step
        row["track_observations"] = st.observations
        row["cumulative_path"] = float(st.cumulative_path)
        row["net_displacement"] = float(net)
        row["persistence_ratio"] = float(net / st.cumulative_path) if st.cumulative_path > 1e-12 else 0.0
        row["step_displacement"] = step_displacement
        rows.append(row)

    return rows, events, overlap_rows


def compute_population_diversity(component_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not component_rows:
        return []

    # Build global z-score from all component observations.
    X = np.array([[float(r.get(c, 0.0)) if r.get(c, "") != "" and not isinstance(r.get(c), str) else 0.0 for c in TRAIT_COLUMNS] for r in component_rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd

    for i, r in enumerate(component_rows):
        r["_zvec"] = Z[i]

    rows: List[Dict[str, Any]] = []
    steps = sorted(set(int(r["step"]) for r in component_rows))
    for step in steps:
        idx = [i for i, r in enumerate(component_rows) if int(r["step"]) == step]
        n = len(idx)
        sub = Z[idx, :]
        if n <= 1:
            mean_pairwise = 0.0
            median_pairwise = 0.0
            trait_dispersion = 0.0
        else:
            # Efficient pairwise distance for moderate component counts.
            diffs = sub[:, None, :] - sub[None, :, :]
            dmat = np.sqrt(np.sum(diffs * diffs, axis=2))
            tri = dmat[np.triu_indices(n, k=1)]
            mean_pairwise = float(np.mean(tri)) if tri.size else 0.0
            median_pairwise = float(np.median(tri)) if tri.size else 0.0
            trait_dispersion = float(np.mean(np.sum((sub - sub.mean(axis=0)) ** 2, axis=1)))

        volumes = [float(component_rows[i]["volume"]) for i in idx]
        mean_abs_M = [float(component_rows[i]["mean_abs_M"]) for i in idx]
        mean_T = [float(component_rows[i]["mean_T"]) for i in idx]
        lumens = [float(component_rows[i]["lumen_count"]) for i in idx]

        rows.append({
            "step": step,
            "component_count": n,
            "total_component_volume": float(np.sum(volumes)),
            "largest_component_volume": float(np.max(volumes)) if volumes else 0.0,
            "mean_pairwise_trait_distance": mean_pairwise,
            "median_pairwise_trait_distance": median_pairwise,
            "trait_dispersion": trait_dispersion,
            "volume_cv": float(np.std(volumes) / (np.mean(volumes) + 1e-12)) if volumes else 0.0,
            "mean_T_cv": float(np.std(mean_T) / (np.mean(mean_T) + 1e-12)) if mean_T else 0.0,
            "mean_abs_M_cv": float(np.std(mean_abs_M) / (np.mean(mean_abs_M) + 1e-12)) if mean_abs_M else 0.0,
            "lumen_count_mean": float(np.mean(lumens)) if lumens else 0.0,
            "lumen_count_max": float(np.max(lumens)) if lumens else 0.0,
        })
    return rows


def compute_change_summary(component_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not component_rows:
        return []
    by_track: Dict[int, List[Dict[str, Any]]] = {}
    for r in component_rows:
        by_track.setdefault(int(r["track_id"]), []).append(r)

    rows: List[Dict[str, Any]] = []
    for tid, rs in sorted(by_track.items()):
        rs = sorted(rs, key=lambda x: int(x["step"]))
        first = rs[0]
        last = rs[-1]
        coords = np.array([[float(r["centroid_x"]), float(r["centroid_y"]), float(r["centroid_z"])] for r in rs], dtype=np.float64)
        if len(coords) > 1:
            step_distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
            path = float(np.sum(step_distances))
        else:
            path = 0.0
        net = float(np.sqrt(np.sum((coords[-1] - coords[0]) ** 2)))
        row = {
            "track_id": tid,
            "n_observations": len(rs),
            "first_step": int(first["step"]),
            "last_step": int(last["step"]),
            "duration_steps": int(last["step"]) - int(first["step"]),
            "path_length": path,
            "net_displacement": net,
            "persistence_ratio": float(net / path) if path > 1e-12 else 0.0,
        }
        for c in [
            "volume", "surface_area", "compactness", "elongation", "planarity",
            "mean_B", "mean_T", "var_T", "mean_abs_M", "std_abs_M",
            "mean_R", "mean_X", "lumen_count", "lumen_volume_fraction"
        ]:
            row[f"{c}_first"] = first.get(c, 0.0)
            row[f"{c}_last"] = last.get(c, 0.0)
            row[f"{c}_delta"] = float(last.get(c, 0.0)) - float(first.get(c, 0.0))
            vals = [float(r.get(c, 0.0)) for r in rs]
            row[f"{c}_mean"] = float(np.mean(vals)) if vals else 0.0
            row[f"{c}_sd"] = float(np.std(vals)) if vals else 0.0
        rows.append(row)
    return rows


def compute_pca_rows(component_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not component_rows:
        return []
    X = np.array([[float(r.get(c, 0.0)) if r.get(c, "") != "" and not isinstance(r.get(c), str) else 0.0 for c in TRAIT_COLUMNS] for r in component_rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    if Z.shape[0] < 2:
        pcs = np.zeros((Z.shape[0], 2), dtype=np.float64)
        explained = [0.0, 0.0]
    else:
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        pcs = U[:, :2] * S[:2]
        var = S * S
        total = float(np.sum(var))
        explained = [float(var[i] / total) if total > 0 and i < len(var) else 0.0 for i in range(2)]
    rows = []
    for r, pc in zip(component_rows, pcs):
        rows.append({
            "step": int(r["step"]),
            "track_id": int(r["track_id"]),
            "component_label": int(r["component_label"]),
            "PC1": float(pc[0]),
            "PC2": float(pc[1]) if len(pc) > 1 else 0.0,
            "PC1_explained_fraction": explained[0],
            "PC2_explained_fraction": explained[1],
            "volume": r.get("volume", 0),
            "mean_T": r.get("mean_T", 0.0),
            "mean_abs_M": r.get("mean_abs_M", 0.0),
            "lumen_count": r.get("lumen_count", 0),
        })
    return rows


def make_figures(outdir: Path, diversity_rows: List[Dict[str, Any]], pca_rows: List[Dict[str, Any]]) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    figdir = ensure_dir(outdir / "figures")

    if diversity_rows:
        steps = [r["step"] for r in diversity_rows]
        counts = [r["component_count"] for r in diversity_rows]
        div = [r["mean_pairwise_trait_distance"] for r in diversity_rows]
        largest = [r["largest_component_volume"] for r in diversity_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, counts, marker="o")
        plt.xlabel("Open-sea step")
        plt.ylabel("Detected component count")
        plt.title("Component count after open-sea release")
        plt.tight_layout()
        plt.savefig(figdir / "component_count_timeseries.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(steps, div, marker="o")
        plt.xlabel("Open-sea step")
        plt.ylabel("Mean pairwise trait distance")
        plt.title("Population trait diversity after open-sea release")
        plt.tight_layout()
        plt.savefig(figdir / "diversity_timeseries.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(steps, largest, marker="o")
        plt.xlabel("Open-sea step")
        plt.ylabel("Largest component volume")
        plt.title("Largest component volume after open-sea release")
        plt.tight_layout()
        plt.savefig(figdir / "largest_component_volume_timeseries.png", dpi=200)
        plt.close()

    if pca_rows:
        pc1 = [r["PC1"] for r in pca_rows]
        pc2 = [r["PC2"] for r in pca_rows]
        steps = [r["step"] for r in pca_rows]
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(pc1, pc2, c=steps, s=16)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Component trait space after open-sea release")
        plt.colorbar(sc, label="Open-sea step")
        plt.tight_layout()
        plt.savefig(figdir / "pca_trait_space.png", dpi=200)
        plt.close()


def write_report(
    outdir: Path,
    cfg: SimConfig,
    component_rows: List[Dict[str, Any]],
    diversity_rows: List[Dict[str, Any]],
    change_rows: List[Dict[str, Any]],
    event_rows: List[Dict[str, Any]],
) -> None:
    report = outdir / "change_and_diversity_report.md"

    n_snap = len(set(r["step"] for r in component_rows)) if component_rows else 0
    n_comp_obs = len(component_rows)
    n_tracks = len(set(r["track_id"] for r in component_rows)) if component_rows else 0

    if diversity_rows:
        first_div = diversity_rows[0]["mean_pairwise_trait_distance"]
        last_div = diversity_rows[-1]["mean_pairwise_trait_distance"]
        max_div = max(r["mean_pairwise_trait_distance"] for r in diversity_rows)
        first_count = diversity_rows[0]["component_count"]
        last_count = diversity_rows[-1]["component_count"]
        max_count = max(r["component_count"] for r in diversity_rows)
    else:
        first_div = last_div = max_div = 0.0
        first_count = last_count = max_count = 0

    long_tracks = [r for r in change_rows if int(r["n_observations"]) >= 3]
    if long_tracks:
        mean_path = float(np.mean([float(r["path_length"]) for r in long_tracks]))
        mean_volume_delta = float(np.mean([float(r["volume_delta"]) for r in long_tracks]))
        mean_T_delta = float(np.mean([float(r["mean_T_delta"]) for r in long_tracks]))
        mean_M_delta = float(np.mean([float(r["mean_abs_M_delta"]) for r in long_tracks]))
    else:
        mean_path = mean_volume_delta = mean_T_delta = mean_M_delta = 0.0

    event_counts: Dict[str, int] = {}
    for e in event_rows:
        et = str(e.get("event_type", "unknown"))
        event_counts[et] = event_counts.get(et, 0) + 1

    lines = []
    lines.append("# Open-Sea Change and Diversity Assay\n")
    lines.append(f"Version: `{VERSION}`\n")
    lines.append("\n## Scope\n")
    lines.append("This analysis tests change and diversity after open-sea release. It does not test evolution, selection, adaptation, species formation, or genetic inheritance.\n")
    lines.append("\n## Fixed component definition\n")
    lines.append(f"- Membrane voxel: `B >= {cfg.b_threshold}` and `T >= {cfg.t_threshold}`\n")
    lines.append("- Connectivity: 26-connected 3D components\n")
    lines.append(f"- Minimum component size: `{cfg.min_component_voxels}` voxels\n")
    lines.append(f"- Minimum lumen size: `{cfg.min_lumen_voxels}` voxels\n")
    lines.append("- Tracking: voxel-level overlap between consecutive open-sea snapshots\n")
    lines.append("\n## Simulation settings\n")
    for k, v in asdict(cfg).items():
        lines.append(f"- `{k}`: `{v}`\n")
    lines.append("\n## Main observations\n")
    lines.append(f"- Open-sea snapshots with detected components: `{n_snap}`\n")
    lines.append(f"- Component observations: `{n_comp_obs}`\n")
    lines.append(f"- Component tracks: `{n_tracks}`\n")
    lines.append(f"- Component count: first `{first_count}`, last `{last_count}`, maximum `{max_count}`\n")
    lines.append(f"- Mean pairwise trait distance: first `{first_div:.6f}`, last `{last_div:.6f}`, maximum `{max_div:.6f}`\n")
    lines.append(f"- Mean path length among tracks with >=3 observations: `{mean_path:.6f}`\n")
    lines.append(f"- Mean volume change among tracks with >=3 observations: `{mean_volume_delta:.6f}`\n")
    lines.append(f"- Mean membrane-thickness change among tracks with >=3 observations: `{mean_T_delta:.6f}`\n")
    lines.append(f"- Mean internal abs(M) change among tracks with >=3 observations: `{mean_M_delta:.6f}`\n")
    lines.append("\n## Event counts\n")
    if event_counts:
        for k, v in sorted(event_counts.items()):
            lines.append(f"- `{k}`: `{v}`\n")
    else:
        lines.append("- No component events detected.\n")

    lines.append("\n## Output files\n")
    for fn in [
        "component_traits_over_time.csv",
        "component_overlap_audit.csv",
        "component_events.csv",
        "population_diversity_timeseries.csv",
        "component_change_summary.csv",
        "pca_component_trait_space.csv",
    ]:
        lines.append(f"- `{fn}`\n")
    lines.append("\n## Interpretation boundary\n")
    lines.append("A positive diversity result means that open-sea components changed over time and occupied a broader trait space. It does not, by itself, establish evolution. Evolutionary claims require a separate analysis of persistence, parent-child similarity, and null-model comparison.\n")

    report.write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Pipeline
# =============================================================================

def run_generation_and_open_sea(cfg: SimConfig, outdir: Path, logger: ProgressLogger) -> Path:
    rng = np.random.default_rng(cfg.seed)
    snapdir = ensure_dir(outdir / "open_sea_snapshots")

    logger.log("STEP 1/5: constructing confined environment")
    env_conf = make_confined_environment(cfg.confined_n, rng, cfg)
    f = init_fields(cfg.confined_n, rng, env_conf)

    logger.log(f"STEP 2/5: confined preformation steps={cfg.pre_steps}")
    for step in range(1, cfg.pre_steps + 1):
        update_fields(f, env_conf, rng, cfg, phase="confined_pre")
        if step % max(1, cfg.pre_steps // 10) == 0:
            summary = summarize_components_for_release(f.B, f.T, cfg)
            logger.log(
                "  confined preformation progress "
                f"{step}/{cfg.pre_steps}; components={summary['component_count']}; "
                f"largest_fraction={summary['largest_component_fraction']:.3f}; "
                f"membrane_voxels={summary['total_membrane_voxels']}"
            )

    logger.log(
        "STEP 3/5: confined generation until multi-component release trigger "
        f"or fallback at {cfg.gen_steps} steps"
    )
    logger.log(
        "  release trigger: "
        f"components >= {cfg.release_min_components}, "
        f"check_every={cfg.release_check_every}, "
        f"min_gen_step={cfg.release_min_gen_step}, "
        f"max_largest_fraction={cfg.release_max_largest_fraction} "
        "(<=0 means disabled)"
    )

    release_step = None
    release_summary = None
    for step in range(1, cfg.gen_steps + 1):
        update_fields(f, env_conf, rng, cfg, phase="confined_gen")

        should_check = (step % max(1, cfg.release_check_every) == 0) or (step == cfg.gen_steps)
        if should_check:
            summary = summarize_components_for_release(f.B, f.T, cfg)
            if step % max(1, cfg.gen_steps // 10) == 0 or release_trigger_met(summary, cfg) or step == cfg.gen_steps:
                logger.log(
                    "  confined generation "
                    f"{step}/{cfg.gen_steps}; components={summary['component_count']}; "
                    f"largest_fraction={summary['largest_component_fraction']:.3f}; "
                    f"largest_voxels={summary['largest_component_voxels']}; "
                    f"total_membrane_voxels={summary['total_membrane_voxels']}"
                )
            if step >= cfg.release_min_gen_step and release_trigger_met(summary, cfg):
                release_step = step
                release_summary = summary
                logger.log(
                    "  RELEASE TRIGGER MET: "
                    f"step={step}; components={summary['component_count']}; "
                    f"largest_fraction={summary['largest_component_fraction']:.3f}"
                )
                break

    if release_step is None:
        release_step = cfg.gen_steps
        release_summary = summarize_components_for_release(f.B, f.T, cfg)
        logger.log(
            "  RELEASE TRIGGER NOT MET; using fallback release at end of generation: "
            f"step={release_step}; components={release_summary['component_count']}; "
            f"largest_fraction={release_summary['largest_component_fraction']:.3f}"
        )

    with open(outdir / "release_trigger_summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_safe({"release_step": release_step, "release_summary": release_summary, "config": asdict(cfg)}), fp, indent=2, ensure_ascii=False)

    logger.log("STEP 4/5: embedding selected pre-saturation state into open sea")
    f_open = embed_center(f, cfg.open_n)
    env_open = make_open_environment(cfg.open_n, rng, cfg.source_regime)

    logger.log(f"STEP 5/5: open-sea release simulation steps={cfg.open_steps}; snapshots every {cfg.sample_every}")
    save_snapshot(snapdir, 0, f_open, cfg, recenter_shift=(0, 0, 0))
    saved = 1
    for step in range(1, cfg.open_steps + 1):
        update_fields(f_open, env_open, rng, cfg, phase="open")
        shift = recenter_fields_if_needed(f_open, cfg, logger, step)
        if step % cfg.sample_every == 0 or step == cfg.open_steps:
            save_snapshot(snapdir, step, f_open, cfg, recenter_shift=shift)
            saved += 1
        if step % max(1, cfg.open_steps // 10) == 0:
            summary = summarize_components_for_release(f_open.B, f_open.T, cfg)
            logger.log(
                f"  open-sea progress {step}/{cfg.open_steps}; "
                f"components={summary['component_count']}; "
                f"largest_fraction={summary['largest_component_fraction']:.3f}; "
                f"membrane_voxels={summary['total_membrane_voxels']}; snapshots_saved={saved}"
            )

    logger.log(f"saved open-sea snapshots to: {snapdir}")
    return snapdir


def analyze_snapshots(snapdir: Path, outdir: Path, cfg: SimConfig, logger: ProgressLogger) -> None:
    logger.log("ANALYSIS 1/6: loading snapshot list")
    snaps = sorted(snapdir.glob("open_sea_step_*.npz"))
    if not snaps:
        raise FileNotFoundError(f"no open_sea_step_*.npz snapshots found in {snapdir}")
    logger.log(f"  found {len(snaps)} snapshots")

    tracker = ComponentTracker(min_overlap_voxels=5)
    component_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []

    logger.log("ANALYSIS 2/6: detecting and tracking components")
    for i, sp in enumerate(snaps, start=1):
        rows, events, overlaps = extract_component_traits_from_snapshot(sp, cfg, tracker, logger)
        component_rows.extend(rows)
        event_rows.extend(events)
        overlap_rows.extend(overlaps)
        if i % max(1, len(snaps) // 10) == 0 or i == len(snaps):
            logger.log(f"  processed snapshots {i}/{len(snaps)}; component_observations={len(component_rows)}")

    logger.log("ANALYSIS 3/6: computing population diversity")
    diversity_rows = compute_population_diversity(component_rows)

    logger.log("ANALYSIS 4/6: computing component change summaries")
    change_rows = compute_change_summary(component_rows)

    logger.log("ANALYSIS 5/6: computing PCA trait space")
    pca_rows = compute_pca_rows(component_rows)

    logger.log("ANALYSIS 6/6: writing CSVs, figures, and report")
    # Remove private z-vectors before CSV.
    for r in component_rows:
        if "_zvec" in r:
            del r["_zvec"]

    write_csv(outdir / "component_traits_over_time.csv", component_rows)
    write_csv(outdir / "component_overlap_audit.csv", overlap_rows)
    write_csv(outdir / "component_events.csv", event_rows)
    write_csv(outdir / "population_diversity_timeseries.csv", diversity_rows)
    write_csv(outdir / "component_change_summary.csv", change_rows)
    write_csv(outdir / "pca_component_trait_space.csv", pca_rows)

    make_figures(outdir, diversity_rows, pca_rows)
    write_report(outdir, cfg, component_rows, diversity_rows, change_rows, event_rows)

    logger.log("analysis outputs written")
    logger.log(f"  component observations: {len(component_rows)}")
    logger.log(f"  tracks: {len(set(r['track_id'] for r in component_rows)) if component_rows else 0}")
    logger.log(f"  report: {outdir / 'change_and_diversity_report.md'}")


def mode_defaults(mode: str) -> Dict[str, Any]:
    if mode == "smoke":
        return dict(confined_n=28, open_n=44, pre_steps=80, gen_steps=160, open_steps=180, sample_every=20)
    if mode == "quick":
        return dict(confined_n=36, open_n=60, pre_steps=240, gen_steps=480, open_steps=600, sample_every=20)
    if mode == "full":
        return dict(confined_n=44, open_n=72, pre_steps=1200, gen_steps=2400, open_steps=1800, sample_every=20)
    raise ValueError(f"unknown mode: {mode}")



# =============================================================================
# Fixed 10-hole formation + open-sea environmental heterogeneity sweep
# =============================================================================

VERSION = "2026-06-05-fixed10holes-openenv-sweep"

OPEN_ENV_TYPES = [
    "homogeneous",
    "patchy_resource",
    "flow_diverse",
    "shear_diverse",
    "residence_diverse",
    "mixed_niche",
]


def rescale_to_mean(a: np.ndarray, target_mean: float, max_iter: int = 8) -> np.ndarray:
    """Rescale a nonnegative field to an approximate target mean after clipping to [0,1]."""
    a = np.maximum(a.astype(np.float32), 0.0)
    if float(np.max(a)) <= 1e-12:
        return np.full_like(a, float(target_mean), dtype=np.float32)
    a = normalize(a)
    scale = float(target_mean) / (float(np.mean(a)) + 1e-12)
    out = np.clip(a * scale, 0.0, 1.0).astype(np.float32)
    for _ in range(max_iter):
        m = float(np.mean(out))
        if abs(m - target_mean) < 0.002:
            break
        out = np.clip(out * (float(target_mean) / (m + 1e-12)), 0.0, 1.0).astype(np.float32)
    return out.astype(np.float32)


def deterministic_open_centers(n: int, count: int, rng: np.random.Generator, margin: float = 0.18) -> np.ndarray:
    """Separated centers for open-sea niches."""
    centers = []
    attempts = 0
    min_dist = 0.23 * n
    while len(centers) < count and attempts < 10000:
        attempts += 1
        c = rng.uniform(margin * n, (1.0 - margin) * n, size=3)
        if not centers or min(float(np.linalg.norm(c - q)) for q in centers) >= min_dist:
            centers.append(c)
    while len(centers) < count:
        centers.append(rng.uniform(margin * n, (1.0 - margin) * n, size=3))
    return np.array(centers, dtype=np.float64)


def gaussian_field_from_centers(n: int, centers: np.ndarray, radius_frac: float, amplitudes: Optional[np.ndarray] = None) -> np.ndarray:
    x, y, z = np.indices((n, n, n), dtype=np.float32)
    field = np.zeros((n, n, n), dtype=np.float32)
    if amplitudes is None:
        amplitudes = np.ones(len(centers), dtype=np.float64)
    sig = float(radius_frac) * n
    for amp, (cx, cy, cz) in zip(amplitudes, centers):
        d = ((x - cx) / sig) ** 2 + ((y - cy) / sig) ** 2 + ((z - cz) / sig) ** 2
        field += float(amp) * np.exp(-0.5 * d).astype(np.float32)
    return normalize(field).astype(np.float32)


def make_vortex_flow(n: int, rng: np.random.Generator, centers: np.ndarray, strength: float = 0.035) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = np.indices((n, n, n), dtype=np.float32)
    vx = np.zeros((n, n, n), dtype=np.float32)
    vy = np.zeros_like(vx)
    vz = np.zeros_like(vx)
    for c in centers:
        cx, cy, cz = c
        axis = rng.normal(0, 1, size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        rx = x - cx
        ry = y - cy
        rz = z - cz
        # cross(axis, r)
        wx = axis[1] * rz - axis[2] * ry
        wy = axis[2] * rx - axis[0] * rz
        wz = axis[0] * ry - axis[1] * rx
        r2 = rx * rx + ry * ry + rz * rz
        envelope = np.exp(-0.5 * r2 / ((0.18 * n) ** 2)).astype(np.float32)
        norm = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-6
        vx += (wx / norm * envelope).astype(np.float32)
        vy += (wy / norm * envelope).astype(np.float32)
        vz += (wz / norm * envelope).astype(np.float32)
    mag = np.sqrt(vx * vx + vy * vy + vz * vz)
    mean_mag = float(np.mean(mag)) + 1e-12
    vx = (vx / mean_mag * strength).astype(np.float32)
    vy = (vy / mean_mag * strength).astype(np.float32)
    vz = (vz / mean_mag * strength).astype(np.float32)
    return vx, vy, vz


def normalize_mean_flow(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, target_mean: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = np.sqrt(vx * vx + vy * vy + vz * vz)
    m = float(np.mean(mag)) + 1e-12
    scale = float(target_mean) / m
    return (vx * scale).astype(np.float32), (vy * scale).astype(np.float32), (vz * scale).astype(np.float32)


def make_open_environment_by_type(n: int, rng: np.random.Generator, env_type: str, source_regime: str = "continuous") -> Environment3D:
    """Open-sea environment sweep.

    Formation state is fixed across conditions. Only the release/open-sea field is
    changed here. Resource total, mean flow magnitude, domain size, and integration
    settings are kept approximately comparable.
    """
    if env_type not in OPEN_ENV_TYPES:
        raise ValueError(f"unknown open_env_type: {env_type}. Allowed: {OPEN_ENV_TYPES}")

    target_resource_mean = 0.18 if source_regime == "continuous" else (0.027 if source_regime == "low" else 0.0)
    target_flow_mean = 0.030

    zeros = np.zeros((n, n, n), dtype=np.float32)
    pore = zeros.copy()
    mineral = zeros.copy()
    vent = zeros.copy()

    # Base flow direction is shared in type but seeded per condition.
    base = rng.normal(0, 1, size=3)
    base = base / (np.linalg.norm(base) + 1e-12)
    vx_base = np.full((n, n, n), target_flow_mean * base[0], dtype=np.float32)
    vy_base = np.full((n, n, n), target_flow_mean * base[1], dtype=np.float32)
    vz_base = np.full((n, n, n), target_flow_mean * base[2], dtype=np.float32)

    pressure = np.full((n, n, n), 0.12, dtype=np.float32)
    residence = np.full((n, n, n), 0.04, dtype=np.float32)
    source_shape = np.full((n, n, n), target_resource_mean, dtype=np.float32)
    vx, vy, vz = vx_base.copy(), vy_base.copy(), vz_base.copy()
    shear = normalize(gradmag3(vx) + gradmag3(vy) + gradmag3(vz))

    centers_resource = deterministic_open_centers(n, 8, rng, margin=0.16)
    centers_flow = deterministic_open_centers(n, 5, rng, margin=0.18)
    centers_residence = deterministic_open_centers(n, 8, rng, margin=0.18)
    centers_shear = deterministic_open_centers(n, 5, rng, margin=0.14)

    if env_type == "homogeneous":
        # Minimal spatial structure. This is the baseline collapse condition.
        source_shape = np.full((n, n, n), target_resource_mean, dtype=np.float32)
        residence = np.full((n, n, n), 0.04, dtype=np.float32)
        pressure = np.full((n, n, n), 0.12, dtype=np.float32)
        vx, vy, vz = normalize_mean_flow(vx_base, vy_base, vz_base, target_flow_mean)
        shear = np.full((n, n, n), 0.08, dtype=np.float32)

    elif env_type == "patchy_resource":
        field = gaussian_field_from_centers(n, centers_resource, radius_frac=0.105, amplitudes=rng.uniform(0.8, 1.2, size=8))
        source_shape = rescale_to_mean(field, target_resource_mean)
        residence = np.full((n, n, n), 0.04, dtype=np.float32)
        pressure = np.full((n, n, n), 0.12, dtype=np.float32)
        vx, vy, vz = normalize_mean_flow(vx_base, vy_base, vz_base, target_flow_mean)
        shear = np.full((n, n, n), 0.08, dtype=np.float32)

    elif env_type == "flow_diverse":
        evx, evy, evz = make_vortex_flow(n, rng, centers_flow, strength=0.030)
        vx = vx_base + evx
        vy = vy_base + evy
        vz = vz_base + evz
        vx, vy, vz = normalize_mean_flow(vx, vy, vz, target_flow_mean)
        source_shape = np.full((n, n, n), target_resource_mean, dtype=np.float32)
        residence = np.full((n, n, n), 0.04, dtype=np.float32)
        pressure = normalize(0.20 + 0.20 * gaussian_field_from_centers(n, centers_flow, radius_frac=0.16))
        shear = normalize(gradmag3(vx) + gradmag3(vy) + gradmag3(vz))

    elif env_type == "shear_diverse":
        band = gaussian_field_from_centers(n, centers_shear, radius_frac=0.080, amplitudes=rng.uniform(0.9, 1.3, size=5))
        source_shape = np.full((n, n, n), target_resource_mean, dtype=np.float32)
        residence = np.full((n, n, n), 0.035, dtype=np.float32)
        vx = vx_base + (0.016 * rng.normal(size=(n, n, n)).astype(np.float32) * band)
        vy = vy_base + (0.016 * rng.normal(size=(n, n, n)).astype(np.float32) * band)
        vz = vz_base + (0.016 * rng.normal(size=(n, n, n)).astype(np.float32) * band)
        vx, vy, vz = normalize_mean_flow(vx, vy, vz, target_flow_mean)
        shear = rescale_to_mean(0.25 * normalize(gradmag3(vx) + gradmag3(vy) + gradmag3(vz)) + 0.75 * band, 0.18)
        pressure = clip01(0.10 + 0.35 * band)

    elif env_type == "residence_diverse":
        pockets = gaussian_field_from_centers(n, centers_residence, radius_frac=0.105, amplitudes=rng.uniform(0.8, 1.2, size=8))
        residence = clip01(0.025 + 0.68 * pockets)
        source_shape = np.full((n, n, n), target_resource_mean, dtype=np.float32)
        # Flow slows inside residence pockets, helping spatial separation persist.
        attenuation = np.clip(1.0 - 0.78 * residence, 0.18, 1.0).astype(np.float32)
        vx, vy, vz = normalize_mean_flow(vx_base * attenuation, vy_base * attenuation, vz_base * attenuation, target_flow_mean)
        shear = normalize(gradmag3(vx) + gradmag3(vy) + gradmag3(vz))
        pressure = clip01(0.10 + 0.18 * (1.0 - residence))

    elif env_type == "mixed_niche":
        res_patch = gaussian_field_from_centers(n, centers_resource, radius_frac=0.095, amplitudes=rng.uniform(0.8, 1.25, size=8))
        pockets = gaussian_field_from_centers(n, centers_residence, radius_frac=0.105, amplitudes=rng.uniform(0.8, 1.2, size=8))
        shear_patch = gaussian_field_from_centers(n, centers_shear, radius_frac=0.075, amplitudes=rng.uniform(0.8, 1.3, size=5))
        evx, evy, evz = make_vortex_flow(n, rng, centers_flow, strength=0.030)
        source_shape = rescale_to_mean(0.85 * res_patch + 0.15 * pockets, target_resource_mean)
        residence = clip01(0.020 + 0.55 * pockets + 0.10 * res_patch)
        attenuation = np.clip(1.0 - 0.65 * residence + 0.10 * shear_patch, 0.18, 1.10).astype(np.float32)
        vx = vx_base * attenuation + evx + 0.006 * shear_patch * rng.normal(size=(n, n, n)).astype(np.float32)
        vy = vy_base * attenuation + evy + 0.006 * shear_patch * rng.normal(size=(n, n, n)).astype(np.float32)
        vz = vz_base * attenuation + evz + 0.006 * shear_patch * rng.normal(size=(n, n, n)).astype(np.float32)
        vx, vy, vz = normalize_mean_flow(vx, vy, vz, target_flow_mean)
        shear = rescale_to_mean(0.40 * normalize(gradmag3(vx) + gradmag3(vy) + gradmag3(vz)) + 0.60 * shear_patch, 0.18)
        pressure = clip01(0.08 + 0.20 * shear_patch + 0.12 * (1.0 - residence))

    if source_regime == "none":
        source_shape = np.zeros((n, n, n), dtype=np.float32)
    elif source_regime == "low":
        source_shape = rescale_to_mean(source_shape, 0.027)

    return Environment3D(
        pore=pore.astype(np.float32),
        mineral=mineral.astype(np.float32),
        residence=clip01(residence).astype(np.float32),
        vent=vent.astype(np.float32),
        pressure=clip01(pressure).astype(np.float32),
        vx=vx.astype(np.float32),
        vy=vy.astype(np.float32),
        vz=vz.astype(np.float32),
        shear=clip01(shear).astype(np.float32),
        source_shape=clip01(source_shape).astype(np.float32),
    )


def copy_fields(f: Fields) -> Fields:
    return Fields(R=f.R.copy(), L=f.L.copy(), H=f.H.copy(), X=f.X.copy(), M=f.M.copy(), B=f.B.copy(), T=f.T.copy())


def save_release_state(path: Path, f: Fields, release_step: int, release_summary: Dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        release_step=np.array(release_step, dtype=np.int32),
        B=f.B.astype(np.float32),
        T=f.T.astype(np.float32),
        M=f.M.astype(np.float32),
        R=f.R.astype(np.float32),
        L=f.L.astype(np.float32),
        H=f.H.astype(np.float32),
        X=f.X.astype(np.float32),
        release_summary_json=np.array(json.dumps(json_safe(release_summary), ensure_ascii=False)),
    )


def generate_fixed_release_state(cfg: SimConfig, outdir: Path, logger: ProgressLogger) -> Tuple[Fields, int, Dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    logger.log("FORMATION 1/3: constructing fixed 10-hole confined environment")
    env_conf = make_confined_environment(cfg.confined_n, rng, cfg)
    f = init_fields(cfg.confined_n, rng, env_conf)

    logger.log(f"FORMATION 2/3: confined preformation steps={cfg.pre_steps}")
    for step in range(1, cfg.pre_steps + 1):
        update_fields(f, env_conf, rng, cfg, phase="confined_pre")
        if step % max(1, cfg.pre_steps // 10) == 0:
            summary = summarize_components_for_release(f.B, f.T, cfg)
            logger.log(
                f"  preformation {step}/{cfg.pre_steps}; components={summary['component_count']}; "
                f"largest_fraction={summary['largest_component_fraction']:.3f}; membrane_voxels={summary['total_membrane_voxels']}"
            )

    logger.log(
        "FORMATION 3/3: confined generation until fixed release trigger; "
        f"target components >= {cfg.release_min_components}"
    )
    release_step = None
    release_summary = None
    for step in range(1, cfg.gen_steps + 1):
        update_fields(f, env_conf, rng, cfg, phase="confined_gen")
        should_check = (step % max(1, cfg.release_check_every) == 0) or (step == cfg.gen_steps)
        if should_check:
            summary = summarize_components_for_release(f.B, f.T, cfg)
            if step % max(1, cfg.gen_steps // 10) == 0 or release_trigger_met(summary, cfg) or step == cfg.gen_steps:
                logger.log(
                    f"  generation {step}/{cfg.gen_steps}; components={summary['component_count']}; "
                    f"largest_fraction={summary['largest_component_fraction']:.3f}; membrane_voxels={summary['total_membrane_voxels']}"
                )
            if step >= cfg.release_min_gen_step and release_trigger_met(summary, cfg):
                release_step = step
                release_summary = summary
                logger.log(
                    f"  FIXED RELEASE STATE SELECTED: step={step}; components={summary['component_count']}; "
                    f"largest_fraction={summary['largest_component_fraction']:.3f}"
                )
                break

    if release_step is None:
        release_step = cfg.gen_steps
        release_summary = summarize_components_for_release(f.B, f.T, cfg)
        logger.log(
            "  RELEASE TRIGGER NOT MET; fallback=end; "
            f"step={release_step}; components={release_summary['component_count']}; "
            f"largest_fraction={release_summary['largest_component_fraction']:.3f}"
        )

    save_release_state(outdir / "fixed_release_state.npz", f, release_step, release_summary)
    with open(outdir / "fixed_release_summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_safe({"release_step": release_step, "release_summary": release_summary, "config": asdict(cfg)}), fp, indent=2, ensure_ascii=False)
    return f, int(release_step), release_summary


def run_open_sea_condition_from_release(
    release_fields: Fields,
    cfg: SimConfig,
    env_type: str,
    cond_dir: Path,
    release_step: int,
    release_summary: Dict[str, Any],
    condition_index: int,
    logger: ProgressLogger,
) -> Path:
    snapdir = ensure_dir(cond_dir / "open_sea_snapshots")
    rng = np.random.default_rng(cfg.seed + 10000 + 1009 * condition_index)
    f_open = embed_center(copy_fields(release_fields), cfg.open_n)
    env_open = make_open_environment_by_type(cfg.open_n, rng, env_type, cfg.source_regime)

    with open(cond_dir / "release_trigger_summary.json", "w", encoding="utf-8") as fp:
        json.dump(json_safe({
            "release_step": release_step,
            "release_summary": release_summary,
            "open_env_type": env_type,
            "config": asdict(cfg),
        }), fp, indent=2, ensure_ascii=False)

    logger.log(f"OPEN-SEA CONDITION: {env_type}; steps={cfg.open_steps}; snapshots every {cfg.sample_every}")
    save_snapshot(snapdir, 0, f_open, cfg, recenter_shift=(0, 0, 0))
    saved = 1
    for step in range(1, cfg.open_steps + 1):
        update_fields(f_open, env_open, rng, cfg, phase="open")
        shift = recenter_fields_if_needed(f_open, cfg, logger, step)
        if step % cfg.sample_every == 0 or step == cfg.open_steps:
            save_snapshot(snapdir, step, f_open, cfg, recenter_shift=shift)
            saved += 1
        if step % max(1, cfg.open_steps // 10) == 0:
            summary = summarize_components_for_release(f_open.B, f_open.T, cfg)
            logger.log(
                f"  open-sea {env_type} {step}/{cfg.open_steps}; components={summary['component_count']}; "
                f"largest_fraction={summary['largest_component_fraction']:.3f}; membrane_voxels={summary['total_membrane_voxels']}; snapshots_saved={saved}"
            )
    return snapdir


def parse_env_types(text: str) -> List[str]:
    if text is None or str(text).strip() == "" or str(text).strip().lower() == "all":
        return list(OPEN_ENV_TYPES)
    vals = []
    for part in str(text).split(','):
        p = part.strip()
        if p:
            if p not in OPEN_ENV_TYPES:
                raise ValueError(f"unknown env type: {p}. allowed={OPEN_ENV_TYPES}")
            vals.append(p)
    return vals or list(OPEN_ENV_TYPES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed 10-hole formation + open-sea environmental heterogeneity sweep.")
    p.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--env-types", type=str, default="all", help="Comma-separated open-sea env types or 'all'.")
    p.add_argument("--hole-count", type=int, default=10, help="Formation holes. Default fixed at 10.")
    p.add_argument("--confined-n", type=int, default=None)
    p.add_argument("--open-n", type=int, default=None)
    p.add_argument("--pre-steps", type=int, default=None)
    p.add_argument("--gen-steps", type=int, default=None)
    p.add_argument("--open-steps", type=int, default=None)
    p.add_argument("--sample-every", type=int, default=None)
    p.add_argument("--dt", type=float, default=0.035)
    p.add_argument("--b-threshold", type=float, default=0.16)
    p.add_argument("--t-threshold", type=float, default=0.018)
    p.add_argument("--min-component-voxels", type=int, default=12)
    p.add_argument("--min-lumen-voxels", type=int, default=20)
    p.add_argument("--source-regime", choices=["continuous", "low", "none"], default="continuous")
    p.add_argument("--release-min-components", type=int, default=8, help="Release fixed formation state as soon as at least this many components are detected.")
    p.add_argument("--release-check-every", type=int, default=20)
    p.add_argument("--release-min-gen-step", type=int, default=0)
    p.add_argument("--release-max-largest-fraction", type=float, default=0.0)
    p.add_argument("--release-fallback", choices=["end"], default="end")
    p.add_argument("--no-recenter", action="store_true")
    p.add_argument("--save-float16", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> SimConfig:
    d = mode_defaults(args.mode)
    confined_n = args.confined_n if args.confined_n is not None else d["confined_n"]
    open_n = args.open_n if args.open_n is not None else d["open_n"]
    pre_steps = args.pre_steps if args.pre_steps is not None else d["pre_steps"]
    gen_steps = args.gen_steps if args.gen_steps is not None else d["gen_steps"]
    open_steps = args.open_steps if args.open_steps is not None else d["open_steps"]
    sample_every = args.sample_every if args.sample_every is not None else d["sample_every"]
    return SimConfig(
        seed=int(args.seed),
        hole_count=int(args.hole_count),
        fixed_total_pore=False,  # hole size is fixed; no scaling by hole count
        reference_hole_count=int(args.hole_count),
        confined_n=int(confined_n),
        open_n=int(open_n),
        pre_steps=int(pre_steps),
        gen_steps=int(gen_steps),
        open_steps=int(open_steps),
        sample_every=int(sample_every),
        dt=float(args.dt),
        b_threshold=float(args.b_threshold),
        t_threshold=float(args.t_threshold),
        min_component_voxels=int(args.min_component_voxels),
        min_lumen_voxels=int(args.min_lumen_voxels),
        source_regime=str(args.source_regime),
        recenter=not bool(args.no_recenter),
        save_float16=bool(args.save_float16),
        release_min_components=int(args.release_min_components),
        release_check_every=int(args.release_check_every),
        release_min_gen_step=int(args.release_min_gen_step),
        release_max_largest_fraction=float(args.release_max_largest_fraction),
        release_fallback=str(args.release_fallback),
    )



def load_summary_metric_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def summarize_env_condition_outputs(cond_dir: Path, env_type: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"open_env_type": env_type, "condition_dir": str(cond_dir)}
    rel_path = cond_dir / "release_trigger_summary.json"
    if rel_path.exists():
        try:
            rel = json.loads(rel_path.read_text(encoding="utf-8"))
            rs = rel.get("release_summary", {})
            row.update({
                "release_step": rel.get("release_step", ""),
                "release_component_count": rs.get("component_count", ""),
                "release_largest_component_fraction": rs.get("largest_component_fraction", ""),
                "release_total_membrane_voxels": rs.get("total_membrane_voxels", ""),
            })
        except Exception as e:
            row["release_summary_error"] = str(e)
    div_rows = load_summary_metric_csv(cond_dir / "population_diversity_timeseries.csv")
    if div_rows:
        steps = [float(r.get("step", 0) or 0) for r in div_rows]
        counts = [float(r.get("component_count", 0) or 0) for r in div_rows]
        dists = [float(r.get("mean_pairwise_trait_distance", 0) or 0) for r in div_rows]
        if len(steps) >= 2:
            row["component_count_auc"] = float(np.trapezoid(counts, steps))
            row["trait_distance_auc"] = float(np.trapezoid(dists, steps))
        else:
            row["component_count_auc"] = 0.0
            row["trait_distance_auc"] = 0.0
        row["first_component_count"] = counts[0]
        row["final_component_count"] = counts[-1]
        row["max_component_count"] = max(counts)
        row["first_trait_distance"] = dists[0]
        row["final_trait_distance"] = dists[-1]
        row["max_trait_distance"] = max(dists)
        t_single = ""
        for st, c in zip(steps, counts):
            if st > steps[0] and c <= 1:
                t_single = st
                break
        row["time_to_single_component"] = t_single
    ch_rows = load_summary_metric_csv(cond_dir / "component_change_summary.csv")
    if ch_rows:
        persistent = [r for r in ch_rows if int(float(r.get("n_observations", 0) or 0)) >= 3]
        row["persistent_track_count"] = len(persistent)
        if persistent:
            row["mean_path_length_persistent"] = float(np.mean([float(r.get("path_length", 0) or 0) for r in persistent]))
            row["mean_volume_delta_persistent"] = float(np.mean([float(r.get("volume_delta", 0) or 0) for r in persistent]))
    ev_rows = load_summary_metric_csv(cond_dir / "component_events.csv")
    if ev_rows:
        row["fusion_event_count"] = sum(1 for r in ev_rows if str(r.get("event_type", "")) == "fusion_into_component")
        row["emergence_event_count"] = sum(1 for r in ev_rows if str(r.get("event_type", "")) == "emergence")
    return row


def write_openenv_sweep_report(outdir: Path, rows: List[Dict[str, Any]], cfg: SimConfig) -> None:
    lines = []
    lines.append("# Fixed 10-Hole Formation / Open-Sea Environment Sweep\n\n")
    lines.append(f"Version: `{VERSION}`\n\n")
    lines.append("Scope: change and diversity after open-sea release. This does not test evolution.\n\n")
    lines.append("Fixed formation condition: 10 holes, fixed hole size, fixed release state reused across all open-sea environments.\n\n")
    lines.append("Comparison variable: open-sea 3D environmental structure only.\n\n")
    lines.append("| open sea env | release comps | max comps | final comps | max trait distance | trait AUC | time to single | persistent tracks | fusion events |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| {r.get('open_env_type','')} | {r.get('release_component_count','')} | {r.get('max_component_count','')} | "
            f"{r.get('final_component_count','')} | {float(r.get('max_trait_distance',0) or 0):.6f} | "
            f"{float(r.get('trait_distance_auc',0) or 0):.6f} | {r.get('time_to_single_component','')} | "
            f"{r.get('persistent_track_count','')} | {r.get('fusion_event_count','')} |\n"
        )
    lines.append("\nInterpretation boundary: increased trait AUC or delayed time-to-single indicates extended diversity after release. Final component count >1 indicates sustained component multiplicity under that open-sea environment.\n")
    (outdir / "openenv_sweep_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.outdir.expanduser().resolve()
    ensure_dir(outdir)
    logger = ProgressLogger(outdir)
    try:
        env_types = parse_env_types(args.env_types)
        cfg = build_config(args)
        if cfg.hole_count != 10:
            logger.log(f"WARNING: hole_count={cfg.hole_count}; requested design is fixed 10 holes. Continuing with user value.")
        logger.log(f"START fixed-formation/open-env sweep :: version={VERSION}")
        logger.log(f"outdir: {outdir}")
        logger.log(f"mode: {args.mode}; fixed holes={cfg.hole_count}; env_types={env_types}")
        logger.log(f"scipy_available: {SCIPY_AVAILABLE}; matplotlib_available: {MATPLOTLIB_AVAILABLE}")
        with open(outdir / "analysis_parameters.json", "w", encoding="utf-8") as f:
            json.dump(json_safe({"version": VERSION, "args": vars(args), "config": asdict(cfg), "env_types": env_types}), f, indent=2, ensure_ascii=False)

        release_fields, release_step, release_summary = generate_fixed_release_state(cfg, outdir, logger)
        summary_rows: List[Dict[str, Any]] = []
        for i, env_type in enumerate(env_types):
            cond_dir = ensure_dir(outdir / env_type)
            cond_logger = ProgressLogger(cond_dir)
            try:
                cond_logger.log(f"START open-sea env condition: {env_type}")
                snapdir = run_open_sea_condition_from_release(
                    release_fields=release_fields,
                    cfg=cfg,
                    env_type=env_type,
                    cond_dir=cond_dir,
                    release_step=release_step,
                    release_summary=release_summary,
                    condition_index=i,
                    logger=cond_logger,
                )
                analyze_snapshots(snapdir, cond_dir, cfg, cond_logger)
                cond_logger.log(f"DONE condition: {env_type}")
            except Exception as e:
                cond_logger.log(f"ERROR condition {env_type}: {type(e).__name__}: {e}")
                (cond_dir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                cond_logger.close()
            row = summarize_env_condition_outputs(cond_dir, env_type)
            summary_rows.append(row)
            logger.log(
                f"CONDITION END {env_type}: max_components={row.get('max_component_count','NA')}; "
                f"final_components={row.get('final_component_count','NA')}; trait_auc={row.get('trait_distance_auc','NA')}"
            )

        write_csv(outdir / "openenv_sweep_summary.csv", summary_rows)
        write_openenv_sweep_report(outdir, summary_rows, cfg)
        logger.log(f"DONE; summary={outdir / 'openenv_sweep_summary.csv'}; report={outdir / 'openenv_sweep_report.md'}")
        return 0
    except Exception as e:
        logger.log(f"ERROR: {type(e).__name__}: {e}")
        (outdir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    finally:
        logger.close()



# =============================================================================
# 07A — Environmental-history-generated explicit chemical closure
# =============================================================================

import hashlib
import io
import zipfile
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass as _dc
from collections import defaultdict as _defaultdict

VERSION_07A = "07A-v2-explicit-chemical-closure-transport-corrected-standalone-1.1.0"
EXPECTED_07A_LOCK_SHA256 = "db0bee29192dd96f94cbd473a2b7ee002799a8378265a7354226298ced5922f9"
EXPECTED_07A_V2_CORRECTION_LOCK_SHA256 = "4a17f0629d038eda21c4842b50e2deadaa08999caf1ab9878448056cfdf5763d"
EMBEDDED_07A_METHOD_LOCK_B64 = "ewogICJzdHVkeV9pZCI6ICIwN0FfRU5WSVJPTk1FTlRBTF9ISVNUT1JZX0VYUExJQ0lUX0NIRU1JQ0FMX0NMT1NVUkUiLAogICJzdGF0dXMiOiAiRlJPWkVOX0JFRk9SRV9JTVBMRU1FTlRBVElPTiIsCiAgImNlbnRyYWxfcXVlc3Rpb24iOiAiQ2FuIGluaXRpYWxseSBpZGVudGljYWwgZmllbGQtZ2VuZXJhdGVkIHByb3RvY2VsbC1saWtlIGNvbXBvbmVudHMgYWNxdWlyZSBjb21wbGVtZW50YXJ5IGNhdGFseXRpYyBzdGF0ZXMgc29sZWx5IGZyb20gZGlmZmVyZW50IGVudmlyb25tZW50YWwgaGlzdG9yaWVzIGFuZCwgYWZ0ZXIgcGFzc2l2ZSByZWNvbnZlcmdlbmNlIGFuZCBuYXR1cmFsIHR3by1wYXJlbnQgZnVzaW9uLCBzaG93IGhpZ2hlciBzZXJpYWwgcmVhY3Rpb24gdGhyb3VnaHB1dCB0aGFuIGVxdWFsLXRvdGFsLWNhdGFseXN0IHNhbWUtaGlzdG9yeSBjb250cm9scyB3aXRob3V0IHByZWRlZmluZWQgcGFydG5lciB0eXBlcywgcGFydG5lciBzZW5zaW5nLCBwYWlyIGJvbnVzZXMsIGV2b2x1dGlvbiwgb3IgcHJvZ3JhbW1lZCByZXByb2R1Y3Rpb24/IiwKICAic291cmNlX3N1YnN0cmF0ZSI6IHsKICAgICJpbmhlcml0X3VuY2hhbmdlZCI6IFsKICAgICAgIjNEIGNvbnRpbnVvdXMtZmllbGQgcHJvdG9jZWxsIHN1YnN0cmF0ZSIsCiAgICAgICJSIHJlc291cmNlIGZpZWxkIiwKICAgICAgIkwgbGlwaWQgcHJlY3Vyc29yIGZpZWxkIiwKICAgICAgIkggYWN0aXZhdGluZy9oeWRyb3RoZXJtYWwgaW5wdXQgZmllbGQiLAogICAgICAiWCB3YXN0ZS9kZWdyYWRhdGlvbiBmaWVsZCIsCiAgICAgICJCIG1lbWJyYW5lLWRlbnNpdHkgZmllbGQiLAogICAgICAiVCBtZW1icmFuZS10aGlja25lc3MgZmllbGQiLAogICAgICAiTSBoaXN0b3J5LXNlbnNpdGl2ZSBpbnRlcm5hbCBmaWVsZCIsCiAgICAgICJQKEIsVCkgcGVybWVhYmlsaXR5IiwKICAgICAgInYgcGFzc2l2ZSBmbG93L2FkdmVjdGlvbiIsCiAgICAgICJmaWVsZC1nZW5lcmF0ZWQgY29tcG9uZW50IGRldGVjdGlvbiBmcm9tIEIgYW5kIFQiLAogICAgICAib3ZlcmxhcC1iYXNlZCBjb250aW51YXRpb24vZnVzaW9uIHRyYWNraW5nIgogICAgXSwKICAgICJleHBsaWNpdGx5X25vdF9yZWludGVycHJldGVkIjogWwogICAgICAiTSBpcyBub3QgdHJlYXRlZCBhcyBhIGJpb2NoZW1pY2FsIGNhdGFseXN0IG9yIGdlbmV0aWMgdmFyaWFibGUiLAogICAgICAiQiBhbmQgVCByZW1haW4gcGhlbm9tZW5vbG9naWNhbCBtZW1icmFuZSBmaWVsZHMiCiAgICBdLAogICAgIm5vdF9hZGRlZF9pbl8wN0EiOiBbCiAgICAgICJmaXNzaW9uIiwKICAgICAgInJlcHJvZHVjdGlvbiBydWxlIiwKICAgICAgImZpdG5lc3MiLAogICAgICAic2VsZWN0aW9uIiwKICAgICAgIm11dGF0aW9uIiwKICAgICAgImdlbmVzIiwKICAgICAgInNleCIsCiAgICAgICJtYXRpbmcgdHlwZSIsCiAgICAgICJwYXJ0bmVyIHNlbnNpbmciLAogICAgICAicS1kZXBlbmRlbnQgb3IgY2F0YWx5c3QtZGVwZW5kZW50IG1vdmVtZW50IiwKICAgICAgInBhaXIgYm9udXMiLAogICAgICAiZnVzaW9uLXRyaWdnZXJlZCBzcGVjaWFsIGNoZW1pc3RyeSIKICAgIF0KICB9LAogICJuZXdfZW52aXJvbm1lbnRhbF9jdWUiOiB7CiAgICAic3ltYm9sIjogIkUiLAogICAgIm1lYW5pbmciOiAiZ2VuZXJpYyBsb2NhbCBwaHlzaWNvY2hlbWljYWwgYmlhcyBheGlzIChlLmcuIHJlZG94L3BILWxpa2UpLCBub3QgYSBjZWxsIGlkZW50aXR5IiwKICAgICJjb25kaXRpb25lZF9zdGFnZSI6IHsKICAgICAgImxlZnRfaGFsZiI6IC0xLjAsCiAgICAgICJyaWdodF9oYWxmIjogMS4wLAogICAgICAidW5kaWZmZXJlbnRpYXRlZF9jb250cm9sIjogMC4wCiAgICB9LAogICAgInRyYW5zcG9ydF9zdGFnZSI6IDAuMCwKICAgICJkaXJlY3RfZWZmZWN0cyI6ICJFIGFmZmVjdHMgb25seSBDMTwtPkMyIGludGVyY29udmVyc2lvbjsgaXQgZG9lcyBub3QgZGlyZWN0bHkgYWx0ZXIgQixULE0sUixMLEgsWCxQIG9yIHYiCiAgfSwKICAiY2F0YWx5dGljX3N0YXRlcyI6IHsKICAgICJmaWVsZHMiOiBbCiAgICAgICJDMSIsCiAgICAgICJDMiIKICAgIF0sCiAgICAiaW5pdGlhbGl6YXRpb24iOiAiQWxsIGRldGVjdGVkIHJlbGVhc2UgY29tcG9uZW50cyBzdGFydCB3aXRoIGlkZW50aWNhbCBDMT1DMj1DX3RvdGFsLzIgcGVyIHVuaXQgbWVtYnJhbmUtYXNzb2NpYXRlZCBjYXRhbHl0aWMgbWFzcy4iLAogICAgImNvbnNlcnZhdGlvbiI6ICJDMSArIEMyID0gQ190b3RhbCBleGFjdGx5IGF0IGV2ZXJ5IHVwZGF0ZSwgYXBhcnQgZnJvbSBudW1lcmljYWwgcm91bmRvZmYgY29ycmVjdGVkIGJ5IHJlbm9ybWFsaXphdGlvbi4iLAogICAgImludGVycHJldGF0aW9uIjogIlR3byBpbnRlcmNvbnZlcnRpYmxlIGNhdGFseXRpYyBzdGF0ZXMgb2YgdGhlIHNhbWUgZmluaXRlIGNhdGFseXN0IHBvb2w7IG5vdCB0d28gY2VsbCB0eXBlcy4iLAogICAgImR5bmFtaWNzIjogewogICAgICAiZXF1YXRpb25zIjogWwogICAgICAgICJkQzEvZHQgPSBrMjEoRSkqQzIgLSBrMTIoRSkqQzEiLAogICAgICAgICJkQzIvZHQgPSAtZEMxL2R0IiwKICAgICAgICAiazIxKEUpID0gazAqZXhwKEUpIiwKICAgICAgICAiazEyKEUpID0gazAqZXhwKC1FKSIKICAgICAgXSwKICAgICAgInRhdV9DIjogMTAuMCwKICAgICAgImswIjogMC4wNSwKICAgICAgIm5vdGUiOiAiQXQgRT0wIHRoZSByZWxheGF0aW9uIHRpbWUgb2YgdGhlIEMxL0MyIGltYmFsYW5jZSBpcyB0YXVfQz0xLygyKmswKT0xMC4iCiAgICB9CiAgfSwKICAic3RhZ2VfdGltaW5nIjogewogICAgImNvbmRpdGlvbmluZ19kdXJhdGlvbiI6IDMwLjAsCiAgICAiY29uZGl0aW9uaW5nX2R1cmF0aW9uX2luX3RhdV9DIjogMy4wLAogICAgInRyYW5zcG9ydF9kdXJhdGlvbiI6IDQwLjAsCiAgICAidHJhbnNwb3J0X2R1cmF0aW9uX2luX3RhdV9DIjogNC4wLAogICAgInRhdV9jb252X292ZXJfdGF1X0Nfc2NyZWVuIjogWwogICAgICAwLjI1LAogICAgICAwLjUsCiAgICAgIDEuMCwKICAgICAgMi4wLAogICAgICA0LjAKICAgIF0sCiAgICAiYWxsX3JhdGlvc19yZXBvcnRlZCI6IHRydWUsCiAgICAibm9fcG9zdGhvY19yYXRpb19zZWxlY3Rpb24iOiB0cnVlCiAgfSwKICAidHJhbnNwb3J0IjogewogICAgInByaW5jaXBsZSI6ICJQYXNzaXZlIHN0YXRlLWJsaW5kIGNvbnZlcmdlbmNlIG9ubHkuIiwKICAgICJmbG93X2Zvcm0iOiAiVXNlIHRoZSBleGlzdGluZyB2LWZpZWxkIG1hY2hpbmVyeSB0byBpbXBsZW1lbnQgYSBjZW50cmFsbHkgY29udmVyZ2VudCBmbG93IHdob3NlIGNoYXJhY3RlcmlzdGljIHRpbWUgaXMgdGF1X2NvbnYuIiwKICAgICJub19jb252ZXJnZW5jZV9jb250cm9sIjogIlNhbWUgcGh5c2ljYWwgc3Vic3RyYXRlIGFuZCBkaWZmdXNpb24vbm9pc2UsIGJ1dCBubyBjb252ZXJnZW50IGNvbXBvbmVudCBvZiB2LiIsCiAgICAiZm9yYmlkZGVuIjogWwogICAgICAibW92ZW1lbnQgZGVwZW5kaW5nIG9uIEMxLCBDMiwgRS1oaXN0b3J5IGxhYmVsLCBNLCBwYXJ0bmVyIHN0YXRlLCBjb21wYXRpYmlsaXR5LCBvciBmdXR1cmUgZnVzaW9uIG91dGNvbWUiLAogICAgICAiYWN0aXZlIGhvbWluZyIsCiAgICAgICJhdHRyYWN0aW9uIGJldHdlZW4gY29tcGxlbWVudGFyeSBzdGF0ZXMiCiAgICBdCiAgfSwKICAibmF0dXJhbF9mdXNpb25fZXZlbnQiOiB7CiAgICAiZGV0ZWN0aW9uIjogIlVzZSBpbmhlcml0ZWQgb3ZlcmxhcC1iYXNlZCBjb21wb25lbnQgdHJhY2tpbmcuIiwKICAgICJwcmltYXJ5X2V2ZW50cyI6ICJFeGFjdCB0d28tcGFyZW50IGZ1c2lvbiBldmVudHMgb25seS4iLAogICAgInN0YXRlX3NhbXBsaW5nIjogIlBhcmVudCBjYXRhbHl0aWMgY29tcG9zaXRpb24gaXMgbWVhc3VyZWQgYXQgdGhlIGxhc3Qgc2FtcGxlZCBzdGF0ZSBiZWZvcmUgdGhlIGZ1c2lvbiBldmVudC4iLAogICAgImZ1dHVyZV9pbmZvcm1hdGlvbiI6ICJObyBmdXR1cmUgZmlzc2lvbiBvciBsYXRlciBwcm9kdWN0IG91dHB1dCBpcyB1c2VkIHRvIGRlZmluZSBlbGlnaWJsZSBmdXNpb24gZXZlbnRzLiIsCiAgICAicGFpcl9jb3VudGluZyI6ICJFYWNoIHVub3JkZXJlZCBwYXJlbnQgcGFpciBjb250cmlidXRlcyBhdCBtb3N0IG9uY2UgcGVyIHJ1bi4iCiAgfSwKICAiZXhwbGljaXRfc2VyaWFsX2NoZW1pc3RyeSI6IHsKICAgICJyZWFjdGlvbl9zY2hlbWUiOiBbCiAgICAgICJTIC0tKEMxKS0tPiBJIiwKICAgICAgIkkgLS0oQzIpLS0+IFkiCiAgICBdLAogICAgIlMiOiAiZ2VuZXJpYyBzdXBwbGllZCBwcmVjdXJzb3Ivc3Vic3RyYXRlIiwKICAgICJJIjogInJlYWN0aW9uIGludGVybWVkaWF0ZSIsCiAgICAiWSI6ICJtZW1icmFuZS1wcmVjdXJzb3ItbGlrZSBwcm9kdWN0OyByZWFkb3V0IG9ubHkgaW4gMDdBIGFuZCBkb2VzIG5vdCBmZWVkIGJhY2sgdG8gQiBvciBUIiwKICAgICJyYXRlX2NvbnN0YW50cyI6IHsKICAgICAgImsxIjogMS4wLAogICAgICAiazIiOiAxLjAKICAgIH0sCiAgICAidGhyb3VnaHB1dF9kZWZpbml0aW9uIjogIkZvciB0d28gc2VxdWVudGlhbCBwc2V1ZG8tZmlyc3Qtb3JkZXIgc3RlcHMsIGV4cGVjdGVkIGNvbXBsZXRpb24gdGltZSBpcyB0YXVfcnhuID0gMS8oazEqQzEpICsgMS8oazIqQzIpOyB0aHJvdWdocHV0IEogPSAxL3RhdV9yeG4uIFRoaXMgaXMgdGhlIHByaW1hcnkgY2hlbWljYWwtY2xvc3VyZSByZWFkb3V0LiIsCiAgICAibm9ybWFsaXphdGlvbiI6ICJBbGwgY29tcGFyaXNvbnMgdXNlIHRoZSBzYW1lIHRvdGFsIGNhdGFseXRpYyBtYXNzIENfdG90YWw7IG5vIGFkZGl0aXZlIGNhdGFseXN0IGJ1ZGdldCBpcyBjcmVhdGVkIGJ5IHBhaXJpbmcuIgogIH0sCiAgImVxdWFsX21hc3NfYXNzYXlzIjogewogICAgIkFCIjogIlBvb2wgZXF1YWwgY2F0YWx5dGljIG1hc3MgZnJvbSBwYXJlbnQgQSBhbmQgcGFyZW50IEIsIHRoZW4gY29tcHV0ZSBKX0FCLiIsCiAgICAiQUEiOiAiVXNlIHRoZSBzYW1lIHRvdGFsIGNhdGFseXRpYyBtYXNzIGFzIEFCIGJ1dCBib3RoIGhhbHZlcyBoYXZlIEEgY29tcG9zaXRpb24sIHlpZWxkaW5nIEpfQUEuIiwKICAgICJCQiI6ICJVc2UgdGhlIHNhbWUgdG90YWwgY2F0YWx5dGljIG1hc3MgYXMgQUIgYnV0IGJvdGggaGFsdmVzIGhhdmUgQiBjb21wb3NpdGlvbiwgeWllbGRpbmcgSl9CQi4iLAogICAgInByaW1hcnlfY29udGludW91c19nYWluIjogIkdfY2hlbSA9IEpfQUIgLSBtYXgoSl9BQSwgSl9CQikiLAogICAgInNlY29uZGFyeV9mdWxsX3RocmVzaG9sZF9zdXBwb3J0IjogIkFjcm9zcyBhIGZpeGVkIDUwMS1wb2ludCB0aGV0YSBncmlkIHNwYW5uaW5nIFswLCBKX21heF0sIHJlcG9ydCB0aGUgZnJhY3Rpb24gb2YgdGhldGEgZm9yIHdoaWNoIEpfQUE8dGhldGEsIEpfQkI8dGhldGEsIGFuZCBKX0FCPj10aGV0YS4gTm8gc2luZ2xlIHRoZXRhIGlzIHNlbGVjdGVkLiIKICB9LAogICJmYWN0b3JpYWxfY29uZGl0aW9ucyI6IFsKICAgICJESUZGRVJFTlRJQVRFRCArIENPTlZFUkdFTkNFIGF0IGV2ZXJ5IHRhdV9jb252L3RhdV9DIHJhdGlvIiwKICAgICJESUZGRVJFTlRJQVRFRCArIE5PX0NPTlZFUkdFTkNFIiwKICAgICJVTkRJRkZFUkVOVElBVEVEICsgQ09OVkVSR0VOQ0UgYXQgZXZlcnkgdGF1X2NvbnYvdGF1X0MgcmF0aW8iLAogICAgIlVORElGRkVSRU5USUFURUQgKyBOT19DT05WRVJHRU5DRSIKICBdLAogICJtZWNoYW5pc3RpY19jb250cm9scyI6IHsKICAgICJOT19QT09MSU5HIjogIkFCIGNoZW1pc3RyeSBpcyBldmFsdWF0ZWQgd2l0aG91dCBtaXhpbmcgY2F0YWx5dGljIGNvbXBvc2l0aW9uczsgcGFpciBnYWluIG11c3QgYmUgemVyby4iLAogICAgIk5PX1NFUklBTF9SRVFVSVJFTUVOVCI6ICJSZXBsYWNlIHRoZSB0d28tc3RlcCByZXF1aXJlbWVudCBieSBhIHJlYWRvdXQgcHJvcG9ydGlvbmFsIG9ubHkgdG8gdG90YWwgY2F0YWx5c3Q7IHBhaXIgZ2FpbiBtdXN0IGJlIHplcm8uIiwKICAgICJTQU1FX0hJU1RPUlkiOiAiQUEgYW5kIEJCIGVxdWFsLXRvdGFsLWNhdGFseXN0IGNvbnRyb2xzLiIsCiAgICAiTk9fRElGRkVSRU5USUFUSU9OIjogIkU9MCB0aHJvdWdob3V0IGNvbmRpdGlvbmluZy4iLAogICAgIk5PX0NPTlZFUkdFTkNFIjogIk5vIGNvbnZlcmdlbnQgZmxvdyBjb21wb25lbnQuIiwKICAgICJPUklHSU5fQVVESVQiOiAiUmVwb3J0IHNhbWUtb3JpZ2luIHZlcnN1cyBvcHBvc2l0ZS1vcmlnaW4gZXhhY3QgdHdvLXBhcmVudCBmdXNpb25zOyBvcmlnaW4gbGFiZWwgaXMgbmV2ZXIgcmVhZCBieSBkeW5hbWljcy4iCiAgfSwKICAicHJpbWFyeV9zdGF0aXN0aWNzIjogewogICAgInVuaXQiOiAic2VlZCIsCiAgICAic2VlZF9jb3VudF9mdWxsIjogMzIsCiAgICAibWF0Y2hlZF9hY3Jvc3NfY29uZGl0aW9ucyI6IHRydWUsCiAgICAicHJpbWFyeV9zZWVkX21ldHJpYyI6ICJtZWFuIEdfY2hlbSBhY3Jvc3MgZWxpZ2libGUgZXhhY3QgdHdvLXBhcmVudCBmdXNpb24gZXZlbnRzIiwKICAgICJpbnRlcmFjdGlvbiI6ICIoRElGRkVSRU5USUFURUQrQ09OVkVSR0VOQ0UgLSBESUZGRVJFTlRJQVRFRCtOT19DT05WRVJHRU5DRSkgLSAoVU5ESUZGRVJFTlRJQVRFRCtDT05WRVJHRU5DRSAtIFVORElGRkVSRU5USUFURUQrTk9fQ09OVkVSR0VOQ0UpIiwKICAgICJ0ZXN0IjogInR3by1zaWRlZCBwYWlyZWQgV2lsY294b24gc2lnbmVkLXJhbmsgcGVyIGZpeGVkIHRpbWVzY2FsZSByYXRpbyIsCiAgICAibXVsdGlwbGljaXR5IjogIkhvbG0gY29ycmVjdGlvbiBhY3Jvc3MgdGhlIGZpdmUgcHJlLXNwZWNpZmllZCB0YXVfY29udi90YXVfQyByYXRpb3MiLAogICAgImVmZmVjdF9yZXBvcnRpbmciOiBbCiAgICAgICJtZWFuIiwKICAgICAgIlNEIiwKICAgICAgIm1lZGlhbiIsCiAgICAgICJib290c3RyYXAgOTUlIENJIiwKICAgICAgInBvc2l0aXZlL3plcm8vbmVnYXRpdmUgc2VlZCBjb3VudHMiCiAgICBdLAogICAgIm1pc3NpbmdfZXZlbnRfcnVsZSI6ICJJZiBhIHNlZWQtY29uZGl0aW9uIGhhcyB6ZXJvIGVsaWdpYmxlIGV4YWN0IHR3by1wYXJlbnQgZnVzaW9uIGV2ZW50cywgaXQgaXMgcmVwb3J0ZWQgYXMgbm8tZXZlbnQgYW5kIGlzIG5vdCBhc3NpZ25lZCBhbiBhcnRpZmljaWFsIHplcm8gR19jaGVtLiBFdmVudCBhdmFpbGFiaWxpdHkgaXMgcmVwb3J0ZWQgc2VwYXJhdGVseS4iCiAgfSwKICAicm9idXN0bmVzc19hZnRlcl9wcmltYXJ5X3NhbWVfbG9jayI6IHsKICAgICJwcmluY2lwbGUiOiAiT25seSBpZiBjb21wdXRhdGlvbmFsbHkgZmVhc2libGUgdW5kZXIgdGhlIHNhbWUgZnJvemVuIGVxdWF0aW9uczsgbm8gcGFyYW1ldGVyIHR1bmluZyBmcm9tIHByaW1hcnkgcmVzdWx0cy4iLAogICAgInNjcmVlbnMiOiBbCiAgICAgICJjb21wb25lbnQtZGV0ZWN0aW9uIHRocmVzaG9sZHMgaW5oZXJpdGVkIGZyb20gdGhlIHNvdXJjZSBtb2RlbDogbG9vc2UvbWFpbi9zdHJpY3QiLAogICAgICAiQy1zdGF0ZSByZWxheGF0aW9uIHNjYWxpbmc6IHRhdV9DIG11bHRpcGxpZWQgYnkgMC41LCAxLCAyIHdoaWxlIHByZXNlcnZpbmcgZGltZW5zaW9ubGVzcyBzdGFnZSByYXRpb3MiLAogICAgICAiY29udmVyZ2VudC1mbG93IHN0cmVuZ3RoIHJlcHJlc2VudGVkIG9ubHkgdGhyb3VnaCB0aGUgYWxyZWFkeSBmaXhlZCB0YXVfY29udi90YXVfQyBzY3JlZW4iCiAgICBdLAogICAgImZvcmJpZGRlbiI6IFsKICAgICAgImNoYW5naW5nIGNhdGFseXN0IGVxdWF0aW9ucyBhZnRlciBzZWVpbmcgcmVzdWx0cyIsCiAgICAgICJjaGFuZ2luZyBFIGFtcGxpdHVkZXMgYWZ0ZXIgc2VlaW5nIHJlc3VsdHMiLAogICAgICAic2VsZWN0aW5nIG9ubHkgZmF2b3JhYmxlIGZ1c2lvbiBldmVudHMiLAogICAgICAiYWRkaW5nIGEgc3VjY2VzcyB0aHJlc2hvbGQiLAogICAgICAiY2hhbmdpbmcgc2VyaWFsIGNoZW1pc3RyeSB0byByZXNjdWUgYSBudWxsIHJlc3VsdCIKICAgIF0KICB9LAogICJpbnRlcnByZXRhdGlvbl9ib3VuZGFyeSI6ICIwN0EgY2FuIGVzdGFibGlzaCBlbnZpcm9ubWVudGFsLWhpc3RvcnktZ2VuZXJhdGVkIGV4cGxpY2l0IGNoZW1pY2FsIGNvbXBsZW1lbnRhcml0eSBhbmQgbmF0dXJhbCBmdXNpb24tYXNzb2NpYXRlZCBwYWlyLWxldmVsIHJlYWN0aW9uIGNsb3N1cmUuIEl0IGRvZXMgbm90IGVzdGFibGlzaCBtZW1icmFuZSBncm93dGgsIHBoeXNpY2FsIGZpc3Npb24sIGJpb2xvZ2ljYWwgcmVwcm9kdWN0aW9uLCBldm9sdXRpb24sIHNleCwgb3IgZ2VuZXRpYyBiaXBhcmVudGFsaXR5LiIsCiAgIm5leHRfc3RhZ2Vfb25seV9pZl8wN0FfaXNfZml4ZWQiOiAiMDdCIG1heSBjb3VwbGUgWSBwcm9kdWN0aW9uIHRvIGEgcHJlLWV4aXN0aW5nIGZpc3Npb24tY2FwYWJsZSBwaHlzaWNhbCBzdWJzdHJhdGUgYXMgYSBnZW5lcmljIG1hdGVyaWFsLWdyb3d0aCBmbHV4LCB3aXRob3V0IGFueSBwYWlyLXRyaWdnZXJlZCBvciB0aHJlc2hvbGQtdHJpZ2dlcmVkIGRpdmlzaW9uIHJ1bGUuIgp9Cg=="
EMBEDDED_07A_V2_CORRECTION_LOCK_B64 = "ewogICJzdHVkeV9pZCI6ICIwN0FfVjJfVFJBTlNQT1JUX0NPUlJFQ1RJT04iLAogICJzdGF0dXMiOiAiRlJPWkVOX0FGVEVSX05VTUVSSUNBTF9BVURJVF9CRUZPUkVfUkVSVU4iLAogICJwYXJlbnRfbWV0aG9kX2xvY2tfc2hhMjU2IjogImRiMGJlZTI5MTkyZGQ5NmY5NGNiZDQ3M2EyYjdlZTAwMjc5OWE4Mzc4MjY1YTczNTQyMjYyOThjZWQ1OTIyZjkiLAogICJyZWFzb25fZm9yX2NvcnJlY3Rpb24iOiAiVGhlIHYxIGNlbnRyYWxseSByYWRpYWwgc2luayBmaWVsZCB3YXMgY29tcHJlc3NpYmxlIGFuZCB3YXMgY29tYmluZWQgd2l0aCBub24tY29uc2VydmF0aXZlIHNjYWxhciBhZHZlY3Rpb24uIEF1ZGl0IHNob3dlZCB0aGF0IHRoaXMgaW1wbGVtZW50YXRpb24gcmVtb3ZlZCBCL1QgbWF0ZXJpYWwgYXQgYSByYXRlIG1hdGNoaW5nIHRoZSB2ZWxvY2l0eSBkaXZlcmdlbmNlIGFuZCB2aW9sYXRlZCBDRkwgbGltaXRzIGF0IHRoZSBmYXN0ZXN0IHByZS1zcGVjaWZpZWQgcmF0aW9zLiBUaG9zZSB2MSB0cmFuc3BvcnQgb3V0Y29tZXMgYXJlIGludmFsaWQgZm9yIGJpb2xvZ2ljYWwgaW50ZXJwcmV0YXRpb24uIiwKICAic2NpZW50aWZpY19lbGVtZW50c19oZWxkX2ZpeGVkIjogWwogICAgIkMxL0MyIGNhdGFseXRpYyBpbnRlcmNvbnZlcnNpb24gZXF1YXRpb25zIiwKICAgICJDMStDMiBmaW5pdGUgY2F0YWx5c3QgYnVkZ2V0IiwKICAgICJFPS0xLysxIGRpZmZlcmVudGlhdGVkIGNvbmRpdGlvbmluZyBhbmQgRT0wIGNvbnRyb2xzIiwKICAgICJjb25kaXRpb25pbmcgZHVyYXRpb24gLyB0YXVfQyIsCiAgICAidHJhbnNwb3J0IGR1cmF0aW9uIC8gdGF1X0MiLAogICAgInRhdV9jb252L3RhdV9DIHJhdGlvcyAwLjI1LCAwLjUsIDEsIDIsIDQiLAogICAgIlMgLT4gSSAtPiBZIHNlcmlhbCBjaGVtaXN0cnkiLAogICAgImVxdWFsLXRvdGFsLWNhdGFseXN0IEFCL0FBL0JCIGFzc2F5IiwKICAgICJHX2NoZW0gZW5kcG9pbnQiLAogICAgIjUwMS1wb2ludCB0aGV0YS1zdXBwb3J0IHNlY29uZGFyeSBlbmRwb2ludCIsCiAgICAiZXhhY3QgdHdvLXBhcmVudCBmdXNpb24gZWxpZ2liaWxpdHkiLAogICAgIjMyIG1hdGNoZWQgZnVsbCBzZWVkcyIsCiAgICAicGFpcmVkIFdpbGNveG9uIHRlc3RzIGFuZCBIb2xtIGNvcnJlY3Rpb24iLAogICAgIk5PX1BPT0xJTkcsIE5PX1NFUklBTCwgTk9fRElGRkVSRU5USUFUSU9OLCBOT19DT05WRVJHRU5DRSBjb250cm9scyIKICBdLAogICJ0cmFuc3BvcnRfY29ycmVjdGlvbnNfb25seSI6IHsKICAgICJmbG93IjogIlJlcGxhY2UgdGhlIGNvbXByZXNzaWJsZSByYWRpYWwgc2luayBieSBhIGJvdW5kZWQgYW5hbHl0aWNhbGx5IGRpdmVyZ2VuY2UtZnJlZSByZWNpcmN1bGF0aW5nIHN0cmFpbiBmaWVsZCBmb3JtZWQgZnJvbSB4LXkgYW5kIHgteiBzdHJlYW0tZnVuY3Rpb24gcm9sbHMuIFRoZSBsb2NhbCBjZW50ZXJsaW5lIG1lbWJyYW5lLWFkdmVjdGlvbiB4LXN0cmFpbiBpcyBmaXhlZCB0byAxL3RhdV9jb252IHNvIHRoZSBvcmlnaW5hbCBkaW1lbnNpb25sZXNzIHJhdGlvIHJldGFpbnMgaXRzIG1lYW5pbmcuIiwKICAgICJib3VuZGFyaWVzIjogIk5vcm1hbCB2ZWxvY2l0eSBpcyBleGFjdGx5IHplcm8gb24gYWxsIHNpeCBkb21haW4gZmFjZXM7IHRoZSBpbmhlcml0ZWQgb3Blbi1zZWEgc3BvbmdlIHJlbWFpbnMgdW5jaGFuZ2VkLiIsCiAgICAiYWR2ZWN0aW9uIjogIlVzZSBmaXJzdC1vcmRlciBtYXNzLWNvbnNlcnZhdGl2ZSB1cHdpbmQgZmx1eCBkaXZlcmdlbmNlIGZvciB0aGUgMDdBIHRyYW5zcG9ydCBzdGFnZSBvbmx5LiBGb3JtYXRpb24vZ2VuZXJhdGlvbiBwaHlzaWNzIHJlbWFpbnMgdW5jaGFuZ2VkLiIsCiAgICAiY2ZsIjogIk5vIHJhdGlvLWRlcGVuZGVudCByZXR1bmluZyBpcyBhbGxvd2VkLiBCZWZvcmUgYSBzY2llbnRpZmljIHJ1biwgdGhlIG1heGltdW0gc2NhbGFyLWZpZWxkIENGTCBhdCB0aGUgZmFzdGVzdCByYXRpbyBtdXN0IGJlIDwwLjg7IG90aGVyd2lzZSB0aGUgcnVuIGZhaWxzIHJhdGhlciB0aGFuIHJlc2NhbGluZyB0aGUgZmxvdy4iLAogICAgImF1ZGl0cyI6IFsKICAgICAgIm1heGltdW0gc2NhbGFyIGFuZCBtZW1icmFuZSBDRkwiLAogICAgICAibWF4aW11bSBhbmQgbWVhbiBkaXNjcmV0ZSBjZW50ZXJlZCBkaXZlcmdlbmNlIiwKICAgICAgIm1heGltdW0gYm91bmRhcnktbm9ybWFsIHZlbG9jaXR5IiwKICAgICAgInJlbGF0aXZlIGRvbWFpbi1zdW0gZHJpZnQgb2YgYSBkZXRlcm1pbmlzdGljIHBvc2l0aXZlIHByb2JlIHVuZGVyIHRoZSBjb25zZXJ2YXRpdmUgYWR2ZWN0aW9uIG9wZXJhdG9yIgogICAgXQogIH0sCiAgImZvcmJpZGRlbl9hZnRlcl9yZXJ1biI6IFsKICAgICJjaGFuZ2luZyBmbG93IHRvcG9sb2d5IHRvIGltcHJvdmUgYSByZXN1bHQiLAogICAgImNoYW5naW5nIGZsb3cgYW1wbGl0dWRlIG91dHNpZGUgdGhlIGxvY2tlZCB0YXVfY29udiBtYXBwaW5nIiwKICAgICJjaGFuZ2luZyBjYXRhbHlzdCBvciByZWFjdGlvbiBlcXVhdGlvbnMiLAogICAgInNlbGVjdGluZyBmYXZvcmFibGUgdGltZXNjYWxlIHJhdGlvcyIsCiAgICAiYWRkaW5nIHBhaXIgYXR0cmFjdGlvbiBvciBwYXJ0bmVyIHNlbnNpbmciLAogICAgImFkZGluZyBhIHN1Y2Nlc3MgdGhyZXNob2xkIG9yIGZpc3Npb24gcnVsZSIKICBdLAogICJpbnRlcnByZXRhdGlvbiI6ICJUaGUgY29ycmVjdGVkIHJlcnVuIHJlcGxhY2VzIHRoZSBpbnZhbGlkIHYxIHRyYW5zcG9ydCByZXN1bHRzLiBJdCBpcyBub3QgYW4gYWRkaXRpb25hbCByZXBsaWNhdGUgdG8gYmUgcG9vbGVkIHdpdGggdjEuIFRoZSB2MSBjaGVtaWNhbCBldmVudC1sZXZlbCBsb2dpYyBtYXkgYmUgdXNlZCBvbmx5IGFzIGEgY29kZS1wYXRoIGF1ZGl0LCBub3QgYXMgY29uZmlybWF0b3J5IGV2aWRlbmNlIGZvciB0aGUgdHJhbnNwb3J0LXRpbWVzY2FsZSBpbnRlcmFjdGlvbi4iCn0K"
SOURCE_CORE_NAME = "02_FIXED10_HISTORY_CORE.py"
SOURCE_CORE_SHA256 = "0062f30235be698241839c151597bc3411335831cbded3c666f73f7a5a876712"
TAU_C_BASE = 10.0
K0_BASE = 0.05
E_AMPLITUDE = 1.0
CONDITIONING_DURATION_FULL = 30.0
TRANSPORT_DURATION_FULL = 40.0
RATIOS_07A = (0.25, 0.5, 1.0, 2.0, 4.0)
THETA_07A = np.linspace(0.0, 0.25, 501, dtype=np.float64)
MEMBRANE_ADV_COEFF = 0.010  # inherited B/T advection multiplier in update_fields
MIN_OVERLAP_07A = 5
EPS_07A = 1e-12


@_dc
class CatState07A:
    track_id: int
    c1_diff: float
    c1_undiff: float
    origin_balance: float
    mass: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_embedded_lock(payload_b64: str, expected_sha: str, label: str) -> str:
    raw = base64.b64decode(payload_b64.encode("ascii"))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(
            f"{label} embedded SHA mismatch: got={digest} expected={expected_sha}"
        )
    # Also ensure the embedded payload is valid JSON.
    json.loads(raw.decode("utf-8"))
    return digest


def locate_method_lock_07a(explicit: Optional[Path]) -> Tuple[Path, str]:
    """
    Standalone behavior:
    - If --method-lock is explicitly supplied, validate that file.
    - Otherwise use the immutable lock embedded in this script.
    """
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Explicit --method-lock file not found: {p}")
        digest = _sha256(p)
        if digest != EXPECTED_07A_LOCK_SHA256:
            raise RuntimeError(
                f"07A method-lock SHA mismatch: got={digest} expected={EXPECTED_07A_LOCK_SHA256} path={p}"
            )
        return p, digest

    digest = _validate_embedded_lock(
        EMBEDDED_07A_METHOD_LOCK_B64,
        EXPECTED_07A_LOCK_SHA256,
        "07A method lock",
    )
    return Path("<embedded:07A_METHOD_LOCK.json>"), digest


def locate_correction_lock_07a(explicit: Optional[Path]) -> Tuple[Path, str]:
    """
    Standalone behavior:
    - If --correction-lock is explicitly supplied, validate that file.
    - Otherwise use the immutable correction lock embedded in this script.
    """
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Explicit --correction-lock file not found: {p}")
        digest = _sha256(p)
        if digest != EXPECTED_07A_V2_CORRECTION_LOCK_SHA256:
            raise RuntimeError(
                f"07A-v2 correction-lock SHA mismatch: got={digest} expected={EXPECTED_07A_V2_CORRECTION_LOCK_SHA256} path={p}"
            )
        return p, digest

    digest = _validate_embedded_lock(
        EMBEDDED_07A_V2_CORRECTION_LOCK_B64,
        EXPECTED_07A_V2_CORRECTION_LOCK_SHA256,
        "07A-v2 correction lock",
    )
    return Path("<embedded:07A_V2_TRANSPORT_CORRECTION_LOCK.json>"), digest

def c1_after_conditioning(E: float, duration: float, tau_c: float = TAU_C_BASE) -> float:
    k0 = 1.0 / (2.0 * float(tau_c))
    k21 = k0 * math.exp(float(E))
    k12 = k0 * math.exp(-float(E))
    rate = k21 + k12
    eq = k21 / rate
    return float(eq + (0.5 - eq) * math.exp(-rate * float(duration)))


def relax_c1(c1: float, dt_time: float, tau_c: float = TAU_C_BASE) -> float:
    return float(0.5 + (float(c1) - 0.5) * math.exp(-float(dt_time) / float(tau_c)))


def throughput_07a(c1: float, k1: float = 1.0, k2: float = 1.0) -> float:
    c1 = float(np.clip(c1, 0.0, 1.0))
    c2 = 1.0 - c1
    a = k1 * c1
    b = k2 * c2
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return float((a * b) / (a + b))


def pair_assay_07a(c1_a: float, c1_b: float) -> Dict[str, float]:
    j_aa = throughput_07a(c1_a)
    j_bb = throughput_07a(c1_b)
    c1_ab = 0.5 * (float(c1_a) + float(c1_b))
    j_ab = throughput_07a(c1_ab)
    g = float(j_ab - max(j_aa, j_bb))
    support = float(np.mean((THETA_07A > max(j_aa, j_bb)) & (THETA_07A <= j_ab)))
    return {
        "c1_ab": c1_ab,
        "J_AA": j_aa,
        "J_BB": j_bb,
        "J_AB": j_ab,
        "G_chem": g,
        "theta_support_fraction": support,
        "NO_POOLING_G": 0.0,
        "NO_SERIAL_G": 0.0,
    }


def detect_components_07a(B: np.ndarray, T: np.ndarray, cfg: SimConfig,
                           b_threshold: Optional[float] = None,
                           t_threshold: Optional[float] = None) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Tuple[float, float, float]]]:
    bt = cfg.b_threshold if b_threshold is None else float(b_threshold)
    tt = cfg.t_threshold if t_threshold is None else float(t_threshold)
    mask = (B >= bt) & (T >= tt)
    labels, _ = detect_labels(mask)
    labels, sizes = filter_small_labels(labels, cfg.min_component_voxels)
    centroids: Dict[int, Tuple[float, float, float]] = {}
    for lab, size in sizes.items():
        pts = np.argwhere(labels == lab)
        if pts.size == 0:
            continue
        w = (B[labels == lab] + T[labels == lab] + 1e-6).astype(np.float64)
        c = np.average(pts.astype(np.float64), axis=0, weights=w)
        centroids[int(lab)] = (float(c[0]), float(c[1]), float(c[2]))
    return labels.astype(np.int32, copy=False), sizes, centroids


def overlaps_07a(prev: np.ndarray, curr: np.ndarray, prev_map: Dict[int, int]) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
    by_curr: Dict[int, List[Tuple[int, int]]] = {}
    by_prev: Dict[int, List[Tuple[int, int]]] = {}
    if prev is None or curr is None or prev.size == 0 or curr.size == 0:
        return by_curr, by_prev
    n_curr = int(curr.max()) + 1
    both = (prev > 0) & (curr > 0)
    if not np.any(both):
        return by_curr, by_prev
    code = prev[both].astype(np.int64) * n_curr + curr[both].astype(np.int64)
    uniq, counts = np.unique(code, return_counts=True)
    for u, c in zip(uniq, counts):
        pl = int(u // n_curr)
        cl = int(u % n_curr)
        if int(c) < MIN_OVERLAP_07A or pl not in prev_map:
            continue
        by_curr.setdefault(cl, []).append((pl, int(c)))
        by_prev.setdefault(pl, []).append((cl, int(c)))
    for d in (by_curr, by_prev):
        for k in d:
            d[k].sort(key=lambda x: x[1], reverse=True)
    return by_curr, by_prev


def classify_origin_07a(a: float, b: float) -> str:
    def pure(x: float) -> int:
        if x <= -0.999:
            return -1
        if x >= 0.999:
            return 1
        if abs(x) <= 1e-12:
            return 0
        return 2
    pa, pb = pure(a), pure(b)
    if pa in (-1, 1) and pb in (-1, 1):
        return "same_origin" if pa == pb else "opposite_origin"
    if pa == 0 and pb == 0:
        return "neutral_origin"
    return "mixed_or_emergent"


def make_transport_environment_07a(n: int, tau_conv: Optional[float], convergent: bool) -> Environment3D:
    """07A-v2 transport environment.

    The v1 radial sink was compressible and, together with non-conservative
    scalar advection, removed membrane material. v2 uses a bounded,
    analytically divergence-free 3-D recirculating strain field. Near the
    domain center its x component has local membrane-advection strain rate
    1/tau_conv, so the original dimensionless timescale ratios are retained.
    Normal velocity is zero on every domain face.
    """
    shape = (n, n, n)
    zeros = np.zeros(shape, dtype=np.float32)
    pore = zeros.copy()
    mineral = zeros.copy()
    vent = zeros.copy()
    residence = np.full(shape, 0.04, dtype=np.float32)
    pressure = np.full(shape, 0.12, dtype=np.float32)
    source_shape = np.full(shape, 0.18, dtype=np.float32)
    shear = np.full(shape, 0.08, dtype=np.float32)
    if not convergent:
        vx = zeros.copy(); vy = zeros.copy(); vz = zeros.copy()
    else:
        if tau_conv is None or tau_conv <= 0:
            raise ValueError("positive tau_conv required for convergence")
        center = (n - 1) / 2.0
        half = max(center, 1.0)
        # Dimensionless coordinates xi,eta,zeta in [-1,1].
        q = (np.arange(n, dtype=np.float64) - center) / half
        sx = np.sin(np.pi * q); cx = np.cos(np.pi * q)
        sy = sx.copy(); cy = cx.copy(); sz = sx.copy(); cz = cx.copy()

        # Sum of two stream-function rolls (x-y and x-z):
        # div(v)=0 analytically, v_normal=0 on all six faces.
        # Near the center, MEMBRANE_ADV_COEFF * vx ~= -x/tau_conv.
        A = half / (2.0 * np.pi**2 * MEMBRANE_ADV_COEFF * float(tau_conv))
        vx = (-A * np.pi * sx[:, None, None] * (cy[None, :, None] + cz[None, None, :])).astype(np.float32)
        vy = ( A * np.pi * cx[:, None, None] * sy[None, :, None]).astype(np.float32)
        vz = ( A * np.pi * cx[:, None, None] * sz[None, None, :]).astype(np.float32)

        # Exact zero normal velocity on faces, including float roundoff.
        vx[0, :, :] = 0.0; vx[-1, :, :] = 0.0
        vy[:, 0, :] = 0.0; vy[:, -1, :] = 0.0
        vz[:, :, 0] = 0.0; vz[:, :, -1] = 0.0
    return Environment3D(pore, mineral, residence, vent, pressure, vx, vy, vz, shear, source_shape)


def transport_audit_07a(env: Environment3D, dt: float) -> Dict[str, float]:
    div = discrete_divergence_centered(env.vx, env.vy, env.vz)
    # The centered diagnostic uses np.roll; exclude the outermost layer because
    # the transport problem uses zero-normal boundary faces rather than a
    # periodic physical boundary. Interior divergence is the relevant audit.
    div_i = div[1:-1, 1:-1, 1:-1] if min(div.shape) > 2 else div
    vmax = float(max(np.max(np.abs(env.vx)), np.max(np.abs(env.vy)), np.max(np.abs(env.vz))))
    # Largest inherited advection coefficient is 0.018 (R/L/H/X/M); B/T use 0.010.
    cfl_scalar = float(dt * 0.018 * vmax)
    cfl_membrane = float(dt * MEMBRANE_ADV_COEFF * vmax)
    bnorm = float(max(
        np.max(np.abs(env.vx[[0, -1], :, :])),
        np.max(np.abs(env.vy[:, [0, -1], :])),
        np.max(np.abs(env.vz[:, :, [0, -1]])),
    ))
    # Conservative-advection domain-sum drift on a deterministic positive probe.
    n = env.vx.shape[0]
    grid = np.indices((n, n, n), dtype=np.float64)
    c = (n - 1) / 2.0
    probe = np.exp(-((grid[0]-c)**2 + (grid[1]-c)**2 + (grid[2]-c)**2)/(2.0*(0.12*n)**2)).astype(np.float32)
    rhs = advect_conservative_upwind(probe, env.vx, env.vy, env.vz)
    denom = float(np.sum(probe))
    drift = float(np.sum(rhs) / denom) if denom > 0 else float('nan')
    return {
        "max_abs_velocity": vmax,
        "cfl_scalar_0p018": cfl_scalar,
        "cfl_membrane_0p010": cfl_membrane,
        "max_abs_centered_divergence": float(np.max(np.abs(div_i))),
        "mean_abs_centered_divergence": float(np.mean(np.abs(div_i))),
        "max_boundary_normal_velocity": bnorm,
        "relative_domain_sum_drift_per_time_probe": drift,
    }

def clone_fields_07a(f: Fields) -> Fields:
    return Fields(R=f.R.copy(), L=f.L.copy(), H=f.H.copy(), X=f.X.copy(), M=f.M.copy(), B=f.B.copy(), T=f.T.copy())


def conditioned_initial_states_07a(labels: np.ndarray, sizes: Dict[int, int], centroids: Dict[int, Tuple[float, float, float]],
                                    tau_c: float, conditioning_duration: float) -> Tuple[Dict[int, CatState07A], Dict[int, int], int]:
    center = (labels.shape[0] - 1) / 2.0
    state_by_track: Dict[int, CatState07A] = {}
    label_to_track: Dict[int, int] = {}
    next_tid = 1
    for lab in sorted(sizes):
        c = centroids.get(lab)
        if c is None:
            continue
        E = -E_AMPLITUDE if c[0] < center else E_AMPLITUDE
        origin = -1.0 if E < 0 else 1.0
        c1d = c1_after_conditioning(E, conditioning_duration, tau_c)
        st = CatState07A(next_tid, c1d, 0.5, origin, float(sizes[lab]))
        state_by_track[next_tid] = st
        label_to_track[int(lab)] = next_tid
        next_tid += 1
    return state_by_track, label_to_track, next_tid


def advance_tracker_07a(
    prev_labels: np.ndarray,
    prev_sizes: Dict[int, int],
    prev_label_to_track: Dict[int, int],
    prev_states: Dict[int, CatState07A],
    curr_labels: np.ndarray,
    curr_sizes: Dict[int, int],
    tau_c: float,
    sample_dt: float,
    next_tid: int,
    seen_pairs: set,
    event_meta: Dict[str, Any],
) -> Tuple[Dict[int, CatState07A], Dict[int, int], int, List[Dict[str, Any]]]:
    by_curr, by_prev = overlaps_07a(prev_labels, curr_labels, prev_label_to_track)
    rows: List[Dict[str, Any]] = []

    # Parent states at the last sampled state before fusion (locked assay state).
    parent_sample_state = prev_states
    # States propagated to the current sample under E=0 relaxation.
    relaxed: Dict[int, CatState07A] = {}
    for tid, st in prev_states.items():
        relaxed[tid] = CatState07A(
            track_id=tid,
            c1_diff=relax_c1(st.c1_diff, sample_dt, tau_c),
            c1_undiff=relax_c1(st.c1_undiff, sample_dt, tau_c),
            origin_balance=st.origin_balance,
            mass=st.mass,
        )

    # If a previous component splits, only the largest-overlap child retains its track id.
    retained_child: Dict[int, int] = {}
    for pl, children in by_prev.items():
        if children:
            retained_child[pl] = children[0][0]

    curr_states: Dict[int, CatState07A] = {}
    curr_label_to_track: Dict[int, int] = {}

    for cl in sorted(curr_sizes):
        parents = by_curr.get(cl, [])
        if len(parents) == 0:
            tid = next_tid; next_tid += 1
            st = CatState07A(tid, 0.5, 0.5, 0.0, float(curr_sizes[cl]))
            curr_states[tid] = st
            curr_label_to_track[cl] = tid
            continue

        parent_tids = [prev_label_to_track[pl] for pl, _ in parents if pl in prev_label_to_track]
        parent_tids = [tid for tid in parent_tids if tid in relaxed]
        if not parent_tids:
            tid = next_tid; next_tid += 1
            st = CatState07A(tid, 0.5, 0.5, 0.0, float(curr_sizes[cl]))
            curr_states[tid] = st
            curr_label_to_track[cl] = tid
            continue

        # Exact two-parent natural fusion assay, first occurrence of the unordered pair only.
        if len(parents) == 2 and len(parent_tids) == 2 and parent_tids[0] != parent_tids[1]:
            key = tuple(sorted(parent_tids))
            if key not in seen_pairs:
                seen_pairs.add(key)
                sa = parent_sample_state[parent_tids[0]]
                sb = parent_sample_state[parent_tids[1]]
                ad = pair_assay_07a(sa.c1_diff, sb.c1_diff)
                au = pair_assay_07a(sa.c1_undiff, sb.c1_undiff)
                origin_class = classify_origin_07a(sa.origin_balance, sb.origin_balance)
                row = dict(event_meta)
                row.update({
                    "parent_track_A": int(parent_tids[0]),
                    "parent_track_B": int(parent_tids[1]),
                    "parent_origin_A": float(sa.origin_balance),
                    "parent_origin_B": float(sb.origin_balance),
                    "origin_class": origin_class,
                    "parent_c1_diff_A": float(sa.c1_diff),
                    "parent_c1_diff_B": float(sb.c1_diff),
                    "parent_c1_undiff_A": float(sa.c1_undiff),
                    "parent_c1_undiff_B": float(sb.c1_undiff),
                    "parent_mass_A": float(sa.mass),
                    "parent_mass_B": float(sb.mass),
                    "G_chem_diff": ad["G_chem"],
                    "G_chem_undiff": au["G_chem"],
                    "J_AB_diff": ad["J_AB"],
                    "J_AA_diff": ad["J_AA"],
                    "J_BB_diff": ad["J_BB"],
                    "theta_support_diff": ad["theta_support_fraction"],
                    "J_AB_undiff": au["J_AB"],
                    "theta_support_undiff": au["theta_support_fraction"],
                    "NO_POOLING_G": 0.0,
                    "NO_SERIAL_G": 0.0,
                })
                rows.append(row)

        if len(parent_tids) == 1:
            pl = parents[0][0]
            pst = relaxed[parent_tids[0]]
            if retained_child.get(pl) == cl:
                tid = parent_tids[0]
            else:
                tid = next_tid; next_tid += 1
            st = CatState07A(tid, pst.c1_diff, pst.c1_undiff, pst.origin_balance, float(curr_sizes[cl]))
        else:
            # Natural fusion state propagation uses physical overlap contribution as mass weight.
            weights = []
            states = []
            for pl, ov in parents:
                tidp = prev_label_to_track.get(pl)
                if tidp is None or tidp not in relaxed:
                    continue
                weights.append(float(ov))
                states.append(relaxed[tidp])
            if not states:
                tid = next_tid; next_tid += 1
                st = CatState07A(tid, 0.5, 0.5, 0.0, float(curr_sizes[cl]))
            else:
                w = np.asarray(weights, dtype=np.float64)
                w /= max(float(w.sum()), EPS_07A)
                c1d = float(sum(float(q) * s.c1_diff for q, s in zip(w, states)))
                c1u = float(sum(float(q) * s.c1_undiff for q, s in zip(w, states)))
                ori = float(sum(float(q) * s.origin_balance for q, s in zip(w, states)))
                tid = next_tid; next_tid += 1
                st = CatState07A(tid, c1d, c1u, ori, float(curr_sizes[cl]))
        curr_states[st.track_id] = st
        curr_label_to_track[cl] = st.track_id

    return curr_states, curr_label_to_track, next_tid, rows


def mode07a(mode: str) -> Dict[str, Any]:
    if mode == "smoke":
        return dict(seed_count=1, scientific=False, time_scale=0.035, source_mode="smoke")
    if mode == "quick":
        return dict(seed_count=2, scientific=False, time_scale=0.10, source_mode="quick")
    if mode == "full":
        return dict(seed_count=32, scientific=True, time_scale=1.0, source_mode="full")
    raise ValueError(mode)


def transport_one_07a(release_fields: Fields, cfg: SimConfig, seed: int, ratio: Optional[float], convergent: bool,
                       tau_c: float, conditioning_duration: float, transport_duration: float,
                       threshold_name: str = "main", b_threshold: Optional[float] = None,
                       t_threshold: Optional[float] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    f = embed_center(clone_fields_07a(release_fields), cfg.open_n)
    # Recentring is deliberately disabled because convergence is defined in a fixed spatial frame.
    cfg_local = SimConfig(**{**asdict(cfg), "recenter": False})
    tau_conv = None if not convergent else float(ratio) * float(tau_c)
    env = make_transport_environment_07a(cfg.open_n, tau_conv, convergent)
    rng = np.random.default_rng(int(seed) + 700001)  # matched stochastic field noise across physical arms

    prev_labels, prev_sizes, centroids = detect_components_07a(f.B, f.T, cfg_local, b_threshold, t_threshold)
    states, prev_map, next_tid = conditioned_initial_states_07a(prev_labels, prev_sizes, centroids, tau_c, conditioning_duration)
    initial_components = len(prev_sizes)
    seen_pairs: set = set()
    event_rows: List[Dict[str, Any]] = []

    nsteps = max(1, int(round(float(transport_duration) / cfg_local.dt)))
    sample_every = max(1, int(cfg_local.sample_every))
    last_sample_step = 0
    max_components = initial_components
    final_components = initial_components

    for step in range(1, nsteps + 1):
        update_fields(f, env, rng, cfg_local, phase="open", advect_fn=advect_conservative_upwind)
        if step % sample_every != 0 and step != nsteps:
            continue
        curr_labels, curr_sizes, _ = detect_components_07a(f.B, f.T, cfg_local, b_threshold, t_threshold)
        max_components = max(max_components, len(curr_sizes))
        final_components = len(curr_sizes)
        sample_dt = (step - last_sample_step) * cfg_local.dt
        event_meta = {
            "seed": int(seed),
            "ratio": "NA" if ratio is None else float(ratio),
            "convergent": int(bool(convergent)),
            "threshold": threshold_name,
            "fusion_step": int(step),
            "parent_sample_step": int(last_sample_step),
            "fusion_time": float(step * cfg_local.dt),
            "parent_sample_time": float(last_sample_step * cfg_local.dt),
        }
        states, curr_map, next_tid, rows = advance_tracker_07a(
            prev_labels, prev_sizes, prev_map, states,
            curr_labels, curr_sizes, tau_c, sample_dt,
            next_tid, seen_pairs, event_meta,
        )
        event_rows.extend(rows)
        prev_labels, prev_sizes, prev_map = curr_labels, curr_sizes, curr_map
        last_sample_step = step

    summary = {
        "seed": int(seed),
        "ratio": "NA" if ratio is None else float(ratio),
        "convergent": int(bool(convergent)),
        "threshold": threshold_name,
        "initial_components": int(initial_components),
        "max_components": int(max_components),
        "final_components": int(final_components),
        "eligible_exact_two_parent_fusions": int(len(event_rows)),
        "transport_steps": int(nsteps),
        "transport_duration": float(transport_duration),
        "tau_c": float(tau_c),
        "tau_conv": "NA" if tau_conv is None else float(tau_conv),
    }
    summary.update(transport_audit_07a(env, cfg_local.dt))
    if event_rows:
        gd = np.array([r["G_chem_diff"] for r in event_rows], dtype=float)
        gu = np.array([r["G_chem_undiff"] for r in event_rows], dtype=float)
        summary.update({
            "mean_G_diff": float(np.mean(gd)),
            "mean_G_undiff": float(np.mean(gu)),
            "mean_support_diff": float(np.mean([r["theta_support_diff"] for r in event_rows])),
            "mean_support_undiff": float(np.mean([r["theta_support_undiff"] for r in event_rows])),
            "opposite_origin_events": int(sum(r["origin_class"] == "opposite_origin" for r in event_rows)),
            "same_origin_events": int(sum(r["origin_class"] == "same_origin" for r in event_rows)),
        })
    else:
        summary.update({
            "mean_G_diff": float("nan"), "mean_G_undiff": float("nan"),
            "mean_support_diff": float("nan"), "mean_support_undiff": float("nan"),
            "opposite_origin_events": 0, "same_origin_events": 0,
        })
    return event_rows, summary


def load_release_from_v1_zip_07a(zip_path: Path, seed: int) -> Tuple[Fields, int, Dict[str, Any]]:
    """Reuse the unchanged v1 full fixed-release state for a given seed.

    Formation physics is identical between v1 and v2; only the post-release
    transport layer changed. This avoids recomputing the 32 formation runs.
    """
    suffix = f"/_work/seed_{int(seed):06d}_formation/fixed_release_state.npz"
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        matches = [n for n in zf.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise FileNotFoundError(f"expected exactly one release state for seed {seed} in {zip_path}; found {len(matches)}")
        raw = zf.read(matches[0])
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        release_step = int(np.asarray(d["release_step"]).item())
        fields = Fields(
            R=np.asarray(d["R"], dtype=np.float32).copy(),
            L=np.asarray(d["L"], dtype=np.float32).copy(),
            H=np.asarray(d["H"], dtype=np.float32).copy(),
            X=np.asarray(d["X"], dtype=np.float32).copy(),
            M=np.asarray(d["M"], dtype=np.float32).copy(),
            B=np.asarray(d["B"], dtype=np.float32).copy(),
            T=np.asarray(d["T"], dtype=np.float32).copy(),
        )
        summary = json.loads(str(np.asarray(d["release_summary_json"]).item()))
    return fields, release_step, summary


def audit_release_zip_07a(zip_path: Path, seeds: Sequence[int]) -> Dict[str, Any]:
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()
    missing = []
    for seed in seeds:
        suffix = f"/_work/seed_{int(seed):06d}_formation/fixed_release_state.npz"
        if sum(1 for n in names if n.endswith(suffix)) != 1:
            missing.append(int(seed))
    if missing:
        raise RuntimeError(f"release zip missing/ambiguous seeds: {missing}")
    return {"release_zip": str(zip_path), "release_zip_sha256": _sha256(zip_path), "release_seed_count_verified": len(seeds)}


def seed_run_07a(seed: int, mode: str, cfg_args: Dict[str, Any]) -> Dict[str, Any]:
    md = mode07a(mode)
    d = mode_defaults(md["source_mode"])
    cfg = SimConfig(
        seed=int(seed), hole_count=10, fixed_total_pore=False, reference_hole_count=10,
        confined_n=int(d["confined_n"]), open_n=int(d["open_n"]), pre_steps=int(d["pre_steps"]),
        gen_steps=int(d["gen_steps"]), open_steps=int(d["open_steps"]),
        sample_every=max(1, int(round(int(d["sample_every"]) * float(md["time_scale"])))),
        dt=0.035, b_threshold=0.16, t_threshold=0.018, min_component_voxels=12, min_lumen_voxels=20,
        source_regime="continuous", recenter=False, save_float16=False,
        release_min_components=8, release_check_every=20, release_min_gen_step=0,
        release_max_largest_fraction=0.0, release_fallback="end",
    )
    # Scientific full uses the frozen durations and tau_C. Smoke/quick shorten
    # wall-clock duration only; tau_C is kept at the scientific value so the
    # fastest locked flow remains within the same CFL-safe numerical regime.
    # Smoke/quick are code-path checks and are never used for effect inference.
    scale = float(md["time_scale"])
    conditioning_duration = CONDITIONING_DURATION_FULL * scale
    transport_duration = TRANSPORT_DURATION_FULL * scale
    tau_c = TAU_C_BASE

    release_zip = cfg_args.get("release_zip")
    if release_zip:
        if mode != "full":
            raise RuntimeError("--release-zip reuse is restricted to mode=full")
        release_fields, release_step, release_summary = load_release_from_v1_zip_07a(Path(release_zip), seed)
    else:
        tmp = Path(cfg_args["work_root"]) / f"seed_{int(seed):06d}_formation"
        tmp.mkdir(parents=True, exist_ok=True)
        logger = ProgressLogger(tmp)
        try:
            release_fields, release_step, release_summary = generate_fixed_release_state(cfg, tmp, logger)
        finally:
            logger.close()

    all_events: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    # No-convergence is physical-ratio independent and is run once.
    ev0, sm0 = transport_one_07a(release_fields, cfg, seed, None, False, tau_c, conditioning_duration, transport_duration)
    sm0["release_step"] = release_step; sm0["release_component_count"] = release_summary.get("component_count", "")
    all_events.extend(ev0); summaries.append(sm0)
    for ratio in RATIOS_07A:
        ev, sm = transport_one_07a(release_fields, cfg, seed, ratio, True, tau_c, conditioning_duration, transport_duration)
        sm["release_step"] = release_step; sm["release_component_count"] = release_summary.get("component_count", "")
        all_events.extend(ev); summaries.append(sm)
    return {"seed": seed, "events": all_events, "summaries": summaries, "scientific": md["scientific"]}


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        r = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = r
        i = j
    return ranks


def wilcoxon_exact_07a(x: List[float]) -> Tuple[float, float, int]:
    vals = np.asarray([float(v) for v in x if np.isfinite(v) and abs(float(v)) > 1e-15], dtype=float)
    n = len(vals)
    if n == 0:
        return 0.0, 1.0, 0
    ranks = _rankdata_average(np.abs(vals))
    r2 = np.rint(ranks * 2.0).astype(int)
    obs = int(np.sum(r2[vals > 0]))
    total = int(np.sum(r2))
    # DP subset-sum distribution over sign assignments; exact and efficient for n<=32.
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    curmax = 0
    for r in r2:
        counts[r:curmax + r + 1] += counts[:curmax + 1]
        curmax += int(r)
    denom = float(2 ** n)
    lo = min(obs, total - obs)
    hi = max(obs, total - obs)
    p = min(1.0, float((counts[:lo + 1].sum() + counts[hi:].sum()) / denom))
    wplus = float(obs) / 2.0
    wminus = float(total - obs) / 2.0
    return min(wplus, wminus), p, n


def holm_07a(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for j, idx in enumerate(order):
        val = min(1.0, (m - j) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def bootstrap_ci_07a(vals: List[float], seed: int = 707, nboot: int = 10000) -> Tuple[float, float]:
    x = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = np.empty(nboot, dtype=float)
    for i in range(nboot):
        means[i] = float(np.mean(rng.choice(x, size=len(x), replace=True)))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_primary_07a(condition_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = _defaultdict(dict)
    for r in condition_rows:
        seed = int(r["seed"])
        key = "NO_CONV" if int(r["convergent"]) == 0 else f"R{float(r['ratio']):g}"
        by_seed[seed][key] = r

    interactions: List[Dict[str, Any]] = []
    tests: List[Dict[str, Any]] = []
    pvals = []
    temp_tests = []
    for ratio in RATIOS_07A:
        vals = []
        for seed in sorted(by_seed):
            c = by_seed[seed].get(f"R{ratio:g}")
            n = by_seed[seed].get("NO_CONV")
            if not c or not n:
                continue
            nums = [c.get("mean_G_diff"), c.get("mean_G_undiff"), n.get("mean_G_diff"), n.get("mean_G_undiff")]
            try:
                nums = [float(v) for v in nums]
            except Exception:
                continue
            if not all(np.isfinite(nums)):
                continue
            interaction = (nums[0] - nums[2]) - (nums[1] - nums[3])
            vals.append(interaction)
            interactions.append({
                "seed": seed, "ratio": ratio, "interaction": interaction,
                "diff_conv": nums[0], "diff_no_conv": nums[2],
                "undiff_conv": nums[1], "undiff_no_conv": nums[3],
            })
        stat, p, n = wilcoxon_exact_07a(vals)
        lo, hi = bootstrap_ci_07a(vals, seed=7070 + int(ratio * 100))
        arr = np.asarray(vals, dtype=float)
        row = {
            "ratio": ratio, "n_complete_seeds": n,
            "mean_interaction": float(np.mean(arr)) if len(arr) else float("nan"),
            "sd_interaction": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0 if len(arr) == 1 else float("nan"),
            "median_interaction": float(np.median(arr)) if len(arr) else float("nan"),
            "bootstrap95_lo": lo, "bootstrap95_hi": hi,
            "positive": int(np.sum(arr > 0)) if len(arr) else 0,
            "zero": int(np.sum(np.abs(arr) <= 1e-15)) if len(arr) else 0,
            "negative": int(np.sum(arr < 0)) if len(arr) else 0,
            "wilcoxon_statistic": stat, "p_raw": p,
        }
        temp_tests.append(row); pvals.append(p)
    adj = holm_07a(pvals)
    for row, pa in zip(temp_tests, adj):
        row["p_holm"] = pa
        tests.append(row)
    return interactions, tests


def self_test_07a(verbose: bool = True) -> Tuple[int, int, List[str]]:
    checks: List[Tuple[str, bool]] = []
    checks.append(("lock_constant_sha_length", len(EXPECTED_07A_LOCK_SHA256) == 64))
    checks.append(("correction_lock_constant_sha_length", len(EXPECTED_07A_V2_CORRECTION_LOCK_SHA256) == 64))
    cp = c1_after_conditioning(1.0, 30.0, 10.0)
    cm = c1_after_conditioning(-1.0, 30.0, 10.0)
    checks.append(("conditioning_symmetry", abs((cp + cm) - 1.0) < 1e-12))
    checks.append(("conditioning_direction", cp > 0.5 and cm < 0.5))
    checks.append(("catalyst_budget", abs(cp + (1.0 - cp) - 1.0) < 1e-15))
    cr = relax_c1(cp, 10.0, 10.0)
    checks.append(("E0_relaxation", abs((cr - 0.5) - (cp - 0.5) / math.e) < 1e-12))
    opp = pair_assay_07a(cm, cp)
    same = pair_assay_07a(cp, cp)
    checks.append(("opposite_pair_positive_gain", opp["G_chem"] > 0))
    checks.append(("opposite_pair_positive_theta_support", opp["theta_support_fraction"] > 0))
    checks.append(("same_history_zero_gain", abs(same["G_chem"]) < 1e-15))
    checks.append(("same_history_zero_theta_support", abs(same["theta_support_fraction"]) < 1e-15))
    checks.append(("no_pooling_zero", opp["NO_POOLING_G"] == 0.0))
    checks.append(("no_serial_zero", opp["NO_SERIAL_G"] == 0.0))
    # Synthetic exact two-parent overlap audit.
    prev = np.zeros((8, 8, 8), dtype=np.int32); curr = np.zeros_like(prev)
    prev[1:4, 1:4, 1:4] = 1; prev[4:7, 1:4, 1:4] = 2
    curr[2:6, 1:4, 1:4] = 1
    bc, bp = overlaps_07a(prev, curr, {1: 11, 2: 22})
    checks.append(("synthetic_exact_two_parent_overlap", len(bc.get(1, [])) == 2))
    st, p, n = wilcoxon_exact_07a([1, 2, 3, 4])
    checks.append(("wilcoxon_exact_valid", n == 4 and 0 <= p <= 1))
    ha = holm_07a([0.01, 0.02, 0.2])
    checks.append(("holm_valid", all(0 <= q <= 1 for q in ha) and ha[0] <= ha[1] <= ha[2]))
    # Convergence environment takes no catalyst arguments and no q/state-dependent term.
    env = make_transport_environment_07a(12, 5.0, True)
    checks.append(("convergence_field_finite", np.all(np.isfinite(env.vx)) and np.all(np.isfinite(env.vy)) and np.all(np.isfinite(env.vz))))
    aud = transport_audit_07a(env, 0.035)
    checks.append(("transport_boundary_no_flux", aud["max_boundary_normal_velocity"] <= 1e-7))
    checks.append(("transport_mass_conservative_probe", abs(aud["relative_domain_sum_drift_per_time_probe"]) <= 1e-6))
    checks.append(("transport_divergence_free_interior", aud["max_abs_centered_divergence"] < 1e-3))
    checks.append(("transport_cfl_safe_ratio_0p25", aud["cfl_scalar_0p018"] < 0.8))
    passed = sum(int(ok) for _, ok in checks)
    msgs = [f"{'PASS' if ok else 'FAIL'} {name}" for name, ok in checks]
    if verbose:
        for m in msgs: print(m)
        print(f"SELF_TEST {passed}/{len(checks)} PASS" if passed == len(checks) else f"SELF_TEST {passed}/{len(checks)} FAIL")
    return passed, len(checks), msgs


def parse_args_07a() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="07A-v2 standalone environmental-history explicit chemical closure validation")
    p.add_argument("--mode", choices=["smoke", "quick", "full"], default="smoke")
    p.add_argument("--seed-count", type=int, default=None)
    p.add_argument("--seed-start", type=int, default=70000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--outdir", type=Path, default=Path.home() / "Desktop" / "ENV_HISTORY_EXPLICIT_CHEMICAL_CLOSURE_V07A_V2")
    p.add_argument("--method-lock", type=Path, default=None, help="Optional external audit file; normally omit because the frozen lock is embedded.")
    p.add_argument("--correction-lock", type=Path, default=None, help="Optional external audit file; normally omit because the frozen correction lock is embedded.")
    p.add_argument("--release-zip", type=Path, default=None, help="Optional previous v1 full output zip; reuses unchanged fixed release states and reruns transport only.")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def write_report_07a(outdir: Path, args: argparse.Namespace, lock_path: Path, lock_sha: str,
                      conditions: List[Dict[str, Any]], events: List[Dict[str, Any]], tests: List[Dict[str, Any]],
                      scientific: bool, elapsed: float, self_pass: int, self_total: int) -> None:
    lines = [
        "# 07A Environmental-history-generated explicit chemical closure",
        "",
        f"script_version: {VERSION_07A}",
        f"mode: {args.mode}",
        f"scientific_inference_allowed: {str(bool(scientific)).upper()}",
        f"method_lock_sha256: {lock_sha}",
        f"method_lock_path: {lock_path}",
        f"source_core_name: {SOURCE_CORE_NAME}",
        f"source_core_sha256: {SOURCE_CORE_SHA256}",
        f"self_test: {self_pass}/{self_total} PASS",
        f"elapsed_seconds: {elapsed:.3f}",
        "",
        "No genes, mutation, selection, fitness, sex, mating types, partner sensing, catalyst-dependent movement, pair bonus, programmed reproduction, or programmed fission are present.",
        "",
        "## Locked interpretation boundary",
        "",
        "07A tests environmental-history-generated explicit chemical complementarity and natural fusion-associated pair-level reaction closure only. Y does not feed back to B or T. Physical fission and biological reproduction are not tested.",
        "",
        "## Run status",
        "",
        f"- condition rows: {len(conditions)}",
        f"- exact two-parent fusion assay rows: {len(events)}",
    ]
    if not scientific:
        lines.extend(["", "**NON-SCIENTIFIC MODE:** smoke/quick durations are shortened only to validate code paths while tau_C remains at the full numerical value for CFL safety. Do not interpret effect sizes or p-values from this run."])
    lines.extend(["", "## Condition summaries", ""])
    for r in conditions:
        lines.append(
            f"- seed={r.get('seed')} convergent={r.get('convergent')} ratio={r.get('ratio')}: "
            f"events={r.get('eligible_exact_two_parent_fusions')}; mean_G_diff={r.get('mean_G_diff')}; "
            f"mean_G_undiff={r.get('mean_G_undiff')}; opposite={r.get('opposite_origin_events')}; same={r.get('same_origin_events')}"
        )
    lines.extend(["", "## Primary interaction tests", ""])
    for r in tests:
        lines.append(
            f"- ratio={r['ratio']}: n={r['n_complete_seeds']}; mean={r['mean_interaction']}; "
            f"95% bootstrap CI=[{r['bootstrap95_lo']},{r['bootstrap95_hi']}]; "
            f"+ / 0 / - = {r['positive']}/{r['zero']}/{r['negative']}; p_raw={r['p_raw']}; p_Holm={r['p_holm']}"
        )
    (outdir / "09_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_07a() -> int:
    args = parse_args_07a()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        lock_path, lock_sha = locate_method_lock_07a(args.method_lock)
        correction_lock_path, correction_lock_sha = locate_correction_lock_07a(args.correction_lock)
        sp, st, smsg = self_test_07a(verbose=True)
        if sp != st:
            raise RuntimeError("07A self-test failed")
        if args.self_test:
            return 0
        md = mode07a(args.mode)
        seed_count = int(args.seed_count) if args.seed_count is not None else int(md["seed_count"])
        seeds = [int(args.seed_start) + i for i in range(seed_count)]
        workers = max(1, min(int(args.workers), len(seeds)))
        work_root = outdir / "_work"
        work_root.mkdir(exist_ok=True)
        release_zip_audit = None
        if args.release_zip is not None:
            if args.mode != "full":
                raise RuntimeError("--release-zip may be used only with --mode full")
            release_zip_audit = audit_release_zip_07a(args.release_zip, seeds)
            print(f"REUSING {seed_count} validated fixed release states from {release_zip_audit['release_zip']}", flush=True)
        print(f"07A START mode={args.mode} seeds={seed_count} workers={workers}", flush=True)
        results = []
        cfg_args = {"work_root": str(work_root), "release_zip": None if args.release_zip is None else str(args.release_zip.expanduser().resolve())}
        if workers == 1:
            for i, seed in enumerate(seeds, 1):
                print(f"[{i}/{len(seeds)}] seed {seed} start", flush=True)
                res = seed_run_07a(seed, args.mode, cfg_args)
                results.append(res)
                print(f"[{i}/{len(seeds)}] seed {seed} done", flush=True)
        else:
            ctx = _mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(seed_run_07a, seed, args.mode, cfg_args): seed for seed in seeds}
                done = 0
                for fut in as_completed(futs):
                    seed = futs[fut]
                    res = fut.result()
                    results.append(res)
                    done += 1
                    print(f"[{done}/{len(seeds)}] seed {seed} done", flush=True)
        results.sort(key=lambda r: r["seed"])
        conditions = [x for r in results for x in r["summaries"]]
        events = [x for r in results for x in r["events"]]
        interactions, tests = summarize_primary_07a(conditions)

        write_csv(outdir / "01_METHOD_LOCK_AUDIT.csv", [{
            "script_version": VERSION_07A, "method_lock_path": str(lock_path), "method_lock_sha256": lock_sha,
            "expected_method_lock_sha256": EXPECTED_07A_LOCK_SHA256,
            "correction_lock_path": str(correction_lock_path), "correction_lock_sha256": correction_lock_sha,
            "expected_correction_lock_sha256": EXPECTED_07A_V2_CORRECTION_LOCK_SHA256, "status": "PASS",
            "source_core_name": SOURCE_CORE_NAME, "source_core_sha256": SOURCE_CORE_SHA256,
            "release_source": "reused_v1_zip" if release_zip_audit else "regenerated",
            "release_zip": "" if not release_zip_audit else release_zip_audit["release_zip"],
            "release_zip_sha256": "" if not release_zip_audit else release_zip_audit["release_zip_sha256"],
            "release_seed_count_verified": 0 if not release_zip_audit else release_zip_audit["release_seed_count_verified"],
        }])
        write_csv(outdir / "02_SEED_CONDITION_SUMMARY.csv", conditions)
        write_csv(outdir / "03_FUSION_EVENT_ASSAYS.csv", events)
        write_csv(outdir / "04_PRIMARY_INTERACTIONS.csv", interactions)
        write_csv(outdir / "05_PRIMARY_TESTS.csv", tests)
        control_rows = [{
            "events": len(events),
            "NO_POOLING_max_abs_G": max([abs(float(r["NO_POOLING_G"])) for r in events], default=0.0),
            "NO_SERIAL_max_abs_G": max([abs(float(r["NO_SERIAL_G"])) for r in events], default=0.0),
            "undifferentiated_max_abs_G": max([abs(float(r["G_chem_undiff"])) for r in events], default=0.0),
        }]
        write_csv(outdir / "06_CONTROL_AUDIT.csv", control_rows)
        origin_rows = []
        for cls in ["opposite_origin", "same_origin", "mixed_or_emergent", "neutral_origin"]:
            rr = [r for r in events if r.get("origin_class") == cls]
            origin_rows.append({
                "origin_class": cls, "events": len(rr),
                "mean_G_diff": float(np.mean([r["G_chem_diff"] for r in rr])) if rr else float("nan"),
                "mean_theta_support_diff": float(np.mean([r["theta_support_diff"] for r in rr])) if rr else float("nan"),
            })
        write_csv(outdir / "07_ORIGIN_AUDIT.csv", origin_rows)
        transport_audit_rows = []
        seen_audit = set()
        for r in conditions:
            key = (r.get("ratio"), r.get("convergent"))
            if key in seen_audit:
                continue
            seen_audit.add(key)
            transport_audit_rows.append({k: r.get(k) for k in ["ratio","convergent","tau_conv","max_abs_velocity","cfl_scalar_0p018","cfl_membrane_0p010","max_abs_centered_divergence","mean_abs_centered_divergence","max_boundary_normal_velocity","relative_domain_sum_drift_per_time_probe"]})
        write_csv(outdir / "08_TRANSPORT_NUMERICAL_AUDIT.csv", transport_audit_rows)
        elapsed = time.time() - t0
        write_report_07a(outdir, args, lock_path, lock_sha, conditions, events, tests, bool(md["scientific"]), elapsed, sp, st)
        print(f"07A DONE elapsed={elapsed:.3f}s outdir={outdir}", flush=True)
        return 0
    except Exception as e:
        print(f"07A ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        (outdir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1



# =============================================================================
# 07D independent end-to-end confirmatory layer
# =============================================================================

try:
    from scipy.stats import binomtest as _binomtest_07d
except Exception as _exc07d:
    raise SystemExit("scipy is required for 07D: python3 -m pip install scipy") from _exc07d

SCRIPT_VERSION_07D = "07D-independent-end-to-end-two-parent-fission-1.0.0"
DISCOVERY_07C_METHOD_LOCK_SHA256 = "730181905f13de0491d3b5facd6991b42ade48ef5a205bab066bb5961a73c1ae"
DISCOVERY_07C_SCRIPT_VERSION = "07C-chemical-closure-to-membrane-fission-1.0.0"
PRIMARY_RATIO_07D = 1.0
FULL_SEED_START_07D = 71000
FULL_SEED_COUNT_07D = 32
DISCOVERY_SEED_MIN_07D = 70000
DISCOVERY_SEED_MAX_07D = 70031
PHYSICAL_ROOT_SEED_07D = 2026082800
ARMS_07D = ("AB_POOL", "AA_EQUAL_MASS", "BB_EQUAL_MASS", "AB_NO_POOL", "NO_Y")
PRIMARY_CONTROLS_07D = ("AA_EQUAL_MASS", "BB_EQUAL_MASS", "AB_NO_POOL")

# Exact 07C physical mechanics are inherited without alteration.
J_MAX_07D = 0.25
TAU_FISSION_REF_REFRESHES_07D = 4.5
GRID_07D = 64
EPSILON_07D = 1.5
INITIAL_RADIUS_07D = 9.0
INITIAL_CENTER_SEPARATION_07D = 12.0
SURFACE_TENSION_07D = 1.0
PERIMETER_PENALTY_07D = 1500.0
BENDING_COEFFICIENT_07D = 2.0
PHASE_DT_07D = 0.0012
PHASE_STEPS_PER_REFRESH_07D = 60
ASSAY_HORIZON_REFRESHES_07D = 12
MIN_DAUGHTER_PIXELS_07D = 20
PHI_COMPONENT_THRESHOLD_07D = 0.0
NUMERICAL_PHI_ABS_LIMIT_07D = 2.0
AREA_REL_DRIFT_LIMIT_07D = 2.0e-5
DAUGHTER_PERIMETER_FRACTION_07D = math.sqrt(2.0) - 1.0
PREFLIGHT_SEEDS_07D = (92001, 92002, 92003, 92004)
PREFLIGHT_PROFILES_07D = (0.0, 0.5, 1.0)

METHOD_LOCK_07D: Dict[str, Any] = {
    "study_id": "07D_INDEPENDENT_END_TO_END_TWO_PARENT_FISSION_CONFIRMATION",
    "status": "FROZEN_AFTER_07C_BEFORE_ANY_07D_SCIENTIFIC_DATA",
    "discovery_parent": {
        "07c_method_lock_sha256": DISCOVERY_07C_METHOD_LOCK_SHA256,
        "07c_script_version": DISCOVERY_07C_SCRIPT_VERSION,
        "discovery_observation_used_for_design_only": (
            "07C ratio=1.0 showed the clearest pair-specific physical separation; "
            "the inherited ratio=2.0 confirmatory test was null. Ratio=1.0 is now the sole "
            "prospective operating point for an independent seed block."
        ),
    },
    "independence": {
        "discovery_seeds": "70000-70031",
        "confirmatory_full_seeds": "71000-71031",
        "fresh_release_states": True,
        "old_release_zip_forbidden": True,
        "fresh_environment_and_transport_random_streams": True,
    },
    "07a_parent_locks": {
        "method_lock_sha256": EXPECTED_07A_LOCK_SHA256,
        "transport_correction_lock_sha256": EXPECTED_07A_V2_CORRECTION_LOCK_SHA256,
        "source_core_name": SOURCE_CORE_NAME,
        "source_core_sha256": SOURCE_CORE_SHA256,
    },
    "single_primary_operating_point": {
        "tau_conv_over_tau_C": PRIMARY_RATIO_07D,
        "other_ratios_run": False,
        "posthoc_ratio_substitution_forbidden": True,
    },
    "fresh_07a_assay": {
        "conditions": [
            "DIFFERENTIATED+CONVERGENCE ratio=1.0",
            "DIFFERENTIATED+NO_CONVERGENCE",
            "UNDIFFERENTIATED readout under the same two physical conditions",
        ],
        "chemical_seed_metric": (
            "(G_diff_convergence-G_diff_no_convergence)-"
            "(G_undiff_convergence-G_undiff_no_convergence)"
        ),
        "chemical_test": "two-sided exact Wilcoxon signed-rank across fresh seeds; alpha=0.05",
        "fusion_for_physics": (
            "chronologically first eligible main-threshold exact two-parent fusion in the "
            "fresh differentiated+convergence ratio=1.0 run"
        ),
        "forbidden_event_filters": ["G_chem sign", "origin class", "J_AB magnitude", "later fission"],
    },
    "chemical_arms": {
        "AB_POOL": "c1_0=(c1_A+c1_B)/2; equal total catalyst",
        "AA_EQUAL_MASS": "c1_0=c1_A; equal total catalyst",
        "BB_EQUAL_MASS": "c1_0=c1_B; equal total catalyst",
        "AB_NO_POOL": "0.5*J(c1_A(t))+0.5*J(c1_B(t))",
        "NO_Y": "zero membrane-material production",
    },
    "physical_handoff_inherited_unchanged_from_07c": {
        "J": "c1*(1-c1)",
        "J_max": J_MAX_07D,
        "tau_C_refreshes": TAU_FISSION_REF_REFRESHES_07D,
        "target_perimeter": "S_target=S0*[1+(sqrt(2)-1)*M_Y]",
        "grid": GRID_07D,
        "epsilon": EPSILON_07D,
        "initial_radius": INITIAL_RADIUS_07D,
        "initial_center_separation": INITIAL_CENTER_SEPARATION_07D,
        "surface_tension": SURFACE_TENSION_07D,
        "perimeter_penalty": PERIMETER_PENALTY_07D,
        "bending_coefficient": BENDING_COEFFICIENT_07D,
        "phase_dt": PHASE_DT_07D,
        "phase_steps_per_refresh": PHASE_STEPS_PER_REFRESH_07D,
        "horizon_refreshes": ASSAY_HORIZON_REFRESHES_07D,
        "area_conservation": "subtract mean dphi every phase step",
        "fission_observer": "first refresh with >=2 macroscopic connected components of phi>0",
    },
    "primary_statistics": {
        "unit": "fresh 07D seed",
        "chemical_gate": "two-sided exact Wilcoxon p<0.05 at ratio=1.0",
        "physical_family": "exact paired McNemar AB_POOL vs AA, BB, AB_NO_POOL; Holm across 3",
        "end_to_end_confirmation": "chemical_gate PASS and all three physical p_Holm<0.05",
        "NO_Y": "mechanistic control, reported but not an extra success gate",
        "missing_natural_fusion": "missing; never imputed as zero/failure",
    },
    "forbidden_after_lock": [
        "changing ratio 1.0 after seeing 07D results",
        "reusing discovery seeds or old release states",
        "changing 07A chemistry, transport, thresholds, or event definition",
        "changing 07C physical parameters",
        "selecting a favorable fusion rather than the first eligible fusion",
        "adding a fission trigger, J threshold, division axis, partner sensing, or pair attraction",
        "adding extra success gates after seeing the result",
    ],
}
METHOD_LOCK_JSON_07D = json.dumps(METHOD_LOCK_07D, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
METHOD_LOCK_SHA256_07D = hashlib.sha256(METHOD_LOCK_JSON_07D.encode("utf-8")).hexdigest()
EMBEDDED_METHOD_LOCK_B64_07D = base64.b64encode(METHOD_LOCK_JSON_07D.encode("utf-8")).decode("ascii")


def stable_seed_07d(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & 0x7FFFFFFF


def throughput_07d(c1: float) -> float:
    c = float(np.clip(float(c1), 0.0, 1.0))
    return c * (1.0 - c)


def relax_c1_07d(c1_0: float, t_refreshes: float) -> float:
    return float(0.5 + (float(c1_0) - 0.5) * math.exp(-float(t_refreshes) / TAU_FISSION_REF_REFRESHES_07D))


def arm_throughput_07d(arm: str, c1a0: float, c1b0: float, t_refreshes: float) -> float:
    if arm == "NO_Y":
        return 0.0
    ca = relax_c1_07d(c1a0, t_refreshes)
    cb = relax_c1_07d(c1b0, t_refreshes)
    if arm == "AB_POOL":
        cp = relax_c1_07d(0.5 * (c1a0 + c1b0), t_refreshes)
        return throughput_07d(cp)
    if arm == "AA_EQUAL_MASS":
        return throughput_07d(ca)
    if arm == "BB_EQUAL_MASS":
        return throughput_07d(cb)
    if arm == "AB_NO_POOL":
        return 0.5 * throughput_07d(ca) + 0.5 * throughput_07d(cb)
    raise ValueError(arm)


def laplacian_07d(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1) - 4.0 * a
    )


def initialize_fused_vesicle_07d(physical_seed: int) -> np.ndarray:
    n = GRID_07D
    y, x = np.indices((n, n), dtype=np.float64)
    cy = cx = (n - 1) / 2.0
    rng = np.random.default_rng(stable_seed_07d("07d-vesicle", physical_seed))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    ux, uy = math.cos(theta), math.sin(theta)
    d = 0.5 * INITIAL_CENTER_SEPARATION_07D
    c1x, c1y = cx - d * ux, cy - d * uy
    c2x, c2y = cx + d * ux, cy + d * uy
    r1 = np.sqrt((x - c1x) ** 2 + (y - c1y) ** 2)
    r2 = np.sqrt((x - c2x) ** 2 + (y - c2y) ** 2)
    p1 = np.tanh((INITIAL_RADIUS_07D - r1) / (math.sqrt(2.0) * EPSILON_07D))
    p2 = np.tanh((INITIAL_RADIUS_07D - r2) / (math.sqrt(2.0) * EPSILON_07D))
    phi = np.maximum(p1, p2)
    interface = np.exp(-((np.minimum(np.abs(r1 - INITIAL_RADIUS_07D), np.abs(r2 - INITIAL_RADIUS_07D)) / 2.5) ** 2))
    phi = phi + 0.0015 * rng.normal(size=phi.shape) * interface
    return phi.astype(np.float64)


def membrane_functional_07d(phi: np.ndarray) -> Tuple[float, float]:
    dx = np.roll(phi, -1, axis=1) - phi
    dy = np.roll(phi, -1, axis=0) - phi
    W = 0.25 * (phi * phi - 1.0) ** 2
    S = float(np.sum(0.5 * EPSILON_07D * (dx * dx + dy * dy) + W / EPSILON_07D))
    area_phase = float(np.sum(0.5 * (phi + 1.0)))
    return area_phase, S


def positive_components_07d(phi: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    labels, n = ndi.label(phi > PHI_COMPONENT_THRESHOLD_07D)
    if n <= 0:
        return labels.astype(np.int32), []
    sizes = np.bincount(labels.ravel())
    keep = [int(i) for i in range(1, len(sizes)) if int(sizes[i]) >= MIN_DAUGHTER_PIXELS_07D]
    return labels.astype(np.int32), keep


def state_hash_07d(phi: np.ndarray) -> str:
    return hashlib.sha256(phi.astype(np.float64, copy=False).tobytes()).hexdigest()


def phase_step_07d(phi: np.ndarray, target_S: float, S0: float) -> Tuple[float, float, float]:
    mu = -EPSILON_07D * laplacian_07d(phi) + (phi ** 3 - phi) / EPSILON_07D
    gb = -EPSILON_07D * laplacian_07d(mu) + ((3.0 * phi * phi - 1.0) / EPSILON_07D) * mu
    _, S = membrane_functional_07d(phi)
    effective_tension = SURFACE_TENSION_07D + PERIMETER_PENALTY_07D * (S - target_S) / (S0 * S0)
    dphi = -(effective_tension * mu + BENDING_COEFFICIENT_07D * gb)
    dphi -= float(np.mean(dphi))
    phi += PHASE_DT_07D * dphi
    maxabs = float(np.max(np.abs(phi)))
    if not np.isfinite(maxabs) or maxabs > NUMERICAL_PHI_ABS_LIMIT_07D:
        raise FloatingPointError(f"07D phase field unstable: max|phi|={maxabs}")
    return S, effective_tension, maxabs


@dataclass(frozen=True)
class BranchResult07D:
    seed: int
    ratio: float
    arm: str
    physical_seed: int
    parent_c1_A: float
    parent_c1_B: float
    G_chem_07A: float
    J_AB_07A: float
    J_AA_07A: float
    J_BB_07A: float
    initial_state_hash: str
    initial_phase_area: float
    initial_membrane_functional: float
    fission_success: int
    fission_refresh: Optional[int]
    daughter_count: int
    daughter_area_ratio_minmax: Optional[float]
    final_component_count: int
    final_phase_area: float
    final_membrane_functional: float
    max_membrane_functional_ratio: float
    integrated_J_refresh: float
    integrated_normalized_Y: float
    final_target_perimeter_ratio: float
    minimum_effective_tension: float
    maximum_abs_phi: float
    relative_phase_area_drift: float
    termination_reason: str


def assay_branch_07d(event: Mapping[str, Any], arm: str) -> BranchResult07D:
    seed = int(event["07a_seed"])
    pseed = PHYSICAL_ROOT_SEED_07D + (seed - FULL_SEED_START_07D)
    phi = initialize_fused_vesicle_07d(pseed)
    init_hash = state_hash_07d(phi)
    A0, S0 = membrane_functional_07d(phi)
    initial_integral = float(np.sum(phi))
    _, comps0 = positive_components_07d(phi)
    if len(comps0) != 1:
        raise RuntimeError(f"07D initial vesicle not one component seed={seed}, n={len(comps0)}")
    c1a = float(event["parent_c1_A"]); c1b = float(event["parent_c1_B"])
    integrated_J = 0.0; material = 0.0; max_S_ratio = 1.0
    min_tension = float("inf"); max_abs_phi = float(np.max(np.abs(phi)))
    fission = 0; fission_refresh: Optional[int] = None; daughter_count = 0
    daughter_ratio: Optional[float] = None; termination = "right_censored"
    for refresh in range(1, ASSAY_HORIZON_REFRESHES_07D + 1):
        t0 = float(refresh - 1)
        J = arm_throughput_07d(arm, c1a, c1b, t0)
        integrated_J += J
        material += (J / J_MAX_07D) / TAU_FISSION_REF_REFRESHES_07D
        target_ratio = 1.0 + DAUGHTER_PERIMETER_FRACTION_07D * material
        target_S = S0 * target_ratio
        for _ in range(PHASE_STEPS_PER_REFRESH_07D):
            S, tension, ma = phase_step_07d(phi, target_S, S0)
            max_S_ratio = max(max_S_ratio, S / S0)
            min_tension = min(min_tension, tension)
            max_abs_phi = max(max_abs_phi, ma)
        labels, comps = positive_components_07d(phi)
        if len(comps) >= 2:
            sizes = [int(np.count_nonzero(labels == lab)) for lab in comps]
            fission = 1; fission_refresh = refresh; daughter_count = len(sizes)
            daughter_ratio = float(min(sizes) / max(1, max(sizes)))
            termination = "spontaneous_topological_split"
            break
    Af, Sf = membrane_functional_07d(phi)
    final_integral = float(np.sum(phi))
    area_drift = abs(final_integral - initial_integral) / max(1.0, abs(initial_integral))
    if area_drift > AREA_REL_DRIFT_LIMIT_07D:
        raise RuntimeError(f"07D phase integral drift seed={seed} arm={arm}: {area_drift}")
    _, compsf = positive_components_07d(phi)
    final_target_ratio = 1.0 + DAUGHTER_PERIMETER_FRACTION_07D * material
    return BranchResult07D(
        seed=seed, ratio=PRIMARY_RATIO_07D, arm=arm, physical_seed=pseed,
        parent_c1_A=c1a, parent_c1_B=c1b, G_chem_07A=float(event["G_chem_07A"]),
        J_AB_07A=float(event["J_AB_07A"]), J_AA_07A=float(event["J_AA_07A"]), J_BB_07A=float(event["J_BB_07A"]),
        initial_state_hash=init_hash, initial_phase_area=A0, initial_membrane_functional=S0,
        fission_success=fission, fission_refresh=fission_refresh, daughter_count=daughter_count,
        daughter_area_ratio_minmax=daughter_ratio, final_component_count=len(compsf),
        final_phase_area=Af, final_membrane_functional=Sf, max_membrane_functional_ratio=max_S_ratio,
        integrated_J_refresh=integrated_J, integrated_normalized_Y=material,
        final_target_perimeter_ratio=final_target_ratio,
        minimum_effective_tension=min_tension if math.isfinite(min_tension) else float("nan"),
        maximum_abs_phi=max_abs_phi, relative_phase_area_drift=area_drift,
        termination_reason=termination,
    )


def synthetic_profile_assay_07d(seed: int, q: float) -> Dict[str, Any]:
    phi = initialize_fused_vesicle_07d(seed)
    A0, S0 = membrane_functional_07d(phi)
    initial_integral = float(np.sum(phi)); fission = 0; refresh_out = ""
    for refresh in range(1, ASSAY_HORIZON_REFRESHES_07D + 1):
        material = float(refresh) * float(q) / TAU_FISSION_REF_REFRESHES_07D
        target_S = S0 * (1.0 + DAUGHTER_PERIMETER_FRACTION_07D * material)
        for _ in range(PHASE_STEPS_PER_REFRESH_07D):
            phase_step_07d(phi, target_S, S0)
        _, comps = positive_components_07d(phi)
        if len(comps) >= 2:
            fission = 1; refresh_out = refresh; break
    rel = abs(float(np.sum(phi))-initial_integral)/max(1.0,abs(initial_integral))
    return {"seed":seed,"J_over_Jmax":q,"fission_success":fission,"fission_refresh":refresh_out,"relative_area_drift":rel,"initial_area":A0,"initial_membrane":S0}


def run_physical_preflight_07d() -> List[Dict[str, Any]]:
    rows = [synthetic_profile_assay_07d(s,q) for q in PREFLIGHT_PROFILES_07D for s in PREFLIGHT_SEEDS_07D]
    zero = [r for r in rows if float(r["J_over_Jmax"]) == 0.0]
    unit = [r for r in rows if float(r["J_over_Jmax"]) == 1.0]
    if any(int(r["fission_success"]) for r in zero):
        raise RuntimeError("07D physical preflight failed: zero-input fission")
    if not all(int(r["fission_success"]) for r in unit):
        raise RuntimeError("07D physical preflight failed: unit-input did not fission in all seeds")
    return rows


def exact_mcnemar_07d(a: Sequence[int], b: Sequence[int]) -> Tuple[int,int,int,float]:
    n10 = sum(1 for x,y in zip(a,b) if int(x)==1 and int(y)==0)
    n01 = sum(1 for x,y in zip(a,b) if int(x)==0 and int(y)==1)
    n = n10+n01
    if n == 0:
        return n10,n01,n,1.0
    p = float(_binomtest_07d(min(n10,n01), n=n, p=0.5, alternative="two-sided").pvalue)
    return n10,n01,n,p


def holm_adjust_07d(pvals: Sequence[float]) -> List[float]:
    n=len(pvals)
    if n==0: return []
    order=sorted(range(n), key=lambda i: float(pvals[i])); out=[1.0]*n; running=0.0
    for rank,idx in enumerate(order):
        val=min(1.0,(n-rank)*float(pvals[idx])); running=max(running,val); out[idx]=running
    return out


def first_event_07d(events: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    q=[]
    for r in events:
        try:
            if int(r.get("convergent",0)) != 1: continue
            if str(r.get("threshold","main")) != "main": continue
            if abs(float(r.get("ratio"))-PRIMARY_RATIO_07D) > 1e-9: continue
            c1a=float(r["parent_c1_diff_A"]); c1b=float(r["parent_c1_diff_B"])
        except Exception:
            continue
        q.append(r)
    if not q: return None
    q.sort(key=lambda r:(int(r.get("fusion_step",10**18)), float(r.get("fusion_time",float("inf"))), int(r.get("parent_track_A",0)), int(r.get("parent_track_B",0))))
    r=q[0]; c1a=float(r["parent_c1_diff_A"]); c1b=float(r["parent_c1_diff_B"])
    return {
        "07a_seed":int(r["seed"]), "ratio":PRIMARY_RATIO_07D,
        "fusion_step":int(r["fusion_step"]), "fusion_time":float(r["fusion_time"]),
        "parent_track_A":int(r["parent_track_A"]), "parent_track_B":int(r["parent_track_B"]),
        "parent_c1_A":c1a, "parent_c1_B":c1b,
        "G_chem_07A":float(r["G_chem_diff"]),
        "J_AB_07A":throughput_07d(0.5*(c1a+c1b)),
        "J_AA_07A":throughput_07d(c1a), "J_BB_07A":throughput_07d(c1b),
        "origin_class":str(r.get("origin_class","")),
        "events_available_seed_ratio":len(q),
    }


def seed_run_07d(seed: int, mode: str, work_root: str) -> Dict[str, Any]:
    if DISCOVERY_SEED_MIN_07D <= int(seed) <= DISCOVERY_SEED_MAX_07D:
        raise RuntimeError(f"07D refuses discovery seed {seed}")
    md = mode07a(mode)
    d = mode_defaults(md["source_mode"])
    cfg = SimConfig(
        seed=int(seed), hole_count=10, fixed_total_pore=False, reference_hole_count=10,
        confined_n=int(d["confined_n"]), open_n=int(d["open_n"]), pre_steps=int(d["pre_steps"]),
        gen_steps=int(d["gen_steps"]), open_steps=int(d["open_steps"]),
        sample_every=max(1, int(round(int(d["sample_every"]) * float(md["time_scale"])))),
        dt=0.035, b_threshold=0.16, t_threshold=0.018, min_component_voxels=12, min_lumen_voxels=20,
        source_regime="continuous", recenter=False, save_float16=False,
        release_min_components=8, release_check_every=20, release_min_gen_step=0,
        release_max_largest_fraction=0.0, release_fallback="end",
    )
    scale=float(md["time_scale"]); conditioning_duration=CONDITIONING_DURATION_FULL*scale
    transport_duration=TRANSPORT_DURATION_FULL*scale; tau_c=TAU_C_BASE
    tmp=Path(work_root)/f"seed_{int(seed):06d}_formation"; tmp.mkdir(parents=True,exist_ok=True)
    release_file=tmp/"fixed_release_state.npz"
    logger=ProgressLogger(tmp)
    try:
        # Fresh release states are generated for 07D. If --resume is used after a completed
        # formation, reuse only this seed's own freshly generated 07D state.
        if release_file.exists():
            dnp=read_npz(release_file)
            release_fields=Fields(R=dnp['R'].copy(),L=dnp['L'].copy(),H=dnp['H'].copy(),X=dnp['X'].copy(),M=dnp['M'].copy(),B=dnp['B'].copy(),T=dnp['T'].copy())
            release_step=int(np.asarray(dnp['release_step']).item())
            try: release_summary=json.loads(str(np.asarray(dnp['release_summary_json']).item()))
            except Exception: release_summary={}
            release_reused_07d_cache=1
        else:
            release_fields, release_step, release_summary = generate_fixed_release_state(cfg,tmp,logger)
            release_reused_07d_cache=0
    finally:
        logger.close()
    ev0, sm0 = transport_one_07a(release_fields,cfg,seed,None,False,tau_c,conditioning_duration,transport_duration)
    ev1, sm1 = transport_one_07a(release_fields,cfg,seed,PRIMARY_RATIO_07D,True,tau_c,conditioning_duration,transport_duration)
    for sm in (sm0,sm1):
        sm["release_step"]=release_step; sm["release_component_count"]=release_summary.get("component_count","")
    vals=[float(sm1.get("mean_G_diff",float("nan"))),float(sm0.get("mean_G_diff",float("nan"))),float(sm1.get("mean_G_undiff",float("nan"))),float(sm0.get("mean_G_undiff",float("nan")))]
    interaction=float("nan") if not all(np.isfinite(vals)) else (vals[0]-vals[1])-(vals[2]-vals[3])
    selected=first_event_07d(ev1)
    release_audit={
        "seed":int(seed),"release_step":int(release_step),"release_component_count":release_summary.get("component_count",""),
        "fresh_seed_block":1,"reused_own_07d_cache":release_reused_07d_cache,
        "release_state_sha256":_sha256(release_file) if release_file.exists() else "",
    }
    return {"seed":int(seed),"summaries":[sm0,sm1],"events_no_convergence":ev0,"events_convergence":ev1,
            "interaction":interaction,"selected_event":selected,"release_audit":release_audit}


def paired_physical_rows_07d(branch_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    group: Dict[int, Dict[str, Mapping[str, Any]]] = _defaultdict(dict)
    for r in branch_rows: group[int(r["seed"])][str(r["arm"])]=r
    out=[]
    for seed,arms in sorted(group.items()):
        if not all(a in arms for a in ARMS_07D): continue
        A=arms["AB_POOL"]; AA=arms["AA_EQUAL_MASS"]; BB=arms["BB_EQUAL_MASS"]; NP=arms["AB_NO_POOL"]; NY=arms["NO_Y"]
        hashes={str(arms[a]["initial_state_hash"]) for a in ARMS_07D}
        out.append({
            "seed":seed,"ratio":PRIMARY_RATIO_07D,"G_chem_07A":float(A["G_chem_07A"]),
            "AB_fission":int(A["fission_success"]),"AA_fission":int(AA["fission_success"]),"BB_fission":int(BB["fission_success"]),
            "NOPOOL_fission":int(NP["fission_success"]),"NOY_fission":int(NY["fission_success"]),
            "pair_specific_fission":int(int(A["fission_success"])==1 and int(AA["fission_success"])==0 and int(BB["fission_success"])==0),
            "pooling_specific_fission":int(int(A["fission_success"])==1 and int(NP["fission_success"])==0),
            "AB_delay":"" if A["fission_refresh"] in (None,"") else A["fission_refresh"],
            "AA_delay":"" if AA["fission_refresh"] in (None,"") else AA["fission_refresh"],
            "BB_delay":"" if BB["fission_refresh"] in (None,"") else BB["fission_refresh"],
            "NOPOOL_delay":"" if NP["fission_refresh"] in (None,"") else NP["fission_refresh"],
            "AB_integrated_Y":float(A["integrated_normalized_Y"]),"AA_integrated_Y":float(AA["integrated_normalized_Y"]),
            "BB_integrated_Y":float(BB["integrated_normalized_Y"]),"NOPOOL_integrated_Y":float(NP["integrated_normalized_Y"]),
            "branch_hash_consistent":int(len(hashes)==1),
        })
    return out


def physical_tests_07d(paired: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows=[]
    for comp,col in (("AA_EQUAL_MASS","AA_fission"),("BB_EQUAL_MASS","BB_fission"),("AB_NO_POOL","NOPOOL_fission")):
        a=[int(r["AB_fission"]) for r in paired]; b=[int(r[col]) for r in paired]
        n10,n01,nd,p=exact_mcnemar_07d(a,b)
        rows.append({"comparison":f"AB_POOL_vs_{comp}","n_seed_pairs":len(paired),"AB1_control0":n10,"AB0_control1":n01,"discordant":nd,"p_raw":p})
    adj=holm_adjust_07d([float(r["p_raw"]) for r in rows])
    for r,p in zip(rows,adj): r["p_holm"]=p; r["pass_holm_0p05"]=int(p<0.05)
    return rows


def self_test_07d() -> List[Tuple[str,bool]]:
    checks=[]
    checks.append(("method_lock_roundtrip",base64.b64decode(EMBEDDED_METHOD_LOCK_B64_07D).decode("utf-8")==METHOD_LOCK_JSON_07D))
    checks.append(("discovery_seeds_excluded", not (FULL_SEED_START_07D <= DISCOVERY_SEED_MAX_07D and FULL_SEED_START_07D+FULL_SEED_COUNT_07D-1 >= DISCOVERY_SEED_MIN_07D)))
    checks.append(("primary_ratio_one", PRIMARY_RATIO_07D==1.0))
    checks.append(("throughput_half",abs(throughput_07d(0.5)-0.25)<1e-15))
    checks.append(("no_y_zero",arm_throughput_07d("NO_Y",0.2,0.8,0)==0.0))
    checks.append(("ab_complementary_max",abs(arm_throughput_07d("AB_POOL",0.2,0.8,0)-0.25)<1e-15))
    phi=initialize_fused_vesicle_07d(PREFLIGHT_SEEDS_07D[0]); _,comps=positive_components_07d(phi)
    checks.append(("initial_one_component",len(comps)==1))
    _,S0=membrane_functional_07d(phi); integ0=float(np.sum(phi))
    for _ in range(5): phase_step_07d(phi,S0,S0)
    checks.append(("phase_integral_conserved",abs(float(np.sum(phi))-integ0)/max(1.0,abs(integ0))<AREA_REL_DRIFT_LIMIT_07D))
    checks.append(("geometry_factor",abs(DAUGHTER_PERIMETER_FRACTION_07D-(math.sqrt(2)-1))<1e-15))
    checks.append(("phase_has_no_arm_argument","arm" not in phase_step_07d.__code__.co_varnames))
    checks.append(("single_ratio_only",PRIMARY_RATIO_07D in RATIOS_07A))
    sp,st,_=self_test_07a(verbose=False); checks.append(("inherited_07a_self_test",sp==st))
    return checks


def parse_args_07d() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="07D independent end-to-end two-parent fission confirmation")
    p.add_argument("--mode",choices=["smoke","quick","full"],default="smoke")
    p.add_argument("--outdir",type=Path,default=None)
    p.add_argument("--workers",type=int,default=max(1,min(8,(os.cpu_count() or 2)-1)))
    p.add_argument("--resume",action="store_true")
    p.add_argument("--seed-start",type=int,default=None,help="Non-full diagnostic modes only. Full mode is locked to 71000.")
    p.add_argument("--seed-count",type=int,default=None,help="Non-full diagnostic modes only. Full mode is locked to 32.")
    p.add_argument("--self-test",action="store_true")
    return p.parse_args()


def main_07d() -> int:
    args=parse_args_07d()
    if args.mode=="full":
        if args.seed_start not in (None,FULL_SEED_START_07D) or args.seed_count not in (None,FULL_SEED_COUNT_07D):
            raise RuntimeError("07D full mode is locked to seeds 71000-71031")
        seed_start=FULL_SEED_START_07D; seed_count=FULL_SEED_COUNT_07D; scientific=True
    else:
        defaults={"smoke":(72000,1),"quick":(72000,8)}
        ds,dc=defaults[args.mode]; seed_start=ds if args.seed_start is None else int(args.seed_start); seed_count=dc if args.seed_count is None else int(args.seed_count); scientific=False
    seeds=[seed_start+i for i in range(seed_count)]
    if any(DISCOVERY_SEED_MIN_07D<=s<=DISCOVERY_SEED_MAX_07D for s in seeds):
        raise RuntimeError("07D refuses any discovery seed 70000-70031")
    outdir=(args.outdir or (Path.home()/"Desktop"/f"INDEPENDENT_END_TO_END_TWO_PARENT_FISSION_V07D_{args.mode.upper()}" )).expanduser().resolve()
    outdir.mkdir(parents=True,exist_ok=True); log=ProgressLogger(outdir); t0=time.time()
    try:
        checks=self_test_07d(); passed=sum(int(ok) for _,ok in checks)
        for name,ok in checks: log.log(f"self-test {name}: {'PASS' if ok else 'FAIL'}")
        if passed!=len(checks): raise RuntimeError(f"07D self-test failed {passed}/{len(checks)}")
        if args.self_test:
            print(f"07D SELF-TEST {passed}/{len(checks)} PASS",flush=True); return 0
        preflight=run_physical_preflight_07d(); write_csv(outdir/"00_PHYSICAL_PREFLIGHT_AUDIT.csv",preflight)
        work_root=outdir/"_work"; work_root.mkdir(exist_ok=True)
        cache_dir=outdir/"_seed_cache"; cache_dir.mkdir(exist_ok=True)
        results=[]; pending=[]
        for seed in seeds:
            cp=cache_dir/f"seed_{seed:06d}.json"
            if args.resume and cp.exists():
                results.append(json.loads(cp.read_text(encoding="utf-8"))); log.log(f"resume seed {seed} loaded")
            else: pending.append(seed)
        workers=max(1,min(int(args.workers),max(1,len(pending))))
        log.log(f"07D start mode={args.mode} scientific={scientific} seeds={len(seeds)} pending={len(pending)} workers={workers} ratio={PRIMARY_RATIO_07D}")
        def save_res(res: Dict[str,Any]) -> None:
            cp=cache_dir/f"seed_{int(res['seed']):06d}.json"; cp.write_text(json.dumps(json_safe(res),ensure_ascii=False),encoding="utf-8")
        if pending:
            if workers==1:
                for i,seed in enumerate(pending,1):
                    log.log(f"seed {seed} start {i}/{len(pending)}"); res=seed_run_07d(seed,args.mode,str(work_root)); results.append(res); save_res(res); log.log(f"seed {seed} done")
            else:
                ctx=_mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
                    futs={ex.submit(seed_run_07d,seed,args.mode,str(work_root)):seed for seed in pending}; done=0
                    for fut in as_completed(futs):
                        seed=futs[fut]; res=fut.result(); results.append(res); save_res(res); done+=1; log.log(f"seed {seed} done {done}/{len(pending)}")
        results.sort(key=lambda r:int(r["seed"]))
        release_audit=[r["release_audit"] for r in results]; write_csv(outdir/"01_FRESH_RELEASE_AUDIT.csv",release_audit)
        condition_rows=[x for r in results for x in r["summaries"]]; write_csv(outdir/"02_07A_SEED_CONDITION_SUMMARY.csv",condition_rows)
        all_events=[x for r in results for x in (r["events_no_convergence"]+r["events_convergence"])]; write_csv(outdir/"03_07A_FUSION_EVENT_ASSAYS.csv",all_events)
        selected=[r["selected_event"] for r in results if r.get("selected_event") is not None]; write_csv(outdir/"04_FIRST_FUSION_SELECTION.csv",selected)
        interactions=[]
        for r in results:
            interactions.append({"seed":int(r["seed"]),"ratio":PRIMARY_RATIO_07D,"interaction":r["interaction"],"first_fusion_available":int(r.get("selected_event") is not None)})
        write_csv(outdir/"05_07A_CHEMICAL_INTERACTIONS.csv",interactions)
        vals=[float(r["interaction"]) for r in interactions if np.isfinite(float(r["interaction"]))]
        W,pchem,nchem=wilcoxon_exact_07a(vals); ci_lo,ci_hi=bootstrap_ci_07a(vals,seed=70707,nboot=10000)
        chem_test=[{"ratio":PRIMARY_RATIO_07D,"n_complete_seeds":nchem,"mean_interaction":float(np.mean(vals)) if vals else float('nan'),"sd_interaction":float(np.std(vals,ddof=1)) if len(vals)>1 else float('nan'),"median_interaction":float(np.median(vals)) if vals else float('nan'),"bootstrap95_lo":ci_lo,"bootstrap95_hi":ci_hi,"positive":sum(v>0 for v in vals),"zero":sum(abs(v)<=1e-15 for v in vals),"negative":sum(v<0 for v in vals),"wilcoxon_W":W,"p_two_sided_exact":pchem,"pass_p_0p05":int(pchem<0.05)}]
        write_csv(outdir/"06_07A_CHEMICAL_TEST.csv",chem_test)
        branches=[]
        jobs=[(e,a) for e in selected for a in ARMS_07D]
        bw=max(1,min(int(args.workers),max(1,len(jobs))))
        if bw==1:
            for i,(e,a) in enumerate(jobs,1):
                branches.append(asdict(assay_branch_07d(e,a)))
                if i%max(1,len(jobs)//10)==0: log.log(f"physical branches {i}/{len(jobs)}")
        else:
            ctx=_mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=bw,mp_context=ctx) as ex:
                futs={ex.submit(assay_branch_07d,e,a):(int(e['07a_seed']),a) for e,a in jobs}; done=0
                for fut in as_completed(futs):
                    key=futs[fut]; branches.append(asdict(fut.result())); done+=1
                    if done%max(1,len(jobs)//10)==0: log.log(f"physical branches {done}/{len(jobs)}")
        branches.sort(key=lambda r:(int(r["seed"]),ARMS_07D.index(str(r["arm"]))))
        write_csv(outdir/"07_BRANCH_RESULTS.csv",branches)
        paired=paired_physical_rows_07d(branches); write_csv(outdir/"08_PAIRED_PHYSICAL_OUTCOMES.csv",paired)
        ptests=physical_tests_07d(paired); write_csv(outdir/"09_PRIMARY_MCNEMAR.csv",ptests)
        mech=[]; gg=_defaultdict(list)
        for r in branches: gg[int(r["seed"])].append(r)
        for seed,rows in sorted(gg.items()):
            mech.append({"seed":seed,"initial_hash_count":len({str(r['initial_state_hash']) for r in rows}),"initial_area_range":max(float(r['initial_phase_area']) for r in rows)-min(float(r['initial_phase_area']) for r in rows),"initial_membrane_range":max(float(r['initial_membrane_functional']) for r in rows)-min(float(r['initial_membrane_functional']) for r in rows),"NO_Y_integrated_Y":next(float(r['integrated_normalized_Y']) for r in rows if r['arm']=='NO_Y'),"max_relative_area_drift":max(float(r['relative_phase_area_drift']) for r in rows)})
        write_csv(outdir/"10_MECHANICAL_AUDIT.csv",mech)
        physical_pass=bool(ptests) and all(float(r["p_holm"])<0.05 for r in ptests)
        chemical_pass=bool(chem_test) and float(chem_test[0]["p_two_sided_exact"])<0.05
        overall_pass=chemical_pass and physical_pass
        summary=[{"n_fresh_seeds":len(seeds),"n_chemical_complete":nchem,"n_first_fusions":len(selected),"n_physical_pairs":len(paired),"AB_fissions":sum(int(r['AB_fission']) for r in paired),"AA_fissions":sum(int(r['AA_fission']) for r in paired),"BB_fissions":sum(int(r['BB_fission']) for r in paired),"NOPOOL_fissions":sum(int(r['NOPOOL_fission']) for r in paired),"NOY_fissions":sum(int(r['NOY_fission']) for r in paired),"chemical_gate_pass":int(chemical_pass),"physical_family_pass":int(physical_pass),"end_to_end_confirmation_pass":int(overall_pass)}]
        write_csv(outdir/"11_END_TO_END_SUMMARY.csv",summary)
        method_audit=[{"07D_method_lock_path":"<embedded:07D_METHOD_LOCK.json>","07D_method_lock_sha256":METHOD_LOCK_SHA256_07D,"07C_discovery_method_lock_sha256":DISCOVERY_07C_METHOD_LOCK_SHA256,"07A_method_lock_sha256":EXPECTED_07A_LOCK_SHA256,"07A_transport_correction_lock_sha256":EXPECTED_07A_V2_CORRECTION_LOCK_SHA256,"07A_source_core_name":SOURCE_CORE_NAME,"07A_source_core_sha256":SOURCE_CORE_SHA256,"script_version":SCRIPT_VERSION_07D,"script_sha256":_sha256(Path(__file__).resolve()),"mode":args.mode,"scientific_inference_allowed":int(scientific),"seed_start":seed_start,"seed_count":seed_count,"primary_ratio":PRIMARY_RATIO_07D,"old_release_zip_used":0,"self_test":f"{passed}/{len(checks)} PASS"}]
        write_csv(outdir/"00_METHOD_LOCK_AUDIT.csv",method_audit)
        lines=["# 07D Independent End-to-End Two-Parent Fission Confirmation","",f"- script_version: {SCRIPT_VERSION_07D}",f"- mode: {args.mode}",f"- scientific_inference_allowed: {'TRUE' if scientific else 'FALSE'}",f"- 07D_method_lock_sha256: `{METHOD_LOCK_SHA256_07D}`",f"- discovery_07C_method_lock_sha256: `{DISCOVERY_07C_METHOD_LOCK_SHA256}`",f"- fresh_seed_block: {seed_start}-{seed_start+seed_count-1}",f"- primary_ratio_tau_conv_over_tau_C: {PRIMARY_RATIO_07D}",f"- elapsed_seconds: {time.time()-t0:.3f}","","## Locked interpretation boundary","","07D is an independent confirmatory replication on fresh seeds. It runs only ratio=1.0 and regenerates every physical release state. No discovery seed or old release ZIP is reused.","No division trigger, J threshold, division axis, partner sensing, favorable-event filtering, fitness, selection, mutation, or genetic mechanism is present.","","## Chemical confirmation","",f"- complete fresh seeds: {nchem}",f"- mean interaction: {chem_test[0]['mean_interaction']}",f"- bootstrap 95% CI: [{ci_lo},{ci_hi}]",f"- + / 0 / -: {chem_test[0]['positive']}/{chem_test[0]['zero']}/{chem_test[0]['negative']}",f"- exact two-sided Wilcoxon p: {pchem}",f"- chemical gate: {'PASS' if chemical_pass else 'FAIL'}","","## Natural fusion availability","",f"- first eligible fresh two-parent fusion events: {len(selected)}/{len(seeds)} seeds","Missing fusion events are reported as missing and are not imputed.","","## Physical fission outcomes","",f"- complete paired physical seeds: {len(paired)}",f"- AB_POOL: {summary[0]['AB_fissions']}",f"- AA_EQUAL_MASS: {summary[0]['AA_fissions']}",f"- BB_EQUAL_MASS: {summary[0]['BB_fissions']}",f"- AB_NO_POOL: {summary[0]['NOPOOL_fissions']}",f"- NO_Y: {summary[0]['NOY_fissions']}",""]
        for r in ptests: lines.append(f"- {r['comparison']}: AB1/control0={r['AB1_control0']}; AB0/control1={r['AB0_control1']}; p_raw={r['p_raw']}; p_Holm={r['p_holm']}")
        lines += ["",f"- physical family: {'PASS' if physical_pass else 'FAIL'}","","## End-to-end confirmatory decision","",f"**{'PASS' if overall_pass else 'FAIL'}**","","PASS requires the pre-specified chemical gate and all three Holm-corrected physical comparisons to pass. NO_Y is mechanistic evidence but is not an additional success gate.",""]
        (outdir/"12_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
        log.log(f"07D complete overall={'PASS' if overall_pass else 'FAIL'} elapsed={time.time()-t0:.1f}s outdir={outdir}")
        return 0
    except Exception as exc:
        log.log(f"07D ERROR {type(exc).__name__}: {exc}"); log.log(traceback.format_exc()); (outdir/"ERROR.txt").write_text(traceback.format_exc(),encoding="utf-8"); return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main_07d())
