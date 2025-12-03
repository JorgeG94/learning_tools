#!/usr/bin/env python3
"""
swe_sim.py

2D Shallow Water Equations (SWE) demo (no plotting).

- Periodic in x and y
- Prognostic variables: u(x,y), v(x,y), h(x,y)
- Time stepping: SSP RK3
- Advection: 1st-order upwind
- Pressure gradient: g * ∇h
- Coriolis: f-plane or beta-plane
- Viscosity: Laplacian
- Output: CSV files of eta = h - H0 every output_every steps
"""

import os
import numpy as np


# ==========================
# CONFIG
# ==========================

class SWEConfig:
    def __init__(self):
        # Grid
        self.nx = 128
        self.ny = 128
        self.Lx = 1.0e6   # domain size in x (m)
        self.Ly = 1.0e6   # domain size in y (m)

        # Time
        self.dt = 5.0           # time step (s)
        self.nt = 12000           # number of steps
        self.print_every = 20
        self.output_every = 100

        # Physical constants
        self.g = 9.81           # gravity (m/s^2)
        self.H0 = 100.0         # mean depth (m)
        self.f0 = 1.0e-4        # base Coriolis (1/s)
        self.beta = 0.0         # beta parameter (1/(m s)), 0 => f-plane

        # Dissipation
        self.nu = 50.0          # viscosity (m^2/s)

        # Output directory
        self.output_dir = "swe_output"


# ==========================
# GRID / STATE
# ==========================

def initialize_grid(cfg):
    dx = cfg.Lx / cfg.nx
    dy = cfg.Ly / cfg.ny

    x = (np.arange(cfg.nx) + 0.5) * dx
    y = (np.arange(cfg.ny) + 0.5) * dy

    return x, y, dx, dy


def initialize_state(cfg, x, y):
    """
    u, v start at rest; h has a Gaussian bump.
    """
    X, Y = np.meshgrid(x, y)

    h = cfg.H0 + 5.0 * np.exp(
        -(((X - cfg.Lx/2.0)**2 + (Y - cfg.Ly/2.0)**2) /
          (2.0 * (cfg.Lx/10.0)**2))
    )

    u = np.zeros_like(h)
    v = np.zeros_like(h)

    return u, v, h


# ==========================
# NUMERICAL OPS
# ==========================

def ddx(f, dx):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)


def ddy(f, dy):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)


def laplacian(f, dx, dy):
    return (
        (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / dx**2 +
        (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / dy**2
    )


def advective_derivative(f, u, v, dx, dy):
    """
    First-order upwind for (u * df/dx + v * df/dy).
    Periodic in x, y.
    """
    f_x_plus = np.roll(f, -1, axis=1)
    f_x_minus = np.roll(f, 1, axis=1)
    df_dx = np.where(u > 0.0, (f - f_x_minus) / dx,
                              (f_x_plus - f) / dx)

    f_y_plus = np.roll(f, -1, axis=0)
    f_y_minus = np.roll(f, 1, axis=0)
    df_dy = np.where(v > 0.0, (f - f_y_minus) / dy,
                              (f_y_plus - f) / dy)

    return u * df_dx + v * df_dy


# ==========================
# DYNAMICS
# ==========================

def compute_coriolis(cfg, y_coords):
    if cfg.beta != 0.0:
        return cfg.f0 + cfg.beta * (y_coords[:, None] - cfg.Ly / 2.0)
    else:
        return cfg.f0 * np.ones((len(y_coords), 1))


def compute_tendencies(u, v, h, cfg, dx, dy, y_coords):
    """
    Compute du/dt, dv/dt, dh/dt for SWE.
    """
    f = compute_coriolis(cfg, y_coords)

    # Flux-form continuity
    hu = h * u
    hv = h * v
    d_hu_dx = (np.roll(hu, -1, axis=1) - np.roll(hu, 1, axis=1)) / (2.0 * dx)
    d_hv_dy = (np.roll(hv, -1, axis=0) - np.roll(hv, 1, axis=0)) / (2.0 * dy)
    dh_dt = -(d_hu_dx + d_hv_dy)

    # Momentum advection
    adv_u = advective_derivative(u, u, v, dx, dy)
    adv_v = advective_derivative(v, u, v, dx, dy)

    # Pressure gradient
    dp_dx = cfg.g * ddx(h, dx)
    dp_dy = cfg.g * ddy(h, dy)

    # Viscosity
    visc_u = cfg.nu * laplacian(u, dx, dy)
    visc_v = cfg.nu * laplacian(v, dx, dy)

    du_dt = -adv_u - dp_dx + f * v + visc_u
    dv_dt = -adv_v - dp_dy - f * u + visc_v

    return du_dt, dv_dt, dh_dt


# ==========================
# TIME INTEGRATION (SSP RK3)
# ==========================

def rk3_step(u, v, h, cfg, dx, dy, y_coords):
    dt = cfg.dt

    # Stage 1
    du1, dv1, dh1 = compute_tendencies(u, v, h, cfg, dx, dy, y_coords)
    u1 = u + dt * du1
    v1 = v + dt * dv1
    h1 = h + dt * dh1

    # Stage 2
    du2, dv2, dh2 = compute_tendencies(u1, v1, h1, cfg, dx, dy, y_coords)
    u2 = 0.75 * u + 0.25 * (u1 + dt * du2)
    v2 = 0.75 * v + 0.25 * (v1 + dt * dv2)
    h2 = 0.75 * h + 0.25 * (h1 + dt * dh2)

    # Stage 3
    du3, dv3, dh3 = compute_tendencies(u2, v2, h2, cfg, dx, dy, y_coords)
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * du3)
    v_new = (1.0/3.0) * v + (2.0/3.0) * (v2 + dt * dv3)
    h_new = (1.0/3.0) * h + (2.0/3.0) * (h2 + dt * dh3)

    return u_new, v_new, h_new


# ==========================
# DIAGNOSTICS / I/O
# ==========================

def compute_cfl(u, v, cfg, dx, dy):
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    c = np.sqrt(cfg.g * cfg.H0)

    adv_cfl = cfg.dt * (umax / dx + vmax / dy)
    wave_cfl = cfg.dt * c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2)
    return adv_cfl, wave_cfl


def ensure_output_dir(cfg):
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)


def write_eta_csv(eta, cfg, step):
    fname = os.path.join(cfg.output_dir, f"eta_step_{step:05d}.csv")
    np.savetxt(fname, eta, delimiter=",")


# ==========================
# MAIN
# ==========================

def main():
    cfg = SWEConfig()
    ensure_output_dir(cfg)

    x, y, dx, dy = initialize_grid(cfg)
    u, v, h = initialize_state(cfg, x, y)
    y_coords = y

    adv_cfl, wave_cfl = compute_cfl(u, v, cfg, dx, dy)
    print(f"Initial advective CFL = {adv_cfl:.3f}, wave CFL = {wave_cfl:.3f}")

    for n in range(1, cfg.nt + 1):
        u, v, h = rk3_step(u, v, h, cfg, dx, dy, y_coords)

        if n % cfg.print_every == 0:
            adv_cfl, wave_cfl = compute_cfl(u, v, cfg, dx, dy)
            eta = h - cfg.H0
            max_eta = np.max(eta)
            min_eta = np.min(eta)
            t_hours = n * cfg.dt / 3600.0
            print(
                f"step {n:5d}, t = {t_hours:7.3f} h, "
                f"max(eta) = {max_eta:7.3f} m, min(eta) = {min_eta:7.3f} m, "
                f"adv CFL = {adv_cfl:6.3f}, wave CFL = {wave_cfl:6.3f}"
            )

        if n % cfg.output_every == 0:
            eta = h - cfg.H0
            write_eta_csv(eta, cfg, n)

    # Final dump
    eta = h - cfg.H0
    write_eta_csv(eta, cfg, cfg.nt)
    print("Run finished, CSV files written to:", cfg.output_dir)


if __name__ == "__main__":
    main()

