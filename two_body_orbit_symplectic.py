"""
two_body_orbit_symplectic.py
============================

Modified version of two-body-orbit.py.

This script keeps the original educational panels and adds a comparison of
three orbit integrators for the ISS-like two-body problem:

1. DOP853 from scipy.solve_ivp:
   high-order adaptive Runge--Kutta, but not symplectic.

2. Velocity Verlet:
   second-order symplectic integrator for separable Hamiltonians.

3. Four-stage Gauss--Legendre implicit Runge--Kutta:
   symplectic Runge--Kutta method of order 8.

Important correction:
There is no standard Gauss--Legendre "7th-order" symplectic RK method.
Gauss--Legendre RK methods have order 2s with s stages. The closest natural
choice above seventh order is therefore s = 4, order 8.

Dependencies: numpy, scipy, matplotlib

Usage:
  python two_body_orbit_symplectic.py

Output:
  /mnt/data/two-body-orbit-symplectic.png, when run in this environment.
"""

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Constants ─────────────────────────────────────────────────────────────────

MU_ND = 1.0                   # non-dimensional gravitational parameter
MU_SI = 3.986004418e5         # Earth μ, km^3/s^2

BG   = '#0d1117'
FG   = '#c9d1d9'
DIM  = '#8b949e'
COLS = ['#4363d8', '#3cb44b', '#e6194b', '#f58231']

# ── Panel 1: Conic orbits ─────────────────────────────────────────────────────

def conic_xy(e, p=1.0, f_range=None):
    """Return (x, y) for a conic section r = p/(1+e*cos f)."""
    if e < 1:
        f = np.linspace(-np.pi, np.pi, 600)
    elif abs(e - 1) < 1e-6:
        f = np.linspace(-2.8, 2.8, 600)
    else:
        f_max = np.arccos(-1 / e) - 0.05
        f = np.linspace(-f_max, f_max, 600)
    if f_range is not None:
        f_start, f_end = f_range
        f = np.linspace(f_start, f_end, 600)
    r = p / (1 + e * np.cos(f))
    x = r * np.cos(f)
    y = r * np.sin(f)
    return x, y

# ── Panel 2: Vis-viva curves ──────────────────────────────────────────────────

def vis_viva(r, a, mu=MU_ND):
    """Return orbital speed from the vis-viva equation."""
    inside = mu * (2 / r - 1 / a)
    return np.sqrt(np.maximum(inside, 0.0))

# ── Two-body dynamics ─────────────────────────────────────────────────────────

def two_body_rhs(t, y, mu):
    r_vec = y[:3]
    v_vec = y[3:]
    r = np.linalg.norm(r_vec)
    a_vec = -mu / r**3 * r_vec
    return np.concatenate([v_vec, a_vec])


def acceleration(r_vec, mu):
    r = np.linalg.norm(r_vec)
    return -mu * r_vec / r**3


def invariants(y_hist, mu):
    """Return specific energy and angular-momentum magnitude for y(t)."""
    pos = y_hist[:, :3]
    vel = y_hist[:, 3:]
    r = np.linalg.norm(pos, axis=1)
    v = np.linalg.norm(vel, axis=1)
    E = 0.5 * v**2 - mu / r
    h = np.linalg.norm(np.cross(pos, vel), axis=1)
    return E, h

# ── Symplectic method 1: velocity Verlet ──────────────────────────────────────

def velocity_verlet(y0, t_grid, mu):
    """Second-order symplectic integrator for q'' = a(q)."""
    y = np.zeros((len(t_grid), len(y0)))
    y[0] = y0
    q = y0[:3].copy()
    v = y0[3:].copy()

    for n in range(len(t_grid) - 1):
        h = t_grid[n + 1] - t_grid[n]
        a_n = acceleration(q, mu)
        v_half = v + 0.5 * h * a_n
        q_new = q + h * v_half
        a_new = acceleration(q_new, mu)
        v_new = v_half + 0.5 * h * a_new
        q, v = q_new, v_new
        y[n + 1, :3] = q
        y[n + 1, 3:] = v

    return y

# ── Symplectic method 2: Gauss--Legendre implicit RK ──────────────────────────

def gauss_legendre_tableau(s):
    """Return A, b, c for the s-stage Gauss--Legendre collocation RK method."""
    # Legendre nodes x_i on [-1,1], shifted to c_i on [0,1].
    x, w = np.polynomial.legendre.leggauss(s)
    c = 0.5 * (x + 1.0)
    b = 0.5 * w

    # Lagrange basis polynomials l_j(tau) through nodes c_j.
    A = np.zeros((s, s))
    for j in range(s):
        coeff = np.poly1d([1.0])
        denom = 1.0
        for m in range(s):
            if m != j:
                coeff *= np.poly1d([1.0, -c[m]])
                denom *= (c[j] - c[m])
        lj = coeff / denom
        int_lj = np.polyint(lj)
        for i in range(s):
            A[i, j] = int_lj(c[i]) - int_lj(0.0)
    return A, b, c


def gauss_legendre_irk(y0, t_grid, mu, s=4, tol=1e-12):
    """
    s-stage Gauss--Legendre implicit Runge--Kutta method.

    For s=4, the method is symplectic and has order 8.
    """
    A, b, c = gauss_legendre_tableau(s)
    dim = len(y0)
    y = np.zeros((len(t_grid), dim))
    y[0] = y0

    y_n = y0.copy()
    K_prev = np.tile(two_body_rhs(t_grid[0], y0, mu), (s, 1))

    for n in range(len(t_grid) - 1):
        t_n = t_grid[n]
        h = t_grid[n + 1] - t_grid[n]

        def residual(K_flat):
            K = K_flat.reshape(s, dim)
            R = np.empty_like(K)
            for i in range(s):
                stage_state = y_n + h * np.sum(A[i, :, None] * K, axis=0)
                R[i] = K[i] - two_body_rhs(t_n + c[i] * h, stage_state, mu)
            return R.ravel()

        # Use previous step's stages as the initial guess. This is much better
        # than starting every nonlinear solve from zero.
        sol = root(residual, K_prev.ravel(), method="hybr", tol=tol)
        if not sol.success:
            # Fall back to a simpler initial guess based on f(y_n).
            K_guess = np.tile(two_body_rhs(t_n, y_n, mu), (s, 1))
            sol = root(residual, K_guess.ravel(), method="hybr", tol=tol)
        if not sol.success:
            raise RuntimeError(f"Gauss--Legendre nonlinear solve failed at step {n}: {sol.message}")

        K = sol.x.reshape(s, dim)
        y_np1 = y_n + h * np.sum(b[:, None] * K, axis=0)
        y[n + 1] = y_np1
        y_n = y_np1
        K_prev = K

    return y

# ── Orbit setup ───────────────────────────────────────────────────────────────

R_E  = 6371.0                 # Earth radius, km
alt  = 407.0                  # ISS altitude, km
a_iss = R_E + alt             # semi-major axis, km
e_iss = 0.001
mu    = MU_SI

# Initial conditions at pericenter, true anomaly f = 0.
r_p  = a_iss * (1 - e_iss)
v_p  = np.sqrt(mu * (1 + e_iss) / (a_iss * (1 - e_iss)))
pos0 = np.array([r_p, 0.0, 0.0])
vel0 = np.array([0.0, v_p, 0.0])
y0   = np.concatenate([pos0, vel0])

T_orb = 2 * np.pi * np.sqrt(a_iss**3 / mu)

# A longer integration makes the distinction between geometric and non-geometric
# behavior clearer. Increase n_orbits if you want a more demanding test.
n_orbits = 20
steps_per_orbit = 80
n_pts = n_orbits * steps_per_orbit + 1
t_eval = np.linspace(0.0, n_orbits * T_orb, n_pts)

# ── Integrate ─────────────────────────────────────────────────────────────────

sol_dop = solve_ivp(
    two_body_rhs,
    [t_eval[0], t_eval[-1]],
    y0,
    args=(mu,),
    t_eval=t_eval,
    method='DOP853',
    rtol=1e-10,
    atol=1e-12,
)
if not sol_dop.success:
    raise RuntimeError(sol_dop.message)

y_dop = sol_dop.y.T

# Velocity Verlet is included to show a simple second-order symplectic baseline.
y_vv = velocity_verlet(y0, t_eval, mu)

# Four-stage Gauss--Legendre implicit RK: symplectic, order 8.
y_gl8 = gauss_legendre_irk(y0, t_eval, mu, s=4, tol=1e-12)

E_dop, h_dop = invariants(y_dop, mu)
E_vv,  h_vv  = invariants(y_vv,  mu)
E_gl8, h_gl8 = invariants(y_gl8, mu)

def rel_err(arr):
    return np.abs(arr - arr[0]) / max(abs(arr[0]), 1e-300)

dE_dop, dh_dop = rel_err(E_dop), rel_err(h_dop)
dE_vv,  dh_vv  = rel_err(E_vv),  rel_err(h_vv)
dE_gl8, dh_gl8 = rel_err(E_gl8), rel_err(h_gl8)

# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(17, 13))
fig.patch.set_facecolor(BG)
fig.suptitle('Two-Body Problem — Conservation and Symplectic Integration',
             color=FG, fontsize=14, fontweight='bold', y=0.995)

gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

for ax in (ax1, ax2, ax3, ax4):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')

# --- Panel 1: Conic sections ---

ax1.set_title('Conic sections: r = p/(1 + e cos f)', color=FG, fontsize=10, pad=6)
ax1.set_aspect('equal')
ax1.set_xlim(-3.2, 2.2)
ax1.set_ylim(-2.2, 2.2)
ax1.set_xlabel('x', color=DIM, fontsize=9)
ax1.set_ylabel('y', color=DIM, fontsize=9)

orbit_cases = [
    (0.0,  'circle (e = 0)',       COLS[0]),
    (0.6,  'ellipse (e = 0.6)',    COLS[1]),
    (1.0,  'parabola (e = 1)',     COLS[2]),
    (1.6,  'hyperbola (e = 1.6)',  COLS[3]),
]
for e_val, label, col in orbit_cases:
    x, y = conic_xy(e_val)
    ax1.plot(x, y, '-', color=col, lw=1.5, label=label)

ax1.plot(0, 0, 'o', color='yellow', ms=10, zorder=6, label='focus')
ax1.axhline(0, color=DIM, lw=0.4, alpha=0.4)
ax1.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, loc='upper right')

# --- Panel 2: Vis-viva ---

ax2.set_title('Vis-viva: v² = μ(2/r − 1/a)', color=FG, fontsize=10, pad=6)
ax2.set_xlabel('r / p', color=DIM, fontsize=9)
ax2.set_ylabel('v  [non-dim.]', color=DIM, fontsize=9)

r_arr_nd = np.linspace(0.2, 5.0, 400)
for a_val, col in zip([0.8, 1.5, 3.0], ['#4363d8', '#3cb44b', '#e6194b']):
    ax2.plot(r_arr_nd, vis_viva(r_arr_nd, a_val), '-', color=col, lw=1.5, label=f'a = {a_val}')

v_esc = np.sqrt(2 * MU_ND / r_arr_nd)
ax2.plot(r_arr_nd, v_esc, '--', color='#f58231', lw=1.2, label='escape')
ax2.set_ylim(0, 4)
ax2.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor=FG)

# --- Panel 3: Orbit ---

ax3.set_title(f'ISS-like orbit over {n_orbits} periods', color=FG, fontsize=10, pad=6)
ax3.set_aspect('equal')
ax3.set_xlabel('x [km]', color=DIM, fontsize=9)
ax3.set_ylabel('y [km]', color=DIM, fontsize=9)
ax3.plot(y_gl8[:, 0], y_gl8[:, 1], '-', color=COLS[0], lw=1.2, alpha=0.9, label='GL8 symplectic RK')
ax3.plot(y_dop[:, 0], y_dop[:, 1], '--', color=COLS[2], lw=0.9, alpha=0.6, label='DOP853')

earth = Circle((0, 0), R_E, color='#1a6b3e', alpha=0.7, zorder=4)
ax3.add_patch(earth)
ax3.plot(0, 0, '+', color='white', ms=8, zorder=5)
ax3.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor=FG)

# --- Panel 4: Conservation comparison ---

ax4.set_title('Relative conservation error', color=FG, fontsize=10, pad=6)
ax4.set_xlabel('orbit number', color=DIM, fontsize=9)
ax4.set_ylabel('relative error', color=DIM, fontsize=9)
ax4.set_yscale('log')

t_orbits = t_eval / T_orb
eps = 1e-18
ax4.plot(t_orbits, np.maximum(dE_dop, eps), color='#e6194b', lw=1.2, label='DOP853 |ΔE/E0|')
ax4.plot(t_orbits, np.maximum(dE_vv,  eps), color='#f58231', lw=1.0, label='Verlet |ΔE/E0|')
ax4.plot(t_orbits, np.maximum(dE_gl8, eps), color='#3cb44b', lw=1.4, label='GL8 sympl. RK |ΔE/E0|')
ax4.plot(t_orbits, np.maximum(dh_gl8, eps), color='#4363d8', lw=1.0, ls='--', label='GL8 |Δh/h0|')
ax4.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor=FG)

out_path = os.path.join('/mnt/data', 'two-body-orbit-symplectic.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)

print(f"Saved {out_path}")
print(f"Orbital period: {T_orb/60:.2f} min ({T_orb:.1f} s)")
print("Maximum relative errors over the integration:")
print(f"  DOP853:  max |ΔE/E0| = {dE_dop.max():.3e}, max |Δh/h0| = {dh_dop.max():.3e}")
print(f"  Verlet:  max |ΔE/E0| = {dE_vv.max():.3e}, max |Δh/h0| = {dh_vv.max():.3e}")
print(f"  GL8 RK:  max |ΔE/E0| = {dE_gl8.max():.3e}, max |Δh/h0| = {dh_gl8.max():.3e}")
