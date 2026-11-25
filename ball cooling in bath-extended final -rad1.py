# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 09:18:20 2026

@author: VK
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER INPUT
# ============================================================
raw = input("Enter target ball temperature [K] (press Enter for 500): ").strip()
T_target = float(raw) if raw else 500.0

t_end = 2000.0   # [s]
dt = 0.5         # [s]

# ============================================================
# GIVEN DATA
# ============================================================
d_ball = 0.010            # [m]
T_ball0 = 1000.0          # [K]
T_oil0 = 300.0            # [K]

m_steel = 200.0           # [kg]
m_oil = 500.0             # [kg]

h = 40.0                  # [W/(m^2*K)]
k_steel = 40.0            # [W/(m*K)]
rho_steel = 7800.0        # [kg/m^3]
cp_steel = 600.0          # [J/(kg*K)]

# radiation parameters (used only in the "with radiation" run)
epsilon = 0.8
sigma = 5.670374419e-8    # [W/(m^2*K^4)]

# oil cp
cp_oil_const = 2118.0     # [J/(kg*K)]

# engine oil cp(T) table (clamped beyond range)
T_tab = np.array([260, 280, 300, 320, 340, 360, 380, 400], dtype=float)
cp_tab = 1000.0 * np.array([1.76, 1.83, 1.91, 1.99, 2.08, 2.16, 2.25, 2.34], dtype=float)

USE_TEMP_DEP_CP = True

def cp_oil_engine(T):
    if T <= T_tab[0]:
        return float(cp_tab[0])
    if T >= T_tab[-1]:
        return float(cp_tab[-1])
    return float(np.interp(T, T_tab, cp_tab))

def cp_oil(T):
    return cp_oil_engine(T) if USE_TEMP_DEP_CP else cp_oil_const

# ============================================================
# GEOMETRY & BIOT
# ============================================================
r = d_ball / 2.0
V_ball = (4.0/3.0) * math.pi * r**3
A_ball = 4.0 * math.pi * r**2
m_ball = rho_steel * V_ball
N = m_steel / m_ball
A_total = N * A_ball

Lc = V_ball / A_ball  # = r/3 for sphere
Bi = h * Lc / k_steel

UA = h * A_total
C_s = m_steel * cp_steel

# ============================================================
# SIMULATOR
# ============================================================
def simulate(include_radiation: bool):
    """
    Returns time array, T_s(t), T_o(t), and time-to-target (or None).
    """
    def rhs(t, y):
        T_s, T_o = y
        Q_conv = UA * (T_s - T_o)

        if include_radiation:
            Q_rad = epsilon * sigma * A_total * (T_s**4 - T_o**4)
        else:
            Q_rad = 0.0

        Q_tot = Q_conv + Q_rad

        dT_s = -Q_tot / C_s
        dT_o = +Q_tot / (m_oil * cp_oil(T_o))
        return np.array([dT_s, dT_o], dtype=float)

    def rk4_step(t, y, dt):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5*dt, y + 0.5*dt*k1)
        k3 = rhs(t + 0.5*dt, y + 0.5*dt*k2)
        k4 = rhs(t + dt, y + dt*k3)
        return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    nsteps = int(t_end / dt) + 1
    t = np.linspace(0.0, t_end, nsteps)
    T_s = np.zeros(nsteps)
    T_o = np.zeros(nsteps)

    T_s[0] = T_ball0
    T_o[0] = T_oil0

    t_reach = None

    for i in range(1, nsteps):
        y_prev = np.array([T_s[i-1], T_o[i-1]], dtype=float)
        y_new = rk4_step(t[i-1], y_prev, dt)
        T_s[i], T_o[i] = y_new

        if t_reach is None and T_s[i] <= T_target:
            Ts_prev, Ts_now = T_s[i-1], T_s[i]
            if Ts_prev != Ts_now:
                frac = (Ts_prev - T_target) / (Ts_prev - Ts_now)
                frac = min(max(frac, 0.0), 1.0)
                t_reach = t[i-1] + frac * dt
            else:
                t_reach = t[i]

    return t, T_s, T_o, t_reach

# Run both cases
t0, Ts_no, To_no, treach_no = simulate(include_radiation=False)
t1, Ts_rad, To_rad, treach_rad = simulate(include_radiation=True)

# ============================================================
# PRINT OUTPUT
# ============================================================
print("\n--- Model summary ---")
print(f"Biot number: {Bi:.5f}  (<0.1 => lumped ball temperature typically OK)")
print(f"Oil cp model: {'engine oil cp(T) table' if USE_TEMP_DEP_CP else f'constant cp = {cp_oil_const:.0f} J/(kg*K)'}")
print(f"Target temperature: {T_target:.1f} K")
print(f"Emissivity used (radiation case): {epsilon:.2f}")

print("\nTime to target:")
if treach_no is None:
    print("  Convection only: not reached within t_end")
else:
    print(f"  Convection only: {treach_no:.1f} s ({treach_no/60:.2f} min)")

if treach_rad is None:
    print("  Conv + radiation: not reached within t_end")
else:
    print(f"  Conv + radiation: {treach_rad:.1f} s ({treach_rad/60:.2f} min)")

# ============================================================
# PLOTS
# 1) Ball temperature comparison
# ============================================================
plt.figure()
plt.plot(t0, Ts_no, label="Ball T (convection only)")
plt.plot(t1, Ts_rad, label="Ball T (convection + radiation)")
plt.axhline(T_target, linestyle=":", label="Target")
plt.xlabel("Time [s]")
plt.ylabel("Ball temperature [K]")
plt.title("Ball Cooling: Radiation vs No Radiation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: oil temperature comparison (helps sanity-check)
plt.figure()
plt.plot(t0, To_no, label="Oil T (convection only)")
plt.plot(t1, To_rad, label="Oil T (convection + radiation)")
plt.xlabel("Time [s]")
plt.ylabel("Oil temperature [K]")
plt.title("Oil Heating: Radiation vs No Radiation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

