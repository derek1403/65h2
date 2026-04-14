import cProfile
import pstats
import numpy as np
import time
from setting import *
from SWE_func2 import SWE_functions

# 只跑少量 timestep 做 profiling
PROFILE_STEPS = 50  # 約 250 秒模擬時間，夠用了

fun = SWE_functions(Lx, Ly, Nx, Ny, dt, H, nu1, nu2)
u, v, P = fun.ini_wind(zeta)
h = np.where(P < 0, 0, P / (rho * g))
u_sfc, v_sfc = u.copy(), v.copy()
w_sfc = np.zeros(X.shape)
Q = np.zeros_like(h)

def run_steps():
    global u, v, h, u_sfc, v_sfc, w_sfc, Q, P
    for t in range(1, PROFILE_STEPS + 1):
        h_pre = h.copy()

        if t * dt <= 3600 * SP:
            Q = np.zeros_like(h)
            w_in = 0; u_in = 0; v_in = 0
        else:
            Q = w_sfc * Q0
            w_in = w_sfc; u_in = u_sfc; v_in = v_sfc

        # 改成你實際用的 model (mm=2 → MF)
        u, v, h = fun.RK4(fun.SWE, np.array([u, v, h]), t, 0, w_in, u_in, v_in)

        w = (h - h_pre) / dt
        h = np.where(h > 0, 0, h)

        P = rho * g * h
        u_sfc, v_sfc = fun.RK4(fun.N_S_EQ, np.array([u_sfc, v_sfc]), t, u, v, w_sfc, h, P)
        w_sfc = -H * (fun.Spatial_diff(u_sfc)[0] + fun.Spatial_diff(v_sfc)[1])

# ---- cProfile ----
profiler = cProfile.Profile()
profiler.enable()
run_steps()
profiler.disable()

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')

print("\n===== Top 20 cumulative time =====")
stats.print_stats(20)

print("\n===== Top 20 total time (no subcalls) =====")
stats.sort_stats('tottime')
stats.print_stats(20)