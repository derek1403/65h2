# initial_conditions.py — 產生初始渦旋場
# 與原版 setting.py 的邏輯完全相同

import numpy as np
from config import Lx, Ly
from core.grid import Grid


def fun_S(s):
    return 1 - 3*s**2 + 2*s**3


def make_initial_vorticity(grid: Grid):
    X, Y = grid.X, grid.Y

    zeta0 = 2e-3
    x0, y0 = Lx // 2, Ly // 2

    # 橢圓渦旋（與原版完全相同）
    a1, b1 = 20e3, 40e3
    a2, b2 = 24e3, 44e3

    r1 = np.sqrt(((X - x0) / a1)**2 + ((Y - y0) / b1)**2)
    r2 = np.sqrt(((X - x0) / a2)**2 + ((Y - y0) / b2)**2)

    zeta = np.where(r1 < 1, zeta0, 0)
    zeta = np.where(
        (r1 >= 1) & (r2 <= 1),
        zeta0 * fun_S((1 - r1) / (r2 - r1)) - 1e-6 * fun_S((r2 - 1) / (r2 - r1)),
        zeta
    )
    zeta = np.where(r2 > 1, 0, zeta)

    return zeta