# core/grid.py — 空間與頻域網格，只負責建立網格，不做任何物理運算

import numpy as np
from config import Lx, Ly, Nx, Ny, mx, my


class Grid:
    def __init__(self):
        # 空間網格
        self.Lx = Lx; self.Ly = Ly
        self.Nx = Nx; self.Ny = Ny

        self.x  = np.linspace(0, Lx, Nx); self.dx = self.x[1] - self.x[0]
        self.y  = np.linspace(0, Ly, Ny); self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 頻域波數（原始，用來做截斷判斷）
        kx_raw = np.fft.fftfreq(Nx, d=Lx / Nx) * Lx
        ky_raw = np.fft.fftfreq(Ny, d=Ly / Ny) * Ly
        self.K_o, self.L_o = np.meshgrid(kx_raw, ky_raw)

        # 頻域波數（截斷後，換算成 rad/m，用來做微分）
        kx = np.where(abs(kx_raw) <= mx, kx_raw * 2 * np.pi / Lx, 0)
        ky = np.where(abs(ky_raw) <= my, ky_raw * 2 * np.pi / Ly, 0)
        self.kx, self.ky = np.meshgrid(kx, ky)

        # k² （截斷外設 0）
        self.k_squared = np.where(
            np.logical_or(abs(self.K_o) > mx, abs(self.L_o) > my),
            0,
            self.kx**2 + self.ky**2
        )

        # 無因次波數（用於 wave_filter 的 sinc 濾波）
        self.K = kx * Lx / (2 * np.pi)
        self.L = ky * Ly / (2 * np.pi)

        # 阻尼用的無因次半徑
        lnt = 350e3
        self.r = (((self.X - self.x[Nx // 2]) / lnt)**2 +
                  ((self.Y - self.y[Ny // 2]) / lnt)**2) ** 0.5