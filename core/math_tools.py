# core/math_tools.py — 純數學算子：FFT 微分、Laplace、RK4、wave_filter
# 完全不知道自己在處理哪個物理變數

import numpy as np
from scipy.fft import fft2 as _fft2, ifft2 as _ifft2
import os

_N_WORKERS = os.cpu_count()

# 多執行緒 FFT（與原版 SWE_func2.py 相同）
def fft2(x):
    return _fft2(x, workers=_N_WORKERS)

def ifft2(x):
    return _ifft2(x, workers=_N_WORKERS)


class MathTools:
    """
    所有算子都需要網格資訊（kx, ky, k_squared, K, L, K_o, L_o），
    所以接收一個 Grid 物件。
    """
    def __init__(self, grid, mx, my):
        self.grid = grid
        self.mx   = mx
        self.my   = my

    # ------------------------------------------------------------------
    # wave_filter：與原版邏輯、np.errstate 完全相同
    # ------------------------------------------------------------------
    def wave_filter(self, f):
        g = self.grid
        with np.errstate(invalid='ignore', divide='ignore'):
            f_hat = np.where(
                np.logical_or(g.K == 0, g.L == 0),
                fft2(f),
                fft2(f)
                * (np.sin(g.K * np.pi / (2 * self.mx)) / (g.K * np.pi / (2 * self.mx)))
                * (np.sin(g.L * np.pi / (2 * self.my)) / (g.L * np.pi / (2 * self.my)))
            )
        f_hat = np.where(
            np.logical_or(abs(g.K_o) > self.mx, abs(g.L_o) > self.my),
            0, f_hat
        )
        return f_hat

    # ------------------------------------------------------------------
    # Spatial_diff：與原版相同，支援傳入已算好的 f_hat 避免重複 fft
    # ------------------------------------------------------------------
    def Spatial_diff(self, f, f_hat=None):
        if f_hat is None:
            f_hat = self.wave_filter(f)
        return (np.real(ifft2(1j * self.grid.kx * f_hat)),
                np.real(ifft2(1j * self.grid.ky * f_hat)))

    # ------------------------------------------------------------------
    # Laplace：與原版相同
    # ------------------------------------------------------------------
    def Laplace(self, f, f_hat=None):
        if f_hat is None:
            f_hat = self.wave_filter(f)
        return np.real(ifft2(-self.grid.k_squared * f_hat))

    # ------------------------------------------------------------------
    # D_Laplace：與原版相同
    # ------------------------------------------------------------------
    def D_Laplace(self, f, f_hat=None):
        if f_hat is None:
            f_hat = self.wave_filter(f)
        return np.real(ifft2(-(self.grid.k_squared ** 2) * f_hat))

    # ------------------------------------------------------------------
    # RK4：與原版完全相同
    # ------------------------------------------------------------------
    def RK4(self, func, y, t, dt, *args):
        k1 = dt * func(y,              t,              *args)
        k2 = dt * func(y + 0.5 * k1,  t + 0.5 * dt,  *args)
        k3 = dt * func(y + 0.5 * k2,  t + 0.5 * dt,  *args)
        k4 = dt * func(y + k3,         t + dt,         *args)
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6