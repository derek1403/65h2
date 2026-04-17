# core/physics.py — SWE 與 N-S 方程
# 算術核心以 Numba @njit(parallel=True) 加速，物理邏輯與正負號與原版完全相同

import numpy as np
from math import sqrt, exp, fabs
from numba import njit, prange
import os

from config import g, rho, f as f_cor
from core.math_tools import MathTools, fft2, ifft2


# ==============================================================================
# Numba 核心函數（類別外部，無 self）
# FFT 相關運算已在呼叫前完成，這裡只做純純量四則運算
# ==============================================================================

@njit(parallel=True, fastmath=True)
def _calc_ns_eq_numba(
    u, v,
    dPx, dPy,
    dux, duy, dvx, dvy,
    lap_u, lap_v,
    w,
    u_r, v_r,
    H, nu2, f, rho,
    vel_factor          # = 0.78，拉出來讓 Numba 當純量接收
):
    Ny, Nx = u.shape
    u_term = np.empty_like(u)
    v_term = np.empty_like(v)

    for i in prange(Ny):
        for j in range(Nx):
            # ---- 壓力梯度力 ----
            PF_u = -dPx[i, j] / rho
            PF_v = -dPy[i, j] / rho

            # ---- 科氏力 ----
            CF_u =  f * v[i, j]
            CF_v = -f * u[i, j]

            # ---- 垂直對流：w_p = -0.5*(|w|-w)，只保留 w<0（下沉）部分 ----
            w_p = -0.5 * (fabs(w[i, j]) - w[i, j])

            # ---- 水平平流 + 垂直夾卷 ----
            ADV_u = (u[i,j]*dux[i,j] + v[i,j]*duy[i,j]
                     + w_p * (u_r[i,j] - u[i,j]) / H)
            ADV_v = (u[i,j]*dvx[i,j] + v[i,j]*dvy[i,j]
                     + w_p * (v_r[i,j] - v[i,j]) / H)

            # ---- 黏滯 ----
            VC_u = nu2 * lap_u[i, j]
            VC_v = nu2 * lap_v[i, j]

            # ---- 地表摩擦（Large & Pond 參數化，與原版完全相同） ----
            vel = vel_factor * sqrt(u[i,j]**2 + v[i,j]**2)

            if vel <= 1e-5:
                CD = 0.0
            elif vel <= 25.0:
                CD = 1e-3 * (2.7 / vel + 0.142 + 0.0764 * vel)
            else:
                CD = 1e-3 * (2.16 + 0.5406 * (1.0 - exp(-((vel - 25.0) / 7.5))))

            FR_u = CD * vel * u[i, j] / H
            FR_v = CD * vel * v[i, j] / H

            # ---- 合計 ----
            u_term[i, j] = PF_u + CF_u - ADV_u + VC_u - FR_u
            v_term[i, j] = PF_v + CF_v - ADV_v + VC_v - FR_v

    return u_term, v_term


@njit(parallel=True, fastmath=True)
def _calc_swe_numba(
    u, v, h,
    dhx, dhy,
    dux, duy, dvx, dvy,
    lap_u, lap_v,
    w_sfc, u_sfc, v_sfc,
    Q,
    H, nu1, f, g
):
    Ny, Nx = u.shape
    u_term = np.empty_like(u)
    v_term = np.empty_like(v)
    h_term = np.empty_like(h)

    for i in prange(Ny):
        for j in range(Nx):
            # ---- 垂直對流：w_p = 0.5*(|w_sfc|+w_sfc)，只保留 w>0（上升）部分 ----
            w_p = 0.5 * (fabs(w_sfc[i, j]) + w_sfc[i, j])

            # ---- u 方程 ----
            u_term[i, j] = (
                -g * dhx[i, j]
                + f * v[i, j]
                - u[i,j] * dux[i,j]
                - v[i,j] * duy[i,j]
                - w_p * (u[i,j] - u_sfc[i,j]) / H
                + nu1 * lap_u[i, j]
            )

            # ---- v 方程 ----
            v_term[i, j] = (
                -g * dhy[i, j]
                - f * u[i, j]
                - u[i,j] * dvx[i,j]
                - v[i,j] * dvy[i,j]
                - w_p * (v[i,j] - v_sfc[i,j]) / H
                + nu1 * lap_v[i, j]
            )

            # ---- h 方程 ----
            h_term[i, j] = (
                -(H + h[i,j]) * (dux[i,j] + dvy[i,j])
                - u[i,j] * dhx[i,j]
                - v[i,j] * dhy[i,j]
                - (H + h[i,j]) * Q[i, j]
            )

    return u_term, v_term, h_term


# ==============================================================================
# Physics 類別：負責 FFT 算子呼叫，然後把導數陣列傳給 Numba 核心
# ==============================================================================

class Physics:
    def __init__(self, math: MathTools, H, nu1, nu2):
        self.math = math
        self.H    = H
        self.nu1  = nu1
        self.nu2  = nu2
        self._warmed_up = False

    def warmup(self, u_dummy, v_dummy, h_dummy):
        """
        Numba 第一次呼叫時會進行 JIT 編譯（約數秒）。
        在模擬開始前主動觸發，避免第一個 timestep 出現延遲。
        在 main.py 初始化完成後呼叫一次即可：
            physics.warmup(u, v, h)
        """
        if not self._warmed_up:
            print("Numba JIT warmup...")
            Q_dummy   = np.zeros_like(h_dummy)
            w_dummy   = np.zeros_like(h_dummy)
            P_dummy   = np.zeros_like(h_dummy)
            self.N_S_EQ(np.array([u_dummy, v_dummy]), 0,
                        u_dummy, v_dummy, w_dummy, h_dummy, P_dummy)
            self.SWE(np.array([u_dummy, v_dummy, h_dummy]), 0,
                     Q_dummy, w_dummy, u_dummy, v_dummy)
            self._warmed_up = True
            print("Numba JIT warmup done.")

    # ------------------------------------------------------------------
    # ini_wind：由渦度場反推初始風場與壓力場，與原版完全相同
    # ------------------------------------------------------------------
    def ini_wind(self, zeta):
        g_  = self.math.grid
        k2  = g_.k_squared

        zeta_hat = fft2(zeta)
        with np.errstate(invalid='ignore', divide='ignore'):
            psi_hat = np.where(k2 == 0, zeta_hat, -zeta_hat / k2)
        psi_hat = np.where(
            np.logical_or(abs(g_.K_o) > self.math.mx, abs(g_.L_o) > self.math.my),
            0, psi_hat
        )

        u = -np.real(ifft2(1j * g_.ky * psi_hat))
        v =  np.real(ifft2(1j * g_.kx * psi_hat))

        Lap_P = (
            rho * f_cor * np.real(ifft2(-k2 * psi_hat))
            + 2 * (
                np.real(ifft2(-(g_.kx**2) * psi_hat)) * np.real(ifft2(-(g_.ky**2) * psi_hat))
                - (np.real(ifft2(-(g_.kx * g_.ky) * psi_hat)))**2
            )
        )

        with np.errstate(invalid='ignore', divide='ignore'):
            P_hat = np.where(k2 == 0, fft2(Lap_P), -fft2(Lap_P) / k2)
        P_hat = np.where(
            np.logical_or(abs(g_.K_o) > self.math.mx, abs(g_.L_o) > self.math.my),
            0, P_hat
        )
        P = np.real(ifft2(P_hat))
        return u, v, P

    # ------------------------------------------------------------------
    # N_S_EQ：先用 MathTools 算出所有導數，再交給 Numba 核心
    # ------------------------------------------------------------------
    def N_S_EQ(self, wind, t, u_r, v_r, w, h, P):
        u, v = wind
        math = self.math

        u_hat = math.wave_filter(u)
        v_hat = math.wave_filter(v)
        P_hat = math.wave_filter(P)

        dPx, dPy = math.Spatial_diff(P, P_hat)
        dux, duy = math.Spatial_diff(u, u_hat)
        dvx, dvy = math.Spatial_diff(v, v_hat)
        lap_u    = math.Laplace(u, u_hat)
        lap_v    = math.Laplace(v, v_hat)

        # 確保陣列為 C-contiguous float64（Numba 要求）
        u    = np.ascontiguousarray(u,    dtype=np.float64)
        v    = np.ascontiguousarray(v,    dtype=np.float64)
        u_r  = np.ascontiguousarray(u_r,  dtype=np.float64)
        v_r  = np.ascontiguousarray(v_r,  dtype=np.float64)
        w    = np.ascontiguousarray(w,    dtype=np.float64)
        dPx  = np.ascontiguousarray(dPx,  dtype=np.float64)
        dPy  = np.ascontiguousarray(dPy,  dtype=np.float64)
        dux  = np.ascontiguousarray(dux,  dtype=np.float64)
        duy  = np.ascontiguousarray(duy,  dtype=np.float64)
        dvx  = np.ascontiguousarray(dvx,  dtype=np.float64)
        dvy  = np.ascontiguousarray(dvy,  dtype=np.float64)
        lap_u = np.ascontiguousarray(lap_u, dtype=np.float64)
        lap_v = np.ascontiguousarray(lap_v, dtype=np.float64)

        u_term, v_term = _calc_ns_eq_numba(
            u, v,
            dPx, dPy,
            dux, duy, dvx, dvy,
            lap_u, lap_v,
            w, u_r, v_r,
            float(self.H), float(self.nu2), float(f_cor), float(rho),
            0.78
        )
        return np.array([u_term, v_term])

    # ------------------------------------------------------------------
    # SWE：同上，先算導數，再交給 Numba 核心
    # ------------------------------------------------------------------
    def SWE(self, var, t, Q, w_sfc, u_sfc, v_sfc):
        u, v, h = var
        math    = self.math

        u_hat = math.wave_filter(u)
        v_hat = math.wave_filter(v)
        h_hat = math.wave_filter(h)

        dhx, dhy = math.Spatial_diff(h, h_hat)
        dux, duy = math.Spatial_diff(u, u_hat)
        dvx, dvy = math.Spatial_diff(v, v_hat)
        lap_u    = math.Laplace(u, u_hat)
        lap_v    = math.Laplace(v, v_hat)

        # 確保 Q 是陣列（spin-up 期間可能傳入純量 0）
        if np.isscalar(Q):
            Q = np.zeros_like(h)
        if np.isscalar(w_sfc):
            w_sfc = np.zeros_like(h)
        if np.isscalar(u_sfc):
            u_sfc = np.zeros_like(h)
        if np.isscalar(v_sfc):
            v_sfc = np.zeros_like(h)

        u     = np.ascontiguousarray(u,     dtype=np.float64)
        v     = np.ascontiguousarray(v,     dtype=np.float64)
        h     = np.ascontiguousarray(h,     dtype=np.float64)
        dhx   = np.ascontiguousarray(dhx,   dtype=np.float64)
        dhy   = np.ascontiguousarray(dhy,   dtype=np.float64)
        dux   = np.ascontiguousarray(dux,   dtype=np.float64)
        duy   = np.ascontiguousarray(duy,   dtype=np.float64)
        dvx   = np.ascontiguousarray(dvx,   dtype=np.float64)
        dvy   = np.ascontiguousarray(dvy,   dtype=np.float64)
        lap_u = np.ascontiguousarray(lap_u, dtype=np.float64)
        lap_v = np.ascontiguousarray(lap_v, dtype=np.float64)
        w_sfc = np.ascontiguousarray(w_sfc, dtype=np.float64)
        u_sfc = np.ascontiguousarray(u_sfc, dtype=np.float64)
        v_sfc = np.ascontiguousarray(v_sfc, dtype=np.float64)
        Q     = np.ascontiguousarray(Q,     dtype=np.float64)

        u_term, v_term, h_term = _calc_swe_numba(
            u, v, h,
            dhx, dhy,
            dux, duy, dvx, dvy,
            lap_u, lap_v,
            w_sfc, u_sfc, v_sfc,
            Q,
            float(self.H), float(self.nu1), float(f_cor), float(g)
        )
        return np.array([u_term, v_term, h_term])

    # ------------------------------------------------------------------
    # damping：與原版完全相同
    # ------------------------------------------------------------------
    def damping(self, f_arr):
        r = self.math.grid.r
        damp = np.where(r < 1, 1 - np.exp(-80 / r * np.exp(1 / (r - 1))), 0)
        return f_arr * damp