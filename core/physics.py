# core/physics.py — SWE 與 N-S 方程，內部邏輯與原版完全相同

import numpy as np
import numexpr as ne
import os
from config import g, rho, f as f_cor
from core.math_tools import MathTools, fft2, ifft2

ne.set_num_threads(os.cpu_count())


class Physics:
    """
    只放物理方程式本體。
    需要一個 MathTools 物件來做微分與 FFT。
    """
    def __init__(self, math: MathTools, H, nu1, nu2):
        self.math = math
        self.H    = H
        self.nu1  = nu1
        self.nu2  = nu2

    # ------------------------------------------------------------------
    # ini_wind：由渦度場反推初始風場與壓力場
    # 與原版邏輯、np.errstate 完全相同
    # ------------------------------------------------------------------
    def ini_wind(self, zeta):
        g_   = self.math.grid
        k2   = g_.k_squared

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
    # N_S_EQ：邊界層方程，與原版邏輯完全相同
    # 每個輸入變數只做一次 wave_filter（原版做兩次）
    # ------------------------------------------------------------------
    def N_S_EQ(self, wind, t, u_r, v_r, w, h, P):
        u, v   = wind
        H      = self.H
        math   = self.math

        u_hat = math.wave_filter(u)
        v_hat = math.wave_filter(v)
        P_hat = math.wave_filter(P)

        dPx, dPy = math.Spatial_diff(P, P_hat)
        dux, duy = math.Spatial_diff(u, u_hat)
        dvx, dvy = math.Spatial_diff(v, v_hat)
        lap_u    = math.Laplace(u, u_hat)
        lap_v    = math.Laplace(v, v_hat)

        PF_u = ne.evaluate("-dPx/rho",  local_dict={'dPx': dPx, 'rho': rho})
        PF_v = ne.evaluate("-dPy/rho",  local_dict={'dPy': dPy, 'rho': rho})
        CF_u = ne.evaluate("f*v",       local_dict={'f': f_cor, 'v': v})
        CF_v = ne.evaluate("-f*u",      local_dict={'f': f_cor, 'u': u})

        # w_p = -0.5*(|w|-w)，只保留 w<0 的部分（與原版相同）
        w_p = ne.evaluate("0.5*(abs(w)-w)", local_dict={'w': w})

        ADV_u = ne.evaluate("u*dux + v*duy + w_p*(u_r-u)/H",
                            local_dict={'u':u,'dux':dux,'v':v,'duy':duy,
                                        'w_p':w_p,'u_r':u_r,'H':H})
        ADV_v = ne.evaluate("u*dvx + v*dvy + w_p*(v_r-v)/H",
                            local_dict={'u':u,'dvx':dvx,'v':v,'dvy':dvy,
                                        'w_p':w_p,'v_r':v_r,'H':H})

        VC_u = self.nu2 * lap_u
        VC_v = self.nu2 * lap_v

        # CD：條件式保留原版 np.where，數值邏輯不動
        vel = ne.evaluate("0.78*sqrt(u**2+v**2)", local_dict={'u': u, 'v': v})
        CD  = np.where((vel > 1e-5) & (vel <= 25),
                       (1e-3) * (2.7/vel + 0.142 + 0.0764*vel), 0)
        CD  = np.where(vel > 25,
                       (1e-3) * (2.16 + 0.5406*(1 - np.exp(-((vel-25)/7.5)))), CD)

        FR_u = ne.evaluate("CD*vel*u/H", local_dict={'CD':CD,'vel':vel,'u':u,'H':H})
        FR_v = ne.evaluate("CD*vel*v/H", local_dict={'CD':CD,'vel':vel,'v':v,'H':H})

        u_term = ne.evaluate("PF_u+CF_u - ADV_u + VC_u - FR_u",
                             local_dict={'PF_u':PF_u,'CF_u':CF_u,'ADV_u':ADV_u,
                                         'VC_u':VC_u,'FR_u':FR_u})
        v_term = ne.evaluate("PF_v+CF_v - ADV_v + VC_v - FR_v",
                             local_dict={'PF_v':PF_v,'CF_v':CF_v,'ADV_v':ADV_v,
                                         'VC_v':VC_v,'FR_v':FR_v})
        return np.array([u_term, v_term])

    # ------------------------------------------------------------------
    # SWE：自由大氣淺水方程，與原版邏輯完全相同
    # ------------------------------------------------------------------
    def SWE(self, var, t, Q, w_sfc, u_sfc, v_sfc):
        u, v, h = var
        H       = self.H
        math    = self.math

        u_hat = math.wave_filter(u)
        v_hat = math.wave_filter(v)
        h_hat = math.wave_filter(h)

        dhx, dhy = math.Spatial_diff(h, h_hat)
        dux, duy = math.Spatial_diff(u, u_hat)
        dvx, dvy = math.Spatial_diff(v, v_hat)
        lap_u    = math.Laplace(u, u_hat)
        lap_v    = math.Laplace(v, v_hat)

        # w_p = 0.5*(|w_sfc|+w_sfc)，只保留 w>0 的部分（與原版相同）
        w_p = ne.evaluate("0.5*(abs(w_sfc)+w_sfc)", local_dict={'w_sfc': w_sfc})

        u_term = ne.evaluate(
            "-g*dhx + f*v - u*dux - v*duy - w_p*(u-u_sfc)/H + nu1*lap_u",
            local_dict={'g':g,'dhx':dhx,'f':f_cor,'v':v,'u':u,'dux':dux,'duy':duy,
                        'w_p':w_p,'u_sfc':u_sfc,'H':H,'nu1':self.nu1,'lap_u':lap_u})
        v_term = ne.evaluate(
            "-g*dhy - f*u - u*dvx - v*dvy - w_p*(v-v_sfc)/H + nu1*lap_v",
            local_dict={'g':g,'dhy':dhy,'f':f_cor,'u':u,'v':v,'dvx':dvx,'dvy':dvy,
                        'w_p':w_p,'v_sfc':v_sfc,'H':H,'nu1':self.nu1,'lap_v':lap_v})
        h_term = ne.evaluate(
            "-(H+h)*(dux+dvy) - u*dhx - v*dhy - (H+h)*Q",
            local_dict={'H':H,'h':h,'dux':dux,'dvy':dvy,
                        'u':u,'dhx':dhx,'v':v,'dhy':dhy,'Q':Q})

        return np.array([u_term, v_term, h_term])

    # ------------------------------------------------------------------
    # damping：與原版完全相同
    # ------------------------------------------------------------------
    def damping(self, f_arr):
        r = self.math.grid.r
        damp = np.where(r < 1, 1 - np.exp(-80/r * np.exp(1/(r-1))), 0)
        return f_arr * damp