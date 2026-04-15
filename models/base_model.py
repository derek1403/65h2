# models/base_model.py — 定義所有模型共用的介面

import numpy as np
from abc import ABC, abstractmethod
from core.math_tools import MathTools
from core.physics import Physics


class BaseModel(ABC):
    """
    每個子類別只需要覆寫 step_free_atmos，
    決定自己的 Q / w_in / u_in / v_in 要怎麼傳給 SWE。
    主迴圈只需呼叫 model.step()，完全不用寫 if-elif。
    """
    def __init__(self, physics: Physics, math: MathTools, dt, H, rho, g, Q0):
        self.physics = physics
        self.math    = math
        self.dt      = dt
        self.H       = H
        self.rho     = rho
        self.g       = g
        self.Q0      = Q0

    @abstractmethod
    def step_free_atmos(self, u, v, h, t, w_sfc, u_sfc, v_sfc):
        """回傳更新後的 u, v, h"""
        ...

    def step(self, u, v, h, u_sfc, v_sfc, w_sfc, t, in_spinup: bool):
        """
        一個完整的 timestep（自由大氣 + 邊界層），
        與原版 shallow_3D_new.py 主迴圈邏輯完全相同。
        """
        h_pre = h.copy()

        # spin-up 期間邊界層不影響自由大氣
        if in_spinup:
            w_sfc_eff = np.zeros_like(w_sfc)
            u_sfc_eff = np.zeros_like(u_sfc)
            v_sfc_eff = np.zeros_like(v_sfc)
        else:
            w_sfc_eff = w_sfc
            u_sfc_eff = u_sfc
            v_sfc_eff = v_sfc

        # 自由大氣（各模型自己實作）
        u, v, h = self.step_free_atmos(u, v, h, t, w_sfc_eff, u_sfc_eff, v_sfc_eff)

        # 垂直速度與 h 截斷（與原版相同）
        w   = (h - h_pre) / self.dt
        h   = np.where(h > 0, 0, h)

        # 邊界層
        P = self.rho * self.g * h
        u_sfc, v_sfc = self.math.RK4(
            self.physics.N_S_EQ,
            np.array([u_sfc, v_sfc]),
            t, self.dt,
            u, v, w_sfc, h, P
        )
        w_sfc = -self.H * (
            self.math.Spatial_diff(u_sfc)[0] +
            self.math.Spatial_diff(v_sfc)[1]
        )

        return u, v, h, w, u_sfc, v_sfc, w_sfc, P