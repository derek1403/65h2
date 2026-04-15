
# ======================================================================
# models/momentum_flux.py
# ======================================================================
import numpy as np
from models.base_model import BaseModel


class MomentumFluxModel(BaseModel):
    """MF：邊界層向上輸送動量（w_in, u_in, v_in 均非零）"""

    def step_free_atmos(self, u, v, h, t, w_sfc, u_sfc, v_sfc):
        return self.math.RK4(
            self.physics.SWE,
            np.array([u, v, h]),
            t, self.dt,
            0, w_sfc, u_sfc, v_sfc   # Q=0, w_in=w_sfc, u_in=u_sfc, v_in=v_sfc
        )