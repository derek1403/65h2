# ======================================================================
# models/one_way.py
# ======================================================================
import numpy as np
from models.base_model import BaseModel


class OneWayModel(BaseModel):
    """OW：邊界層對自由大氣完全無回饋（Q=0, w_in=u_in=v_in=0）"""

    def step_free_atmos(self, u, v, h, t, w_sfc, u_sfc, v_sfc):
        return self.math.RK4(
            self.physics.SWE,
            np.array([u, v, h]),
            t, self.dt,
            0, 0, 0, 0          # Q=0, w_in=0, u_in=0, v_in=0
        )

