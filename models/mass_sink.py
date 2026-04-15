
# ======================================================================
# models/mass_sink.py
# ======================================================================
import numpy as np
from models.base_model import BaseModel


class MassSinkModel(BaseModel):
    """MS：邊界層向上的質量通量作為自由大氣的 mass sink"""

    def step_free_atmos(self, u, v, h, t, w_sfc, u_sfc, v_sfc):
        Q = w_sfc * self.Q0
        return self.math.RK4(
            self.physics.SWE,
            np.array([u, v, h]),
            t, self.dt,
            Q, 0, 0, 0          # Q≠0, w_in=0, u_in=0, v_in=0
        )

