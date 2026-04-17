# main.py — 程式唯一入口，只負責啟動、迴圈控制與輸出排程

import numpy as np
import time
import os

from config import (
    Lx, Ly, Nx, Ny, dt, hours, timesteps, SP,
    H, nu1, nu2, rho, g, Q0, mx, my,
    mm, mdl, mdl_name,
    OT_data, plot_data, OT_plot
)
from core.grid            import Grid
from core.math_tools      import MathTools
from core.physics         import Physics
from initial_conditions   import make_initial_vorticity
from io_utils.writer      import Writer
from io_utils.plotter     import Plotter

# 根據 mm 選對應模型類別
from models.one_way        import OneWayModel
from models.mass_sink      import MassSinkModel
from models.momentum_flux  import MomentumFluxModel

_MODEL_MAP = {
    'OW': OneWayModel,
    'MS': MassSinkModel,
    'MF': MomentumFluxModel,
}

# ==============================================================================
# 初始化
# ==============================================================================
dname     = mdl[mm]
data_path = 'data/';           os.makedirs(data_path, exist_ok=True)
data_name = 'elps_' + dname

plot_uvp_path = 'plot/uvp';    os.makedirs(plot_uvp_path, exist_ok=True)
plot_vor_path = 'plot/vor/';   os.makedirs(plot_vor_path, exist_ok=True)

ttl_s      = hours * 3600
ttl_output = str(ttl_s // OT_data + 1)

print('=' * 73)
print('Model: ' + mdl_name[mm] + '\n-')
print(f'Total Simulation Time: {hours:02d} hrs')
print(f'Time Step            : {dt} s')
print(f'Spin-up Time         : {SP:.1f} hours')
print(f'Data Output Interval : {OT_data} s')
print(f'Total Output Data    : {ttl_output} files')
print('=' * 73)

# 建立各部門物件
grid    = Grid()
math    = MathTools(grid, mx, my)
physics = Physics(math, H, nu1, nu2)
writer  = Writer(grid, nu1, nu2)
plotter = Plotter(grid, dt)

# 選擇模型
ModelClass = _MODEL_MAP[dname]
model = ModelClass(physics, math, dt, H, rho, g, Q0)

# 初始條件
zeta         = make_initial_vorticity(grid)
u, v, P      = physics.ini_wind(zeta)
h            = np.where(P < 0, 0, P / (rho * g))
u_sfc, v_sfc = u.copy(), v.copy()
w_sfc        = np.zeros(grid.X.shape)

# ==============================================================================
# Numba JIT 預熱（避免第一個 timestep 出現編譯延遲）
# ==============================================================================
physics.warmup(u, v, h)

# ==============================================================================
# 第 0 步輸出
# ==============================================================================
st = time.time()
writer.write_single_data(u, v, np.zeros_like(h), u_sfc, v_sfc, w_sfc,
                          h, P, hours, 0, dt, data_path, data_name)
if plot_data:
    plotter.plot_uvp(u, v, h, u_sfc, v_sfc, P, 0, plot_uvp_path)
    plotter.plot_vor(u, v, u_sfc, v_sfc, 0, plot_vor_path)

i = 1
et = time.time()
print(f"{i} of {ttl_output} files completed.")
print(f'Simulation Time: 00 hr(s) 00 min(s) 00 second(s)')
print(f'RunTime: {round(et - st, 2)} seconds.')
print('-' * 73)

# ==============================================================================
# 主時間迴圈
# ==============================================================================
for t in range(1, timesteps + 1):
    in_spinup = (t * dt <= 3600 * SP)

    u, v, h, w, u_sfc, v_sfc, w_sfc, P = model.step(
        u, v, h, u_sfc, v_sfc, w_sfc, t, in_spinup
    )

    # 資料輸出
    if t * dt % OT_data == 0:
        writer.write_single_data(u, v, w, u_sfc, v_sfc, w_sfc,
                                  h, P, hours, t, dt, data_path, data_name)
        i += 1
        et = time.time()
        sim_s = (i - 1) * OT_data
        print(f"{i} of {ttl_output} files completed.")
        print(f'Simulation Time: {sim_s//3600:02d} hr(s) '
              f'{(sim_s%3600)//60:02d} min(s) {sim_s%60:02d} second(s)')
        print(f'RunTime: {round(et - st, 2)} seconds.')
        print('-' * 73)

    # 繪圖輸出
    if plot_data and t * dt % OT_plot == 0:
        plotter.plot_uvp(u, v, h, u_sfc, v_sfc, P, t, plot_uvp_path)
        plotter.plot_vor(u, v, u_sfc, v_sfc, t, plot_vor_path)

    # 發散偵測
    if True in np.isnan(u_sfc):
        print('!' * 73)
        print(f'Simulation blows up at t = {t * dt} s.')
        print('!' * 73)
        break

et = time.time()
print(et - st)