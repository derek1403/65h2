# io_utils/writer.py — 專職將陣列寫入 NetCDF，與原版邏輯完全相同

import numpy as np
import netCDF4 as nc
from core.grid import Grid


class Writer:
    def __init__(self, grid: Grid, nu1, nu2):
        self.grid = grid
        self.nu1  = nu1
        self.nu2  = nu2

    def write_single_data(self, u, v, w, u_sfc, v_sfc, w_sfc, h, P,
                           hour, t, dt, path, name):
        """與原版 write_single_data 完全相同"""
        fname = path + name + '_' + str(t * dt // 60).zfill(4) + '.nc'
        f_w = nc.Dataset(fname, 'w', format='NETCDF4')

        f_w.createDimension('x', self.grid.Nx)
        f_w.createDimension('y', self.grid.Ny)

        for vname in ['x', 'y']:
            f_w.createVariable(vname, np.float32, (vname,))
        for vname in ['u', 'v', 'w', 'u_sfc', 'v_sfc', 'w_sfc', 'h', 'P']:
            f_w.createVariable(vname, np.float32, ('x', 'y'))

        f_w.variables['x'][:] = self.grid.x
        f_w.variables['y'][:] = self.grid.y
        f_w.variables['u'][:]     = u
        f_w.variables['v'][:]     = v
        f_w.variables['w'][:]     = w
        f_w.variables['u_sfc'][:] = u_sfc
        f_w.variables['v_sfc'][:] = v_sfc
        f_w.variables['w_sfc'][:] = w_sfc
        f_w.variables['h'][:]     = h
        f_w.variables['P'][:]     = P

        f_w.description = (f'Total time: {hour} hours'
                           f', at {t * dt // 60} mins'
                           f', nu of free ATM={self.nu1}, nu of BL={self.nu2}')
        f_w.close()

    def write_data(self, u, v, w, u_sfc, v_sfc, w_sfc, h, P,
                   hour, path, name):
        """與原版 write_data 完全相同"""
        Nt = 2 * hour + 1
        f_w = nc.Dataset(path + name + '.nc', 'w', format='NETCDF4')

        f_w.createDimension('time', Nt)
        f_w.createDimension('x', self.grid.Nx)
        f_w.createDimension('y', self.grid.Ny)

        f_w.createVariable('x', np.float32, ('x',))
        f_w.createVariable('y', np.float32, ('y',))
        for vname in ['u', 'v', 'w', 'u_sfc', 'v_sfc', 'w_sfc', 'h', 'P']:
            f_w.createVariable(vname, np.float32, ('time', 'x', 'y'))

        f_w.variables['x'][:] = self.grid.x
        f_w.variables['y'][:] = self.grid.y
        f_w.variables['u'][:]     = u
        f_w.variables['v'][:]     = v
        f_w.variables['w'][:]     = w
        f_w.variables['u_sfc'][:] = u_sfc
        f_w.variables['v_sfc'][:] = v_sfc
        f_w.variables['w_sfc'][:] = w_sfc
        f_w.variables['h'][:]     = h
        f_w.variables['P'][:]     = P

        f_w.description = (f'Total time: {hour} hours'
                           f', nu of free ATM={self.nu1}, nu of BL={self.nu2}')
        f_w.close()