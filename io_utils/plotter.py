# io_utils/plotter.py — 專職繪圖，與原版邏輯完全相同

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import matplotlib.cm as cm
from scipy.fft import fft2 as _fft2, ifft2 as _ifft2
import os

from core.grid import Grid

_N_WORKERS = os.cpu_count()
fft2  = lambda x: _fft2 (x, workers=_N_WORKERS)
ifft2 = lambda x: _ifft2(x, workers=_N_WORKERS)


class Plotter:
    def __init__(self, grid: Grid, dt):
        self.grid = grid
        self.dt   = dt

    def _parse_time(self, ts):
        t    = ts * self.dt
        hour = t // 3600
        mint = (t - hour * 3600) // 60
        secd = t - hour * 3600 - mint * 60
        return int(hour), int(mint), int(secd)

    # ------------------------------------------------------------------
    # plot_uvp：與原版完全相同
    # ------------------------------------------------------------------
    def plot_uvp(self, u, v, h, u_sfc, v_sfc, P, ts, path):
        hour, mint, secd = self._parse_time(ts)

        h_max = 20;  h_min = -50
        P_max = 200; P_min = -500
        c_lev_h = list(np.linspace(h_min, 0, 25)) + list(np.linspace(2, 20, 10))
        c_lev_P = list(np.linspace(P_min, 0, 25)) + list(np.linspace(20, 200, 10))

        color1 = ['#041842','#072563','#093185','#0b3ea6','#0e4ac7',
                  '#1057e9','#2d6df0','#4e84f2','#709bf4','#91b2f7']
        color2 = ['#8bd9d7','#71d0ce','#56c7c5','#3dbbb9','#35a19f']
        color3 = ['#21810d','#2aa111','#32c114','#3be118','#53e934',
                  '#6fec54','#8af074','#a5f394','#ffffff']
        color4 = ['#ffffff','#ffa989','#ff8f66','#ff7542','#ff5b1e']
        color5 = ['#f66','#ff1e1e','#d60000','#8e0000','#6b0000','#470000']
        color  = color1 + color2 + color3 + color4 + color5

        cmap = mcolor.ListedColormap(color)
        cmap.set_over('#6b0000')

        g = self.grid
        norm = mcolor.BoundaryNorm(c_lev_h, ncolors=35)
        fig, ax = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(12, 24))

        CS_h = ax[0].contourf(g.X, g.Y, h, norm=norm, levels=c_lev_h, cmap=cmap, extend='both')
        cbar_ax = fig.add_axes([0.92, 0.51, 0.02, 0.4])
        d_tick = 10
        cb_h = plt.colorbar(CS_h, ticks=np.arange(h_min, h_max+d_tick, d_tick), cax=cbar_ax)
        cb_h.set_ticklabels(np.arange(h_min, h_max+d_tick, d_tick))
        cb_h.set_label('h [m]', fontsize=20)

        norm = mcolor.BoundaryNorm(c_lev_P, ncolors=35)
        CS_P = ax[1].contourf(g.X, g.Y, P, norm=norm, levels=c_lev_P, cmap=cmap, extend='both')
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.4])
        d_tick = 100
        cb_P = plt.colorbar(CS_P, ticks=np.arange(P_min, P_max+d_tick, d_tick), cax=cbar_ax)
        cb_P.set_ticklabels(np.arange(P_min, P_max+d_tick, d_tick).astype(int))
        cb_P.set_label('P [Pa]', fontsize=20)

        inter = 20
        Q_arr   = ax[0].quiver(g.X[::inter,::inter], g.Y[::inter,::inter],
                                u[::inter,::inter],   v[::inter,::inter], scale=250)
        Q_sfc   = ax[1].quiver(g.X[::inter,::inter], g.Y[::inter,::inter],
                                u_sfc[::inter,::inter], v_sfc[::inter,::inter], scale=250)
        ax[0].quiverkey(Q_arr, X=0.8, Y=0.89, U=20, label='20 m/s',
                        labelpos='E', coordinates='figure',
                        fontproperties={'size': 15})

        ax[1].set_xticks(np.linspace(0, g.Lx, 11))
        ax[1].set_xticklabels(np.linspace(0, g.Lx//1000, 11).astype(int))
        for i in range(2):
            ax[i].set_yticks(np.linspace(0, g.Ly, 11))
            ax[i].set_yticklabels(np.linspace(0, g.Ly//1000, 11).astype(int))
            ax[i].set_ylabel('y [km]', fontsize=20)

        ax[0].set_xlim([0, g.Lx]); ax[0].set_ylim([0, g.Ly])
        ax[1].set_xlabel('x [km]', fontsize=20)
        ax[0].set_title('Free Atmosphere', fontsize=20)
        ax[1].set_title('Boundary Layer', fontsize=20)
        plt.suptitle(f'Time :{hour:2d} hr {mint:2d} min {secd:2d} s ,'
                     r' f=5 x 10$^{-5}$ s$^{-1}$', fontsize=25)
        plt.savefig(path + f'/uvp_{ts//12}.png')
        plt.close()

    # ------------------------------------------------------------------
    # plot_vor：與原版完全相同
    # ------------------------------------------------------------------
    def plot_vor(self, u, v, u_sfc, v_sfc, ts, path):
        hour, mint, secd = self._parse_time(ts)
        g = self.grid

        vor     = (np.real(ifft2(1j*g.kx*fft2(v)))
                   - np.real(ifft2(1j*g.ky*fft2(u))))
        vor_sfc = (np.real(ifft2(1j*g.kx*fft2(v_sfc)))
                   - np.real(ifft2(1j*g.ky*fft2(u_sfc))))

        cmap = cm.jet
        cmap.set_under('#ffffff')
        lev = np.round(np.linspace(0.3, 3, 10), 1) * 1e-3

        fig, ax = plt.subplots(2, 1, sharex='all', figsize=(13, 24))
        CS = ax[0].contourf(g.X, g.Y, vor,     levels=lev, cmap=cmap, extend='both')
        ax[1].contourf(g.X, g.Y, vor_sfc, levels=lev, cmap=cmap, extend='both')

        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cb = plt.colorbar(CS, ticks=lev, cax=cbar_ax, extend='both')
        cb.set_ticklabels(np.round(lev * 1e3, 1))
        cb.set_label(r'vorticity [s$^{-1}$]', fontsize=20)

        x_ticks = np.arange(g.Lx//2 - 70e3, g.Lx//2 + 70e3, 10e3)
        y_ticks = np.arange(g.Ly//2 - 70e3, g.Ly//2 + 70e3, 10e3)
        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels(x_ticks.astype(int) // 1000)
        for j in range(2):
            ax[j].set_yticks(y_ticks)
            ax[j].set_yticklabels(y_ticks.astype(int) // 1000)
            ax[j].set_ylabel('y [km]', fontsize=20)

        ax[0].set_xlim([g.Lx//2 - 70e3, g.Lx//2 + 70e3])
        ax[0].set_ylim([g.Ly//2 - 70e3, g.Ly//2 + 70e3])
        ax[1].set_ylim([g.Ly//2 - 70e3, g.Ly//2 + 70e3])
        ax[1].set_xlabel('x [km]', fontsize=20)
        ax[0].set_title('vorticity in the free atmosphere', fontsize=20)
        ax[1].set_title('vorticity in the boundary layer',  fontsize=20)
        plt.suptitle(f'Time : {hour:2d} hr {mint:2d} min {secd:2d} s', fontsize=25)
        plt.savefig(path + f'/vor_{ts//12}.png')
        plt.close()