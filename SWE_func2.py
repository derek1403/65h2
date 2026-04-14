
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2 as _fft2, ifft2 as _ifft2
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolor
import matplotlib.cm as cm
import netCDF4 as nc


import numexpr as ne
import os
ne.set_num_threads(os.cpu_count())


_N_WORKERS = os.cpu_count()   # 自動抓 CPU 數，你的機器會是 16 或 32 之類的
                               # 也可以手動設，例如 _N_WORKERS = 8

fft2  = lambda x: _fft2 (x, workers=_N_WORKERS)
ifft2 = lambda x: _ifft2(x, workers=_N_WORKERS)


#==============================================================================
# Constant
#==============================================================================

g = 9.81; rho = 1.; f = 5e-5
mx, my = 256, 256

#==============================================================================
# FUNCTIONS
#==============================================================================
class SWE_functions:
    def __init__(self, Lx, Ly, Nx, Ny, dt, H, nu1, nu2):
        self.Lx = Lx; self.Ly = Ly; self.Nx = Nx; self.Ny = Ny; self.dt = dt
        self.x = np.linspace(0,Lx,Nx); self.dx = self.x[1]-self.x[0]
        self.y = np.linspace(0,Ly,Ny); self.dy = self.y[1]-self.y[0]
        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.H = H; self.nu1 = nu1; self.nu2 = nu2
        
        self.mx, self.my = mx, my
        kx = np.fft.fftfreq(Nx, d=Lx / Nx) * Lx
        ky = np.fft.fftfreq(Ny, d=Ly / Ny) * Ly
        self.K_o, self.L_o = np.meshgrid(kx, ky)
        kx = np.where(abs(kx)<=mx, kx*2*np.pi/Lx, 0)
        ky = np.where(abs(ky)<=my, ky*2*np.pi/Ly, 0)
        
        self.kx, self.ky = np.meshgrid(kx, ky)
        
        #3/13
        self.k_squared = np.where(np.logical_or(abs(self.K_o)>mx,abs(self.L_o)>my), 0, self.kx**2 + self.ky**2)
        #self.k_squared = self.kx**2 + self.ky**2

        self.K=kx*Lx/(2*np.pi); self.L=ky*Ly/(2*np.pi)
        
        lnt = 350e+3
        self.r = (((self.X-self.x[Nx//2])/lnt)**2+((self.Y-self.y[Nx//2])/lnt)**2)**0.5
    

    # ============================================================
    # 修改 wave_filter（原本有大量逐元素運算）
    # ============================================================
    def wave_filter(self, f):
        K, L = self.K, self.L
        K_o, L_o = self.K_o, self.L_o
        mx, my = self.mx, self.my

        f_hat = fft2(f)

        with np.errstate(invalid='ignore', divide='ignore'):
            # 原本：fft2(f) * (sin(...)/(...)） * (sin(...)/(...))
            # 用 numexpr 一次算完，省去中間暫存陣列
            scale = ne.evaluate(
                "where((K==0)|(L==0), 1.0,"
                " sin(K*pi/(2*mx)) / (K*pi/(2*mx))"
                " * sin(L*pi/(2*my)) / (L*pi/(2*my)))",
                local_dict={'K': K, 'L': L, 'mx': float(mx), 'my': float(my),
                            'pi': np.pi}
            )
            f_hat = ne.evaluate("where((abs(K_o)>mx)|(abs(L_o)>my), 0, f_hat*scale)",
                                local_dict={'K_o': K_o, 'L_o': L_o,
                                            'mx': float(mx), 'my': float(my),
                                            'f_hat': f_hat, 'scale': scale})
        return f_hat

        
    def ini_wind(self, zeta):
        zeta_hat = fft2(zeta)
        with np.errstate(invalid='ignore', divide='ignore'):
          psi_hat  = np.where(self.k_squared==0, zeta_hat, -zeta_hat / self.k_squared)
        psi_hat  = np.where(np.logical_or(abs(self.K_o)>mx,abs(self.L_o)>my), 0, psi_hat)
        
        u = -np.real(ifft2(1j*self.ky*psi_hat)); v = np.real(ifft2(1j*self.kx*psi_hat))
        
        Lap_P=rho*f*np.real(ifft2(-self.k_squared*psi_hat))+2*(np.real(ifft2(-(self.kx**2)*psi_hat))*np.real(ifft2(-(self.ky**2)*psi_hat))-(np.real(ifft2(-(self.kx*self.ky)*psi_hat)))**2)

        with np.errstate(invalid='ignore', divide='ignore'):
          P_hat=np.where(self.k_squared==0, fft2(Lap_P), -fft2(Lap_P)/self.k_squared)
        P_hat=np.where(np.logical_or(abs(self.K_o)>mx,abs(self.L_o)>my), 0, P_hat)
        
        #P_hat=fft2(Lap_P)/self.k_squared
        P = np.real(ifft2(P_hat))
        return( u, v ,P)
        
    def plot_uvp(self,u, v, h, u_sfc, v_sfc, P, ts, path):   
        # calculate time
        time = ts* self.dt
        hour = time//3600
        mint = (time-hour*3600)//60
        secd = time-hour*3600-mint*60
        i=time//60
        # plot
        h_max = 20 ; h_min = -50
        P_max = 200; P_min = -500
        c_lev_h = list(np.linspace(h_min,0,25))+list(np.linspace(2,20,10))
        c_lev_P = list(np.linspace(P_min,0,25))+list(np.linspace(20,200,10))

        color1=['#041842','#072563','#093185','#0b3ea6','#0e4ac7','#1057e9','#2d6df0','#4e84f2','#709bf4','#91b2f7']
        color2=['#8bd9d7','#71d0ce','#56c7c5','#3dbbb9','#35a19f']
        color3=['#21810d','#2aa111','#32c114','#3be118','#53e934','#6fec54','#8af074','#a5f394','#ffffff']
        #color3=['#f9f800','#fffe1e','#fffe42','#fffe66','#fffe89','#ffffff']
        color4=['#ffffff','#ffa989','#ff8f66','#ff7542','#ff5b1e']
        color5=['#f66'   ,'#ff1e1e','#d60000','#8e0000','#6b0000','#470000']
        color =color1+color2+color3+color4+color5

        cmap=mcolor.ListedColormap(color)
        cmap.set_over('#6b0000')
        
        norm=mcolor.BoundaryNorm(c_lev_h,ncolors=35)
        
        fig, ax = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(12,24))
        #CS_h = ax[0].contourf(self.X, self.Y, h, levels=np.linspace(-15,2.5,50), cmap='jet', extend='both')
        CS_h = ax[0].contourf(self.X, self.Y, h, norm=norm, levels=c_lev_h, cmap=cmap, extend='both')
        #ax[0].colorbar(ticks=np.arange(-0.2,1.2,0.2)*10, extend='both', label='h [m]')
        cbar_ax = fig.add_axes([0.92, 0.51, 0.02, 0.4])
        d_tick = 10
        cb_h = plt.colorbar(CS_h, ticks=np.arange(h_min,h_max+d_tick,d_tick), cax=cbar_ax)
        cb_h.set_ticklabels(np.arange(h_min,h_max+d_tick,d_tick))
        cb_h.set_label('h [m]', fontsize=20)
        
        norm=mcolor.BoundaryNorm(c_lev_P,ncolors=35)
        CS_P = ax[1].contourf(self.X, self.Y, P, norm=norm, levels=c_lev_P, cmap=cmap, extend='both')
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.4])
        d_tick = 100
        cb_P = plt.colorbar(CS_P, ticks=np.arange(P_min,P_max+d_tick,d_tick), cax=cbar_ax)
        cb_P.set_ticklabels(np.arange(P_min,P_max+d_tick,d_tick).astype(int))
        cb_P.set_label('P [Pa]', fontsize=20)
        
        inter = 20
        Q = ax[0].quiver(self.X[::inter,::inter], self.Y[::inter,::inter], u[::inter,::inter], v[::inter,::inter], scale=250)
        Q_sfc = ax[1].quiver(self.X[::inter,::inter], self.Y[::inter,::inter], u_sfc[::inter,::inter], v_sfc[::inter,::inter], scale=250)
    
        ax[0].quiverkey(Q, X=0.8, Y=0.89, U=20, label='20 m/s', labelpos='E', coordinates='figure', fontproperties={'size':15})
        
    
        ax[1].set_xticks(np.linspace(0,self.Lx,11))
        ax[1].set_xticklabels(np.linspace(0,self.Lx//1000,11).astype(int))
        for i in range(2):
            ax[i].set_yticks(np.linspace(0,self.Ly,11))
            ax[i].set_yticklabels(np.linspace(0,self.Ly//1000,11).astype(int))
            ax[i].set_ylabel('y [km]', fontsize=20)
       
        ax[0].set_xlim([0,self.Lx])
        ax[0].set_ylim([0,self.Ly])
    
        ax[1].set_xlabel('x [km]', fontsize=20)
        
    
        ax[0].set_title('Free Atmosphere', fontsize=20)
        ax[1].set_title('Boundary Layer', fontsize=20)
        
        plt.suptitle(f'Time :{hour:2d} hr {mint:2d} min {secd:2d} s ,'+' f=5 x 10$^{-5}$ s$^{-1}$', fontsize=25)
        #plt.savefig(f"E:/65h_P/SWE/SWE_{ts//12}")
        plt.savefig(path+f'/uvp_{ts//12}.png')
        #plt.show()
        plt.close()
    
    def plot_vor(self, u, v, u_sfc, v_sfc, ts, path):
        time = ts* self.dt
        hour = time//3600
        mint = (time-hour*3600)//60
        secd = time-hour*3600-mint*60
        i=time//60
        
        vor = np.real(ifft2(1j*self.kx*fft2(v))) - np.real(ifft2(1j*self.ky*fft2(u)))
        vor_sfc = np.real(ifft2(1j*self.kx*fft2(v_sfc))) - np.real(ifft2(1j*self.ky*fft2(u_sfc)))
        
        cmap=cm.jet
        cmap.set_under('#ffffff')
        lev = np.round(np.linspace(0.3,3,10),1)*(1e-3)
        
        fig, ax=plt.subplots(2,1, sharex='all', figsize=(13,24))

        CS = ax[0].contourf(self.X ,self.Y, vor, levels=lev, cmap=cmap, extend='both')
        ax[1].contourf(self.X ,self.Y, vor_sfc, levels=lev, cmap=cmap, extend='both')

        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cb = plt.colorbar(CS, ticks=lev, cax=cbar_ax, extend='both')
        cb.set_ticklabels(np.round(lev*(1e+3),1))
        cb.set_label('vorticity [s$^{-1}$]', fontsize=20)

        ax[1].set_xticks(np.arange(self.Lx//2-70e+3,self.Lx//2+70e+3,10e+3))
        ax[1].set_xticklabels(np.arange(self.Lx//2-70e+3,self.Lx//2+70e+3,10e+3).astype(int)//1000)
        for j in range(2):
            ax[j].set_yticks(np.arange(self.Ly//2-70e+3,self.Ly//2+70e+3,10e+3))
            ax[j].set_yticklabels(np.arange(self.Ly//2-70e+3,self.Ly//2+70e+3,10e+3).astype(int)//1000)
            ax[j].set_ylabel('y [km]', fontsize=20)
       
        ax[0].set_xlim([self.Lx//2-70e+3,self.Lx//2+70e+3])
        ax[0].set_ylim([self.Ly//2-70e+3,self.Ly//2+70e+3]); ax[1].set_ylim([self.Ly//2-70e+3,self.Ly//2+70e+3])
    
        ax[1].set_xlabel('x [km]', fontsize=20)


        ax[0].set_title('vorticity in the free atmosphere',fontsize=20)
        ax[1].set_title('vorticity in the boundary layer',fontsize=20)
        plt.suptitle(f'Time : {hour:2d} hr {mint:2d} min {secd:2d} s', fontsize=25)
        
        #plt.savefig(f"E:/65h_P/vor/vor_{ts//12}")
        plt.savefig(path+f'/vor_{ts//12}.png')
        plt.close()
    
    def Spatial_diff(self, f):
        
        f_hat = self.wave_filter(f)
        #f_hat=fft2(f)
        return (np.real(ifft2(1j*self.kx*f_hat)), np.real(ifft2(1j*self.ky*f_hat)))
        
        #return(self.I1.dot(f.T).T/(2*self.dx), self.I1.dot(f)/(2*self.dy))
    
    def Laplace(self, f):
        
        f_hat = self.wave_filter(f)
        #f_hat=fft2(f)
        return np.real(ifft2(-(self.k_squared) * f_hat))
        
        #return(self.I2.dot(f.T).T/(self.dx**2)+self.I2.dot(f)/(self.dy**2))
    
    def D_Laplace(self, f):
        
        f_hat = self.wave_filter(f)
        #f_hat=fft2(f)
        return np.real(ifft2(-(self.k_squared)**2 * f_hat))
        
        #return(self.I2.dot(f.T).T/(self.dx**2)+self.I2.dot(f)/(self.dy**2))
 
    # ============================================================
    # 修改 N_S_EQ（最耗時的函式，有很多逐元素四則運算）
    # ============================================================
    def N_S_EQ(self, wind, t, u_r, v_r, w, h, P):
        u, v = wind
        dPx, dPy = self.Spatial_diff(P)
        H = self.H

        # pressure gradient + Coriolis（合併成一個 ne.evaluate）
        PF_CF_u = ne.evaluate("-dPx/rho + f*v", local_dict={'dPx': dPx, 'rho': rho, 'f': f, 'v': v})
        PF_CF_v = ne.evaluate("-dPy/rho - f*u", local_dict={'dPy': dPy, 'rho': rho, 'f': f, 'u': u})

        # Advection
        w_p = ne.evaluate("where(w<0, -w, 0) + where(w>0, 0, 0)")   # -0.5*(|w|-w)
        w_p = ne.evaluate("0.5*(abs(w)-w)", local_dict={'w': w})      # 簡化
        dux, duy = self.Spatial_diff(u)
        dvx, dvy = self.Spatial_diff(v)
        ADV_u = ne.evaluate("u*dux + v*duy + w_p*(u_r-u)/H",
                            local_dict={'u':u,'dux':dux,'v':v,'duy':duy,'w_p':w_p,'u_r':u_r,'H':H})
        ADV_v = ne.evaluate("u*dvx + v*dvy + w_p*(v_r-v)/H",
                            local_dict={'u':u,'dvx':dvx,'v':v,'dvy':dvy,'w_p':w_p,'v_r':v_r,'H':H})

        # Viscosity
        VC_u = ne.evaluate("nu2*lap_u", local_dict={'nu2': self.nu2, 'lap_u': self.Laplace(u)})
        VC_v = ne.evaluate("nu2*lap_v", local_dict={'nu2': self.nu2, 'lap_v': self.Laplace(v)})

        # Friction (CD 有條件式，保留 numpy where，其餘用 ne)
        vel = ne.evaluate("0.78*sqrt(u**2+v**2)", local_dict={'u': u, 'v': v})
        CD = np.where((vel > 1e-5) & (vel <= 25),
                    (1e-3) * (2.7 / vel + 0.142 + 0.0764 * vel), 0)
        CD = np.where(vel > 25,
                    (1e-3) * (2.16 + 0.5406 * (1 - np.exp(-((vel - 25) / 7.5)))), CD)
        FR_u = ne.evaluate("CD*vel*u/H", local_dict={'CD':CD,'vel':vel,'u':u,'H':H})
        FR_v = ne.evaluate("CD*vel*v/H", local_dict={'CD':CD,'vel':vel,'v':v,'H':H})

        u_term = ne.evaluate("PF_CF_u - FR_u + VC_u - ADV_u",
                            local_dict={'PF_CF_u':PF_CF_u,'FR_u':FR_u,'VC_u':VC_u,'ADV_u':ADV_u})
        v_term = ne.evaluate("PF_CF_v - FR_v + VC_v - ADV_v",
                            local_dict={'PF_CF_v':PF_CF_v,'FR_v':FR_v,'VC_v':VC_v,'ADV_v':ADV_v})

        return np.array([u_term, v_term])
    

    # ============================================================
    # 修改 SWE
    # ============================================================
    def SWE(self, var, t, Q, w_sfc, u_sfc, v_sfc):
        u, v, h = var
        dhx, dhy = self.Spatial_diff(h)
        dux, duy = self.Spatial_diff(u)
        dvx, dvy = self.Spatial_diff(v)
        H = self.H

        w_p = ne.evaluate("0.5*(abs(w_sfc)+w_sfc)", local_dict={'w_sfc': w_sfc})

        u_term = ne.evaluate(
            "-g*dhx + f*v - u*dux - v*duy - w_p*(u-u_sfc)/H + nu1*lap_u",
            local_dict={'g':g,'dhx':dhx,'f':f,'v':v,'u':u,'dux':dux,'duy':duy,
                        'w_p':w_p,'u_sfc':u_sfc,'H':H,'nu1':self.nu1,'lap_u':self.Laplace(u)})
        v_term = ne.evaluate(
            "-g*dhy - f*u - u*dvx - v*dvy - w_p*(v-v_sfc)/H + nu1*lap_v",
            local_dict={'g':g,'dhy':dhy,'f':f,'u':u,'v':v,'dvx':dvx,'dvy':dvy,
                        'w_p':w_p,'v_sfc':v_sfc,'H':H,'nu1':self.nu1,'lap_v':self.Laplace(v)})
        h_term = ne.evaluate(
            "-(H+h)*(dux+dvy) - u*dhx - v*dhy - (H+h)*Q",
            local_dict={'H':H,'h':h,'dux':dux,'dvy':dvy,'u':u,'dhx':dhx,'v':v,'dhy':dhy,'Q':Q})

        return np.array([u_term, v_term, h_term])
    
    
    def RK4(self, func, y, t, *args):
        k1 = self.dt * func(y, t, *args)
        k2 = self.dt * func(y + 0.5 * k1, t + 0.5 * self.dt, *args)
        k3 = self.dt * func(y + 0.5 * k2, t + 0.5 * self.dt, *args)
        k4 = self.dt * func(y + k3, t + self.dt, *args)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    
    def damping(self, f):
        x0 = self.Lx/2; y0 = self.Ly/2
        #sigma_x = self.Lx/4.5; sigma_y = self.Ly/4.5
        #damp=np.where((self.X-x0)**2+(self.Y-y0)**2<85000**2,1, (np.sin(abs(self.X-x0)*np.pi/(self.Lx-x0))/(abs(self.X-x0)*np.pi/(self.Lx-x0)))*(np.sin(abs(self.Y-y0)*np.pi/(self.Ly-y0))/(abs(self.Y-y0)*np.pi/(self.Ly-y0))))
        
        
        damp = np.where(self.r<1, 1-np.exp(-80/self.r*np.exp(1/(self.r-1))), 0)
    
        f_damp = f*damp
        
        return (f_damp)

    def write_single_data(self, u, v, w, u_sfc, v_sfc, w_sfc, h, P, hour, t, path, name):
        f_w=nc.Dataset(path+name+'_'+str(t*self.dt//60).zfill(4)+'.nc','w',format='NETCDF4')

        f_w.createDimension('x',self.Nx)
        f_w.createDimension('y',self.Ny)

        f_w.createVariable('x',np.float32,('x'))
        f_w.createVariable('y',np.float32,('y'))
        f_w.createVariable('u',np.float32,('x','y'))
        f_w.createVariable('v',np.float32,('x','y'))
        f_w.createVariable('w',np.float32,('x','y'))
        f_w.createVariable('u_sfc',np.float32,('x','y'))
        f_w.createVariable('v_sfc',np.float32,('x','y'))
        f_w.createVariable('w_sfc',np.float32,('x','y'))
        f_w.createVariable('h',np.float32,('x','y'))
        f_w.createVariable('P',np.float32,('x','y'))

        f_w.variables['x'][:]=self.x
        f_w.variables['y'][:]=self.y
        f_w.variables['u'][:]=u
        f_w.variables['v'][:]=v
        f_w.variables['w'][:]=w
        f_w.variables['u_sfc'][:]=u_sfc
        f_w.variables['v_sfc'][:]=v_sfc
        f_w.variables['w_sfc'][:]=w_sfc
        f_w.variables['h'][:]=h
        f_w.variables['P'][:]=P

        f_w.description=f'Total time: {hour} hours'+\
                        f', at {t* self.dt//60} mins'+\
                        f', nu of free ATM={self.nu1}, nu of BL={self.nu2}'

        f_w.close()
    
    
    def write_data(self, u, v, w, u_sfc, v_sfc, w_sfc, h, P, hour, path, name):
        Nt = 2*hour+1
        f_w=nc.Dataset(path+name+'.nc','w',format='NETCDF4')
    
        f_w.createDimension('time',Nt)
        f_w.createDimension('x',self.Nx)
        f_w.createDimension('y',self.Ny)
        
        f_w.createVariable('x',np.float32,('x'))
        f_w.createVariable('y',np.float32,('y'))
        f_w.createVariable('u',np.float32,('time','x','y'))
        f_w.createVariable('v',np.float32,('time','x','y'))
        f_w.createVariable('w',np.float32,('time','x','y'))
        f_w.createVariable('u_sfc',np.float32,('time','x','y'))
        f_w.createVariable('v_sfc',np.float32,('time','x','y'))
        f_w.createVariable('w_sfc',np.float32,('time','x','y'))
        f_w.createVariable('h',np.float32,('time','x','y'))
        f_w.createVariable('P',np.float32,('time','x','y'))
    
        f_w.variables['x'][:]=self.x
        f_w.variables['y'][:]=self.y
        f_w.variables['u'][:]=u
        f_w.variables['v'][:]=v
        f_w.variables['w'][:]=w
        f_w.variables['u_sfc'][:]=u_sfc
        f_w.variables['v_sfc'][:]=v_sfc
        f_w.variables['w_sfc'][:]=w_sfc
        f_w.variables['h'][:]=h
        f_w.variables['P'][:]=P
        
        f_w.description=f'Total time: {hour} hours'+f', nu of free ATM={self.nu1}, nu of BL={self.nu2}'
        
        f_w.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
