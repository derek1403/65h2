import numpy as np
#import netCDF4 as nc
import time
from setting import *
from SWE_func2 import SWE_functions


ttl_s = hours* 3600
ttl_output = str(ttl_s//OT_data + 1)

print('=========================================================================')
print('Model: '+mdl_name[mm]+'\n-')
print(f'Total Simulation Time: {hours:02d} hrs')
print(f'Time Step            : {dt} s')
print(f'Spin-up Time         : {SP:.1f} hours')
print(f'Data Output Interval : {OT_data} s ')
print(f'Total Output Data    : {ttl_output} files')
print('=========================================================================')


st=time.time()

fun = SWE_functions(Lx, Ly, Nx, Ny, dt, H, nu1, nu2)

# free atmos.
u, v, P = fun.ini_wind(zeta)
h = np.where(P<0, 0, P/(rho*g))

#BL
u_sfc, v_sfc = u, v; w_sfc = np.zeros(X.shape)

#fun.plot_vor(u, v, u_sfc, v_sfc, 0)


#%%
#==============================================================================
# Time iteration
#==============================================================================
Q = np.zeros_like(h)
fun.write_single_data(u, v, np.zeros_like(h), u_sfc, v_sfc, w_sfc, h, P, hours, 0, data_path, data_name)
if plot_data:
    fun.plot_uvp(u, v, h, u_sfc, v_sfc, P, 0, plot_uvp_path)
    fun.plot_vor(u, v, u_sfc, v_sfc, 0, plot_vor_path)
    

et=time.time()
i = 1
print(f"{i} of {ttl_output} files completed.")
print(f'Simulation Time: {((i-1)*OT_data)//3600:02d} hr(s) {(((i-1)*OT_data)%3600)//60:02d} min(s) {((i-1)*OT_data)%60:02d} second(s)')
print(f'RunTime: {round(et-st,2)}  seconds.')
print('-------------------------------------------------------------------------')


t_start = 0 #24* 3600//dt
for t in range(t_start+1,timesteps+1):
    h_pre=h; 

    ## spin-up
    if t*dt <= 3600*SP:   
      Q = np.zeros_like(h)
      w_in = 0; u_in = 0; v_in = 0
    else:
      Q = w_sfc* Q0
      w_in = w_sfc; u_in = u_sfc; v_in = v_sfc

    ## different models in free atmos.
    if dname == mdl[0]:
      u, v, h = fun.RK4(fun.SWE, np.array([u,v,h]), t, 0, 0, 0, 0)             # one-way
    elif dname == mdl[1]:
      u, v, h = fun.RK4(fun.SWE, np.array([u,v,h]), t, Q, 0, 0, 0)             # mass sink
    elif dname == mdl[2]:
      u, v, h = fun.RK4(fun.SWE, np.array([u,v,h]), t, 0, w_in, u_in, v_in)    # momentum flux

    ## calculate vertical velocity in free atmos.
    w = (h-h_pre)/dt
    h = np.where(h>0, 0, h)
    
    ## BL
    P = rho*g*h
    u_sfc, v_sfc = fun.RK4(fun.N_S_EQ, np.array([u_sfc, v_sfc]), t, u, v, w_sfc, h, P)
    w_sfc=-H*(fun.Spatial_diff(u_sfc)[0]+fun.Spatial_diff(v_sfc)[1])
    
    ## output data
    if t*dt % OT_data == 0:
        #fun.plot(u, v, h, u_sfc, v_sfc, P, t)
        #fun.plot_vor(u, v, u_sfc, v_sfc, t)
        fun.write_single_data(u, v, w, u_sfc, v_sfc, w_sfc, h, P, hours, t, data_path, data_name)
        
        i += 1                 
        et=time.time()
        print(f"{i} of {ttl_output} files completed.")
        print(f'Simulation Time: {((i-1)*OT_data)//3600:02d} hr(s) {(((i-1)*OT_data)%3600)//60:02d} min(s) {((i-1)*OT_data)%60:02d} second(s)')
        print(f'RunTime: {round(et-st,2)}  seconds.')
        print('-------------------------------------------------------------------------')

    if plot_data and t*dt % OT_plot == 0:
        fun.plot_uvp(u, v, h, u_sfc, v_sfc, P, t, plot_uvp_path)
        fun.plot_vor(u, v, u_sfc, v_sfc, t, plot_vor_path)


    if True in np.isnan(u_sfc):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f'Simulation blows up at t = {t*dt} s.')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')      
        break

#fun.write_data(u_data, v_data, w_data, u_sfc_data, v_sfc_data, w_sfc_data, h_data, P_data, hours, pic_path, data_name)
    
et=time.time()
print(et-st)

