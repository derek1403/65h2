import numpy as np
import netCDF4 as nc
import os

#==============================================================================
# Domain grid & Constant
#==============================================================================
Lx = 700000; Ly = 700000; Nx=512; Ny=512    # Domain size
x = np.linspace(0,Lx,Nx); dx = x[1]-x[0]  
y = np.linspace(0,Ly,Ny); dy = y[1]-y[0]
X, Y = np.meshgrid(x,y)

hours = 36 # simulation time
dt = 5
timesteps = hours* 3600 // dt
SP = 3.  # spin-up time (hrs)

# parameters
g = 9.81; rho = 1.

H = 1000;   # water depth
nu1 = 100;  # Viscosity coefficient in free atmosphere
nu2 = 5000  # Viscosity coefficient in boundary layer

Q0 = 2e-5   # mass flux


#==============================================================================
# Initial condition
#==============================================================================
def fun_S(s):
    return ( 1-3*s**2+2*s**3)

# maximum vorticity
zeta0 = 2e-3

# vortex center
x0 ,y0 = Lx//2, Ly//2

# ellipse vortex
a1, b1 = 20e+3, 40e+3   # minor & major axis
a2, b2 = 24e+3, 44e+3   # 4 km 

r1 = np.sqrt( ((X-x0)/a1)**2 + ((Y-y0)/b1)**2 )
r2 = np.sqrt( ((X-x0)/a2)**2 + ((Y-y0)/b2)**2 )

zeta = np.where(r1<1, zeta0, 0)
zeta = np.where((r1>=1)&(r2<=1), zeta0*fun_S((1-r1)/(r2-r1)) - 1e-6*fun_S((r2-1)/(r2-r1)), zeta)
zeta = np.where(r2>1, 0, zeta)


#==============================================================================
# Models and file path, name
#==============================================================================
mm = 2
mdl = ['OW', 'MS', 'MF']; mdl_name = ['One-way', 'Mass sink', 'Momentum flux']
dname = mdl[mm]

# output data path, name
OT_data = 1800  # data output interval (seconds)
data_path = 'data/'; os.makedirs(data_path, exist_ok=True)
data_name = 'elps_'+dname

# plot path
plot_data = True
if plot_data:
    OT_plot = 600 # plot data interval (seconds)
    plot_uvp_path = 'plot/uvp'; os.makedirs(plot_uvp_path, exist_ok=True)
    plot_vor_path = 'plot/vor/'; os.makedirs(plot_vor_path, exist_ok=True)





'''
## upper layer
#u, v, P = fun.ini_wind(zeta)
#h = np.where(P<0, 0, P/(rho*g))
data = nc.Dataset('/data2/65h/data_MF/data24/ring_MF_1440.nc')
u = data.variables['u'][:]
v = data.variables['v'][:]
w = data.variables['w'][:]
h = data.variables['h'][:]

## surface
#u_sfc, v_sfc = u, v; w_sfc = np.zeros(X.shape)
#P = rho* g* h
u_sfc = data.variables['u_sfc'][:]
v_sfc = data.variables['v_sfc'][:]
w_sfc = data.variables['w_sfc'][:]
P = data.variables['P'][:]

data.close()
'''
