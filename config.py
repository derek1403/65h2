# config.py — 純參數，不做任何運算

# Domain
Lx = 700000; Ly = 700000
Nx = 512;    Ny = 512

# Time
hours     = 36
dt        = 5
timesteps = hours * 3600 // dt
SP        = 3.       # spin-up time (hrs)

# Physical constants
g   = 9.81
rho = 1.
f   = 5e-5 

# Model parameters
H   = 1000
nu1 = 100
nu2 = 5000
Q0  = 2e-5

# FFT truncation
mx = 256
my = 256

# Output
OT_data   = 1800   # data output interval (s)
plot_data = True
OT_plot   = 600    # plot output interval (s)

# Model selection
mm       = 2
mdl      = ['OW', 'MS', 'MF']
mdl_name = ['One-way', 'Mass sink', 'Momentum flux']