#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cavity flow solver with FDM. Pressure regularization and fractional steps 
implemented for the solution of the scheme.
The non dimensional form of the Navier-Stokes equations is used in this solver 
for simplicity purposes
Created on Fri May 18 10:08:05 2018
@author: Antonio Preziosi-Ribero
"""

import numpy as np
import scipy.sparse as sp
import Mat_ass as ma
import Press_reg as pr

# ==============================================================================
# DEFINITION OF PHYSICAL VARIABLES OF THE PROBLEM - MODIFIABLE PART OF THE CODE
# ==============================================================================

# Physical description of the domain

X0 = 0                              # Inital x coordinate [m]
XF = 1                              # Final x coordinate [m]
Y0 = 0                              # Initial y coordinate [m]
YF = 1                              # Final y coordinate [m]
t0 = 0                              # Initial time [s]
tF = 5                              # Final time of the simulation [s]
rho = 1000                          # Fluid density [kg /m3] 
nu = 1e-6                           # Kinematic viscosity [m2/s]
Re = 50                             # Reynolds number of the flow [-]

# ==============================================================================
# NUMERICAL PARAMETERS OF THE MODEL - MODIFIABLE PART OF THE CODE
# ==============================================================================

Nx = 10                             # Nodes in the x direction
Ny = 10                             # Nodes in the y direction
CFL = 1.0                           # Maximum CFL allowed for the simulation
nlt = 0                             # Non linear term treatment (view funct)
dift = 0                            # First derivative precision (view funct)
Der2 = 0                            # Second derivative precision (view funct)

# ==============================================================================
# BOUNDARY CONDITIONS OF THE PROBLEM - MODIFIABLE PART OF THE CODE
# VARIABLES STORED IN THE FOLLOWING ORDER: BOTTOM TOP LEFT RIGHT
# ==============================================================================

# Type of boundary condition. 0 = Dirichlet, 1 = Neumann
BC0_u = np.array([0, 0, 0, 0])      # Type of boundary condition for u
BC0_v = np.array([0, 0, 0, 0])      # Type of boundary condition for v
BC0_p = np.array([1, 1, 1, 1])      # Type of boundary condition for pressure

# Value of boundary condition
BC1_u = np.array([0, 1, 0, 0])      # Value of boundary condition in u
BC1_v = np.array([0, 0, 0, 0])      # Value of boundary condition in v
BC1_p = np.array([0, 0, 0, 0])      # Value of boundary condition for pressure

# ==============================================================================
# START OF THE PROGRAM - INITIAL CALCULATIONS. ASSEMBLYING VECTORS FOR SPACE, 
# TIME, ERROR AND OTHER STUFF
# ==============================================================================

# Calculations made with the physical variables to non-dimentionalize the N-S
# equations (no need to change things here when running the program)

h = XF - X0                         # Characteristic length [m]
U = (Re * nu) / h                   # Lid velocity [m/s]
XS0 = X0 / h                        # Initial X coordinate in non dim form [-]
XSF = XF / h                        # Final X coordinate in non dim form [-]
YS0 = Y0 / h                        # Initial Y coordinate in non dim form [-]
YSF = YF / h                        # Final Y coordinate in non dim form [-]
ts0 = t0 * U / h                    # Initial non dimensional time [-]
tsF = tF * U / h                    # Final non dimensional time [-]

# Calculations made with non dimensionalized and numerical variables. Vectors of
# space, time and all of the things to make the model work

nn = Nx * Ny                        # Number of nodes in the domain

xn = np.linspace(XS0, XSF, Nx)      # Spatial vector for the x axis
yn = np.linspace(YS0, YSF, Ny)      # Spatial vector for the y axis

[X, Y] = np.meshgrid(xn, yn)        # Generating meshgrid for plotting

dx = np.absolute(xn[1] - xn[0])     # Cell size in x direction
dy = np.absolute(yn[1] - yn[0])     # Cell size in y direction
maxd = np.maximum(dx, dy)           # Maximum cell size for timestep calculation

tT = tsF - ts0                      # Total time of the simulation
dT = (CFL * maxd) / U               # Timestep size in non dim form [-]
nT = int(np.ceil(tT / dT))          # Number of timesteps of the model             
del(maxd)                           # Deleting useless variable 

Sx = dT / Re                        # Value for viscous term matrix

# Declaring variables that will store temporal values, differentiation matrices
# and other necessary values
K_x = sp.lil_matrix((nn, nn))       # Viscous term diff matrix for u
K_y = sp.lil_matrix((nn, nn))       # Viscous term diff matrix for v
P_m = sp.lil_matrix((nn, nn))       # Laplacian matrix for pressure

# Assemblying matrix with different boundary conditions
K_x = ma.Assembly(K_x, Nx, Ny, Sx, Sx, BC0_u, Der2)
K_y = ma.Assembly(K_y, Nx, Ny, Sx, Sx, BC0_v, Der2)
P_m = ma.Assembly(P_m, Nx, Ny, 1, 1, BC0_p, Der2)

# Converting matrices to CSR format for faster calculations - just for practical 
# ends. It does not affect computations
K_x = K_x.tocsr()
K_y = K_y.tocsr()
P_m = P_m.tocsr()

# Performing pressure regularization via external function. See external 
# function for further information
qqt = pr.Regularization(P_m)

# ==============================================================================
# INITIAL CONIDTION FOR THE CAVITY PROBLEM - USUALLY IT IS A U = 0 AND V = 0
# ==============================================================================

