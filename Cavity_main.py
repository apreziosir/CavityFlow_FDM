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

# Python libraries imported to do calculations, plots and stuff
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

# Self made modules to perform specifical operations of the program
import Mat_ass as ma
import Press_reg as pr
import Initial_cond as ic
import Boundaries as bnd
import First_der as fd

# ==============================================================================
# DEFINITION OF PHYSICAL VARIABLES OF THE PROBLEM - MODIFIABLE PART OF THE CODE
# ==============================================================================

X0 = 0.                             # Inital x coordinate [m]
XF = 1.                             # Final x coordinate [m]
Y0 = 0.                             # Initial y coordinate [m]
YF = 1.                             # Final y coordinate [m]
t0 = 0.                             # Initial time [s]
tF = 100000.                        # Final time of the simulation [s]
rho = 1000.                         # Fluid density [kg /m3] 
nu = 1e-6                           # Kinematic viscosity [m2/s]
Re = 100.                           # Reynolds number of the flow [-]

# ==============================================================================
# NUMERICAL PARAMETERS OF THE MODEL - MODIFIABLE PART OF THE CODE
# ==============================================================================

Nx = 51                             # Nodes in the x direction (use odd, please)
Ny = 51                             # Nodes in the y direction (use odd, please)
CFL = 0.75                          # Non dimensional timestep size [-]
nlt = 0                             # Non linear term treatment (view t. loop)
dift = 1                            # First derivative precision (view funct)
Der2 = 1                            # Second derivative precision (view funct)
reg = 1                             # P. regularization (0 = False, 1 = True)
tn = Nx                             # Tied node when pressure is not regularized

# ==============================================================================
# BOUNDARY CONDITIONS OF THE PROBLEM - MODIFIABLE PART OF THE CODE
# VARIABLES STORED IN THE FOLLOWING ORDER: BOTTOM TOP LEFT RIGHT
# ==============================================================================

# Type of boundary condition. 0 = Dirichlet, 1 = Neumann
BC0_u = np.array([0, 0, 0, 0])      # Type of boundary condition for u
BC0_v = np.array([0, 0, 0, 0])      # Type of boundary condition for v
BC0_p = np.array([1, 1, 1, 1])      # Type of BC for pressure

# Value of boundary condition (this has to be declared as float not as int)
BC1_u = np.array([0., 1., 0., 0.])  # Value of boundary condition in u
BC1_v = np.array([0., 0., 0., 0.])  # Value of boundary condition in v
BC1_p = np.array([0., 0., 0., 0.])  # Value of BC for pressure

# ==============================================================================
# START OF THE PROGRAM - INITIAL CALCULATIONS. ASSEMBLYING VECTORS FOR SPACE, 
# TIME, ERROR AND OTHER STUFF FOR DIFFERENT PURPOSES
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

# Looking for nodes that are part of boundaries and ring and saving those posi-
# tions in the main part of the program
[B_B, B_R] = bnd.Bot_BR(Nx, Ny)     # Bottom boundary and ring
[T_B, T_R] = bnd.Top_BR(Nx, Ny)     # Top boundary and ring
[L_B, L_R] = bnd.Left_BR(Nx, Ny)    # Left boundary and ring
[R_B, R_R] = bnd.Right_BR(Nx, Ny)   # Right boundary and ring

dx = np.absolute(xn[1] - xn[0])     # Cell size in x direction
dy = np.absolute(yn[1] - yn[0])     # Cell size in y direction
maxd = np.maximum(dx, dy)           # Maximum cell size for timestep calculation

tT = tsF - ts0                      # Total time of the simulation
dT = CFL * maxd                     # Timestep size in dim form [s]
nT = int(np.ceil(tT / dT))          # Number of timesteps of the model             
del(maxd)                           # Deleting useless variable 

Sx_x = dT / (Re * dx ** 2)          # Value for viscous term matrix in x
Sx_y = dT / (Re * dy ** 2)          # Value for viscous term matrix in y
Sp_x = 1 / (dx ** 2)                # Value for pressure matrix in x
Sp_y = 1 / (dy ** 2)                # Value for pressure matrix in y

# Printing interesting results for the simulation - just to check if the 
# set up is correct. 
print('#######################################################################')
print('Maximum CFL for simulation: ', CFL)
print('Lid velocity [m/s]', U)
print('Stability parameter for viscous term: ', Sx_x, Sx_y)
print('Number of time steps for the simulation: ', nT)
print('Timestep size for the simulation: ', dT)
print('#######################################################################')
print('\n')

# Selecting nodes in the centerlines (vertical and horizontal of the domain)
# to plot velocity profiles
n0x = Nx * ((Ny - 1) / 2)           # First node of center line x
n1x = n0x + (Nx - 1)                # Last node of center line x
clx = np.linspace(n0x, n1x, Nx)     # Array with nodes that mark horizontal CL
clx = clx.astype(int)

n0y = (Nx - 1) / 2                  # First node of center line y
n1y = n0y + Nx * (Ny - 1)           # Last node of center line y
cly = np.linspace(n0y, n1y, Ny)     # Array with nodes that mark vertical CL
cly = cly.astype(int)

xr = np.linspace(0, 1, Nx)          # Vector to plot velocities along CL
yr = np.linspace(0, 1, Ny)          # Vector to plot velocities along CL

# Declaring variables that will store temporal values, differentiation matrices
# and other necessary values
K_x = sp.lil_matrix((nn, nn))       # Viscous term diff matrix for u
K_y = sp.lil_matrix((nn, nn))       # Viscous term diff matrix for v
P_m = sp.lil_matrix((nn, nn))       # Laplacian matrix for pressure

# Assemblying matrix with different boundary conditions
K_x = ma.Assembly(K_x, Nx, Ny, Sx_x, Sx_y, BC0_u, Der2)
K_y = ma.Assembly(K_y, Nx, Ny, Sx_x, Sx_y, BC0_v, Der2)
P_m = ma.Assembly(P_m, Nx, Ny, Sp_x, Sp_y, BC0_p, Der2)

if reg == 0:
    P_m[tn, :] = np.zeros(nn)
    P_m[tn, tn] = 1.

# Converting matrices to CSR format for faster calculations - just for practical 
# ends. It does not affect computations
K_x = K_x.tocsr()
K_y = K_y.tocsr()
P_m = P_m.tocsr()

# Performing pressure regularization via external function. See external 
# function for further information (this is goingo to be used in the temporal 
# loop of the program to regularize pressure with Neumann BC) 
if reg == 1 : qqt = pr.Regularization(P_m)
else : qqt = 1.0

# ==============================================================================
# VALUES OF GHIA ET AL. 1986 FOR COMPARING (just for plotting part)
# Works when Re = 100 (JUST WHEN REYNOLDS = 100!!!!)
# ==============================================================================

# Coordinates alosng center lines
Gx = np.array([1., 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, \
               0.500, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, \
               0.])
Gy = np.array([1., 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, \
               0.500, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, \
               0.])

# Velocities along center lines
Gu = np.array([1., 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, \
               -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, \
               -0.04775, -0.04192, -0.03717, 0.])
Gv = np.array([0., -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, \
               -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, \
               0.10091, 0.09233, 0.])

# ==============================================================================
# INITIAL CONIDTION FOR THE CAVITY PROBLEM - USUALLY IT IS A U = 0 AND V = 0
# ==============================================================================

# Setting initial conidtion
[U0, V0] = ic.Init_cond(X, Y)

# Plotting initial condition
plt.ion()
style.use('ggplot')
plt.figure(1)

fig1 = plt.subplot(1, 2, 1)
surf1 = fig1.contourf(X, Y, U0, cmap=cm.coolwarm, antialiased=False)
fig1.set_title('Initial condition for U (m/s)')
fig1.set_xlabel('X coordinate')
fig1.set_ylabel('Y coordinate')
fig1.tick_params(axis='both', which='major', labelsize=6)

fig2 = plt.subplot(1, 2, 2)
surf1 = fig2.contourf(X, Y, V0, cmap=cm.coolwarm, antialiased=False)
fig2.set_title('Initial condition for V (m/s)')
fig2.set_xlabel('X coordinate')
fig2.set_ylabel('Y coordinate')
fig2.tick_params(axis='both', which='major', labelsize=6)

plt.draw()
plt.pause(2)
plt.close()

# Pause command to check printed messages and go on with simulation
#input()

# ==============================================================================
# TIME LOOP OF THE PROGRAM - 
# ==============================================================================

plt.ion()
style.use('ggplot')
plt.figure(1, figsize=(11.5, 11.5))

#for t in range(1, 20):
for t in range(1, nT + 1):
    
# ==============================================================================    
    # Calculating the non linear term with forward Euler explicit scheme
# ==============================================================================
    # Primitive variables 
    if nlt == 0:
        
        # reshaping vectors for dimensional conformity
        U0 = U0.reshape((nn, 1))
        V0 = V0.reshape((nn, 1))
        
        # Calculating non linear term with primitive variables for u and v 
        # velocities
        
        tempu = np.multiply(U0, fd.diffx(U0, dx, L_B, L_R, R_B, R_R, dift)) \
                 + np.multiply(V0, fd.diffy(U0, dy, B_B, B_R, T_B, T_R, dift))
                 
        Ug = U0 - dT * tempu
        
        tempv = np.multiply(U0, fd.diffx(V0, dx, L_B, L_R, R_B, R_R, dift)) \
                + np.multiply(V0, fd.diffy(V0, dy, B_B, B_R, T_B, T_R, dift))
        
        Vg = V0 - dT * tempv
    
    # Divergence form - only appliable to  terms with same vector, the other is
    # still treated as a primitive variable    
    elif nlt == 1:
        
        # reshaping vectors for dimensional conformity
        U0 = U0.reshape((nn, 1))
        V0 = V0.reshape((nn, 1))
        
        # Temporal variables that store the square of the velocity vector u or v
        ugt = U0 ** 2
        vgt = V0 ** 2
        
        Ug = U0 - dT * (0.5 * fd.diffx(ugt, dx, L_B, L_R, R_B, R_R, dift) + \
                        np.multiply(V0, fd.diffy(U0, dy, B_B, B_R, T_B, T_R,\
                        dift)))
        
        Vg = V0 - dT * (0.5 * fd.diffy(vgt, dy, B_B, B_R, T_B, T_R, dift) + \
                        np.multiply(U0, fd.diffx(V0, dx, L_B, L_R, R_B, R_R,\
                        dift)))
                     
    # Error message for wrong choice of non linear term treatment    
    else:
        
        print('Wrong choice of non linear term treatment. Please check the main\
               part of the code and make a good selection')
        break
    
    # Imposing boundary conditions (using same boundary conditions of the origi-
    # nal problem) - implemented after any of the chosen schemes for the non 
    # linear terms
    
    # Horizontal (U) velocity
    Ug[B_B] = np.ones((Nx, 1)) * BC1_u[0]
    Ug[T_B] = np.ones((Nx, 1)) * BC1_u[1]
    Ug[L_B] = np.ones((Nx - 2, 1)) * BC1_u[2]
    Ug[R_B] = np.ones((Nx - 2, 1)) * BC1_u[3]
    
    # Vertical (V) velocity
    Vg[B_B] = np.ones((Nx, 1)) * BC1_v[0]
    Vg[T_B] = np.ones((Nx, 1)) * BC1_v[1]
    Vg[L_B] = np.ones((Nx - 2, 1)) * BC1_v[2]
    Vg[R_B] = np.ones((Nx - 2, 1)) * BC1_v[3]
    
#    # Checking values of hat velocities
#    print(Ug[T_B])
#    print(Vg[T_B])    
    
# ==============================================================================
    # Entering to the pressure calculation
# ==============================================================================
    
    # Taking the divergence of the hat velocity field - it is not 0 necessarly
    divug = fd.diffx(Ug, dx, L_B, L_R, R_B, R_R, dift) + fd.diffy(Vg, dy, B_B,\
                    B_R, T_B, T_R, dift)
    
    # Multiplying the result by -1/dT
    divug *= (-1 / dT)
    
#    print('U hat divergence times -1/dT: ') # Printing statements for checking
#    print(divug)                            # Printing statements for checking
    
    # Applying PRESSURE REGULARIZATION before solving the Laplacian for the 
    # pressure
    if reg == 0 : RHS_p = divug
    else : RHS_p = divug - np.matmul(qqt, divug)
    
#    print('Vector de valores RHS: ')        # Printing statements for checking
#    print(RHS_p)                            # Printing statements for checking
    
    # Pressure solver (meat part of the code) - This is a normal Poisson solver
    # that will find a pressure field (remember that this is a Pressure field 
    # that satisfies conditions, but it may be not the REAL pressure field)
    
    # Imposing boundary conditions - Homogeneous Neumann BC for pressure. 
    # The regularization should make the system non singular
    divug[B_B] = np.ones((Nx, 1)) * BC1_p[0]
    divug[T_B] = np.ones((Nx, 1)) * BC1_p[1]
    divug[L_B] = np.ones((Ny - 2, 1)) * BC1_p[2]
    divug[R_B] = np.ones((Ny - 2, 1)) * BC1_p[3]
    
    # Imposing boundary condition when pressure is not regularized
    if reg == 0 : divug[tn] = 1
        
    # Solving the system
    P1 = spsolve(P_m, divug)
    
    # Calculating u double hat with pressure
    Ugg = Ug - dT * fd.diffx(P1, dx, L_B, L_R, R_B, R_R, dift)
    Vgg = Vg - dT * fd.diffy(P1, dy, B_B, B_R, T_B, T_R, dift)
    
    # Checking incompressibility of field hat hat div(Uhh, Vhh)
    # Fill later
    
# ==============================================================================
    # Calculating the viscous term part
# ==============================================================================
    
    # Imposing boundary conditions for solving the viscous term
    # Boundary conditions for u velocity
    Ugg[B_B] = np.ones((Nx, 1)) * BC1_u[0]
    Ugg[T_B] = np.ones((Nx, 1)) * BC1_u[1]
    Ugg[L_B] = np.ones((Ny - 2, 1)) * BC1_u[2]
    Ugg[R_B] = np.ones((Ny - 2, 1)) * BC1_u[3]
    
    # Boundary conditions for v velocity
    Vgg[B_B] = np.ones((Nx, 1)) * BC1_v[0]
    Vgg[T_B] = np.ones((Nx, 1)) * BC1_v[1]
    Vgg[L_B] = np.ones((Ny - 2, 1)) * BC1_v[2]
    Vgg[R_B] = np.ones((Ny - 2, 1)) * BC1_v[3]
    
    # Calculating the diffusive term - linear system
    U1 = spsolve(K_x, Ugg)    
    V1 = spsolve(K_y, Vgg)
    
    # Checking flow incompressibility
    Cons_m = fd.diffx(U1, dx, L_B, L_R, R_B, R_R, dift) + fd.diffy(V1, dy, B_B, \
                    B_R, T_B, T_R, dift)
    
    # Calculating velocity magnitude
    mag = np.multiply(U1, V1)
    
    print('Mass conservation in each node (should be 0): ')
    print(Cons_m)
    
# ==============================================================================
    # Plotting velocities u and v, magnitude and pressure
# ==============================================================================
    
    # Plotting the solution
    plt.clf()
    
    fig1 = plt.subplot(2, 3, 1)
    surf1 = fig1.contourf(X, Y, U1.reshape((Nx, Ny)), cmap=cm.coolwarm)
    fig1.set_xlim([X0, XF])
    fig1.set_ylim([Y0, YF])
    fig1.set_xlabel(r'x axis')
    fig1.set_ylabel(r'y axis')
    fig1.tick_params(axis='both', which='major', labelsize=6)
    fig1.set_title('Numerical u velocity')
    
    fig2 = plt.subplot(2, 3, 2)
    surf2 = fig2.contourf(X, Y, V1.reshape((Nx, Ny)), cmap=cm.coolwarm)
    fig2.set_xlim([X0, XF])
    fig2.set_ylim([Y0, YF])
    fig2.set_xlabel(r'x axis')
    fig2.set_ylabel(r'y axis')
    fig2.tick_params(axis='both', which='major', labelsize=6)
    fig2.set_title('Numerical v velocity')
    
    fig3 = plt.subplot(2, 3, 3)
    surf3 = fig3.contourf(X, Y, mag.reshape((Nx, Ny)), cmap=cm.coolwarm)
    fig3.set_xlim([X0, XF])
    fig3.set_ylim([Y0, YF])
    fig3.set_xlabel(r'x axis')
    fig3.set_ylabel(r'y axis')
    fig3.tick_params(axis='both', which='major', labelsize=6)
    fig3.set_title('Velocity Magnitude')
    
    fig4 = plt.subplot(2, 3, 4, projection='3d')
    surf4 = fig4.plot_surface(X, Y, P1.reshape((Nx, Ny)), rstride=1, cstride=1,\
                              linewidth=0, cmap=cm.coolwarm, antialiased=False)
    fig4.set_xlim([X0, XF])
    fig4.set_ylim([Y0, YF])
    fig4.tick_params(axis='both', which='major', labelsize=6)
    fig4.set_xlabel(r'x axis')
    fig4.set_ylabel(r'y axis')
    fig4.set_title('Pressure field')
    
    fig5 = plt.subplot(2, 3, 5)
    surf5 = fig5.plot(U1[cly], yr)
    surf6 = fig5.scatter(Gu, Gy, 6, 'b')
    fig5.set_xlim([-0.25, 1.05])
    fig5.set_ylim([0, 1])
    fig5.tick_params(axis='both', which='major', labelsize=6)
    fig5.set_xlabel(r'u velocity')
    fig5.set_ylabel(r'y axis')
    fig5.set_title('U representative')
    
    fig5 = plt.subplot(2, 3, 6)
    surf5 = fig5.plot(xr,V1[clx])
    surf6 = fig5.scatter(Gx, Gv, 6, 'b')
    fig5.set_xlim([0, 1])
    fig5.set_ylim([-0.3, 0.2])
    fig5.tick_params(axis='both', which='major', labelsize=6)
    fig5.set_xlabel(r'x axis')
    fig5.set_ylabel(r'v velocity')
    fig5.set_title('V representative')
    
    plt.pause(0.01)
    
# ==============================================================================
    # Updating values for next step
# ==============================================================================
    
    U0 = U1
    V0 = V1
    