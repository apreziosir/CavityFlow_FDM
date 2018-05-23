#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:49:54 2018
Function that assembles matrices for diffusion problem. Only number of nodes,
dx, constant matrix multipliers and boundary conidtions are introduced. 
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""

import numpy as np
import Boundaries as bnd

# ==============================================================================
# Diffusion matrices assembly. 
# If Der2 == 0 - order of second derivative approximation = 2
# If Der2 == 1 - order of second derivative approximation = 4
# ==============================================================================

def Assembly(K, Nx, Ny, Sx, Sy, BC0, Der2):
    
    # Defining matrix according to number of nodes in the domain
    nn = Nx * Ny
    
    # Looking for boundary nodes in the domain according to Nx and Ny
    [B_B, B_R] = bnd.Bot_BR(Nx, Ny)
    [T_B, T_R] = bnd.Top_BR(Nx, Ny)
    [L_B, L_R] = bnd.Left_BR(Nx, Ny)
    [R_B, R_R] = bnd.Right_BR(Nx, Ny)    
            
    # Filling matrix with typical nodal values (then the boundary nodes are 
    # replaced with rows equal to 0 and changed) - this works for both implemen-
    # tations of the second derivative
    
    if Der2 == 0:
        
        for i in range(int(B_R[0]), int(T_B[0])):
            
            # Second order second derivative for the matrix
            K[i, i] = 1 + 2 * Sx + 2 * Sy
            K[i, i + 1] = -Sx
            K[i, i - 1] = -Sx
            K[i, i + Nx] = -Sy
            K[i, i - Nx] = -Sy
            
    elif Der2 == 1:
        
        # Coefficients of matrix change in higher order
        Sx1 = Sx / 12
        Sy1 = Sy / 12
        
        for i in range(L_B[1] + 1, R_B[-2] - 1):
            
            # Fourth order derivative for the viscous calculation
            K[i, i] = 1 + 30 * Sx1 + 30 * Sy1
            K[i, i + 1] = -16 * Sx1
            K[i, i - 1] = -16 * Sx1
            K[i, i + Nx] = -16 * Sy1
            K[i, i - Nx] = -16 * Sy1
            K[i, i + 2] = Sx1
            K[i, i - 2] = Sx1
            K[i, i + 2 * Nx] = Sy1
            K[i, i - 2 * Nx] = Sy1           
            
        # Setting the value of the rows that represent external ring to 0
        K[B_R, :] = np.zeros(nn)
        K[T_R, :] = np.zeros(nn)
        K[R_R, :] = np.zeros(nn)
        K[L_R, :] = np.zeros(nn)
        
        # Setting the value of the matrix elements in the external ring
        K[B_R, B_R] = 1 + 2 * Sx + 2 * Sy
        K[B_R, B_R + 1] = -Sx
        K[B_R, B_R - 1] = -Sx
        K[B_R, B_R + Nx] = -Sy
        K[B_R, B_R - Nx] = -Sy
        
        K[T_R, T_R] = 1 + 2 * Sx + 2 * Sy
        K[T_R, T_R + 1] = -Sx
        K[T_R, T_R - 1] = -Sx
        K[T_R, T_R + Nx] = -Sy
        K[T_R, T_R - Nx] = -Sy
        
        K[R_R, R_R] = 1 + 2 * Sx + 2 * Sy
        K[R_R, R_R + 1] = -Sx
        K[R_R, R_R - 1] = -Sx
        K[R_R, R_R + Nx] = -Sy
        K[R_R, R_R - Nx] = -Sy
        
        K[L_R, L_R] = 1 + 2 * Sx + 2 * Sy
        K[L_R, L_R + 1] = -Sx
        K[L_R, L_R - 1] = -Sx
        K[L_R, L_R + Nx] = -Sy
        K[L_R, L_R - Nx] = -Sy
        
    # Setting the value of the rows that represent boundary elements to 0 - the 
    # same for both constructions
    K[B_B, :] = np.zeros(nn)
    K[T_B, :] = np.zeros(nn)
    K[R_B, :] = np.zeros(nn)
    K[L_B, :] = np.zeros(nn)
    
    # Setting boundary elements - it is the same for both matrix constructions
    # Bottom
    if BC0[0] == 0 : K[B_B, B_B] = 1
        
    elif BC0[0] == 1:
        
        K[B_B, B_B] = -1
        K[B_B, B_B + Nx] = 1
    
    # Top
    if BC0[1] == 0 : K[T_B, T_B] = 1
        
    elif BC0[1] == 1:
        
        K[T_B, T_B] = 1
        K[T_B, T_B - Nx] = -1
        
    # Left
    if BC0[2] == 0 : K[L_B, L_B] = 1
        
    elif BC0[2] == 1:
        
        K[L_B, L_B] = -1
        K[L_B, L_B + 1] = 1
    
    # Right
    if BC0[3] == 0 : K[R_B, R_B] = 1
    
    elif BC0[3] == 1:
        
        K[R_B, R_B] = 1
        K[R_B, R_B - 1] = -1
            
    return K