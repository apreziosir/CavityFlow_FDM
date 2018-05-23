#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:21:55 2018
First derivative treatment for the Navier-Stokes equations in 2D for the 
Cavity flow. 
@author: Antonio Preziosi-Ribero
Mayo 2018
"""

import numpy as np

# ==============================================================================
# FIRST DERIVATIVES IN SPACE. EACH CASE OF DERIVATION IS TREATED HERE
# THE FLAG THAT MARKS THE TYPE OF DERIVATION IS THE VARIABLE dift
# ==============================================================================
# ==============================================================================
# Differentiating a vector in the x direction - not imposing BC yet. Works for 
# u and v independently
# ==============================================================================
    
def diffx(u, dx, L_B, L_R, R_B, R_R, dift):
    
    # Parameters that can be changed for convenience when running the function
    q = 0.5
    
    # Defining vector that will store the derivative - column vector, like the
    # one that enters the function
    diffx = np.zeros((len(u), 1))
    nn = len(u)
    u = u.reshape((len(u), 1))
    
    # Normal upwind direction
    if dift == 0:
        
        for i in range(1, nn - 1):
            
            if u[i] >= 0:
                
                diffx[i] = (u[i] - u[i - 1]) / dx
            
            else:
                
                diffx[i] = (u[i + 1] - u[i]) / dx
        
        diffx[L_B] = (u[L_B + 1] - u[i]) / dx
        
        diffx[R_B] = (u[i] - u[R_B - 1]) / dx
    
    # Corrected three point upwind from Fletcher 1991    
    elif dift == 1:
        
        # Appending nodes to the boundary and ring arrays to count every possible 
        # scenario
        Nx = L_B[0]
        L_B1 = np.append(0, L_B)
        L_B1 = np.append(L_B1, L_B[-1] + Nx)
        L_R1 = np.append(np.array((L_R[0] - 2 * Nx, L_R[0] - Nx)), L_R)
        L_R1 = np.append(L_R1, np.array((L_R[-1] + Nx, L_R[-1] + 2 * Nx)))
        R_B1 = np.append(R_B[0] - Nx, R_B)
        R_B1 = np.append(R_B1, R_B[-1] + Nx)
        R_R1 = np.append(np.array((R_R[0] - 2 * Nx, R_R[0] - Nx)), R_R)
        R_R1 = np.append(R_R1, np.array((R_R[-1] + Nx, R_R[-1] + 2 * Nx)))
            
        for i in range(0, nn):
                        
            # Testing if element is in Left boundary - since it is a boundary I 
            # have to apply first order upwind scheme
            if np.isin(i, L_B1): 
                
                diffx[i] = (u[i + 1] - u[i]) / dx 
                    
            # Testing if element is in Left ring - Second order is applied only 
            # if velocity is negative
            elif np.isin(i, L_R1):
                
                if u[i] >= 0 : diffx[i] = (u[i] - u[i - 1]) / dx
                
                else:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 1]\
                         - 3 * u[i] + 3 * u[i + 1] - u[i + 2]) / (3 * dx)
                    
            # Testing if element is in right boundary - First order upwind since
            # it is a boundary
            elif np.isin(i, R_B1): 
                
                diffx[i] = (u[i] - u[i - 1]) / dx
                    
            # Testing if element is in right ring - Second order is applied only
            # if velocity is positive
            elif np.isin(i, R_R1):
                
                if u[i] < 0 : diffx[i] = (u[i + 1] - u[i]) / dx
                    
                else:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 2]\
                         - 3 * u[i - 1] + 3 * u[i] - u[i + 1]) / (3 * dx)
                                
            # Internal nodes - The choice is made according to the velocity and 
            # the high order scheme can be applied
            else:
                
                if u[i] >= 0:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 2]\
                         - 3 * u[i - 1] + 3 * u[i] - u[i + 1]) / (3 * dx)              
                    
                else:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 1]\
                         - 3 * u[i] + 3 * u[i + 1] - u[i + 2]) / (3 * dx)
    
    return diffx

# ==============================================================================
# Differentiating a vector in the y direction - not imposing BC yet. Works for
# u and v independently
# ==============================================================================

def diffy(v, dy, B_B, B_R, T_B, T_R, dift):
    
    # Parameters that can be changed for convenience when running the function
    q = 0.5
    
    # Defining vector that will store the derivative - column vector like the 
    # one that enters to the function
    nn = len(v)
    diffy = np.zeros((len(v), 1))
    Nx = len(B_B)
    v = v.reshape((len(v), 1))
    
    # Normal upwind direction
    if dift == 0:
        
        for i in range(B_B[-1] + 1, T_B[0]):
            
            if v[i] >= 0:
                
                diffy[i] = (v[i] - v[i - Nx]) / dy
            
            else:
                
                diffy[i] = (v[i + Nx] - v[i]) / dy
                
        diffy[B_B] = (v[B_B + Nx] - v[B_B]) / dy
        
        diffy[T_B] = (v[T_B] - v[T_B - Nx]) / dy
        
    elif dift == 1:
        
        # Appending nodes to the boundary and ring arrays to count every possible 
        # scenario
        B_R1 = np.append(B_R[0] - 1, B_R)
        B_R1 = np.append(B_R1, B_R[-1] + 1)
        T_R1 = np.append(T_R[0] - 1, T_R)
        T_R1 = np.append(T_R1, T_R[-1] + 1)
        
        for i in range(0, nn):
            
            # Testing if element is in Bottom boundary - since it is a 
            # boundary I have to apply first order upwind scheme
            if np.isin(i, B_B) : diffy[i] = (v[i + Nx] - v[i]) / dy 
                    
            # Testing if element is in Bottom ring - Second order is applied 
            # only if velocity is negative
            elif np.isin(i, B_R1):
                
                if v[i] >= 0 : diffy[i] = (v[i] - v[i - Nx]) / dy
                
                else:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - Nx] - 3 * v[i] + 3 * v[i + Nx] - v[i + 2 * Nx]) \
                    / (3 * dy)
                    
            # Testing if element is in top boundary - First order upwind since
            # it is a boundary
            elif np.isin(i, T_B) : diffy[i] = (v[i] - v[i - Nx]) / dy               
                    
            # Testing if element is in top ring - Second order is applied only
            # if velocity is positive
            elif np.isin(i, T_R1):
                
                if v[i] < 0 : diffy[i] = (v[i + Nx] - v[i]) / dy
                    
                else:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - 2 * Nx] - 3 * v[i - Nx] + 3 * v[i] - v[i + Nx]) \
                    / (3 * dy)
                                
            # Internal nodes - The choice is made according to the velocity and 
            # the high order scheme can be applied
            else:
                
                if v[i] >= 0:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - 2 * Nx] - 3 * v[i - Nx] + 3 * v[i] - v[i + Nx]) \
                    / (3 * dy)              
                    
                else:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - Nx] - 3 * v[i] + 3 * v[i + Nx] - v[i + 2 * Nx]) \
                    / (3 * dy)
    
    return diffy