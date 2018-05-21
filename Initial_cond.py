#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:41:19 2018
Initial condition for the cavity flow problem solved via finite differences
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""

def Init_cond(X, Y):
    
    # Initial velocities are 0 for the initial condition of the cavity flow 
    # problem. This can be modified in order to run the program with other 
    # initial conidtions
    
    U0 = 0 * X * Y
    V0 = 0 * X * Y
        
    return [U0, V0]