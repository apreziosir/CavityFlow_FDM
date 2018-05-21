#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure regularization function. It takes as an input the Laplacian matrix for 
the calculation of pressure, extracts the singular values, finds the minimum 
vector associated to that value and returns that vector
@author: Antonio Preziosi-Ribero
Created on Fri May 18 12:11:26 2018
CFD - Pontificia Universidad Javeriana
"""

import numpy as np
import numpy.linalg as nl
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import style

def Regularization(K):
    
    # Spying K just for checking
    plt.ion()
    style.use('ggplot')
    plt.figure()
    ax = plt.gca()
    ax.spy(K)
    ax.set_title('treated matrix')
    plt.draw()
    plt.pause(5)
    plt.close()
    
    # Setting up tolerance for checking
    tol = 1e-5
    
    # Cheking for matrix symmetry (matrix should be equal or machine precision 
    # close to its transpose)
    are_same = K - K.T
    are_same = are_same.A
    
    # Estimating euclidean norm of the difference between matrices - should be 
    # extremely close to zero or exactly zero if matrices are equal    
    norm = nl.norm(are_same, 2)

    # Selecting between eigenvalues or singular values according to matrix 
    # symmetry
    if norm <= tol:
        
        # Finding eigenvalues and eigenvectors
        [s, vh] = nl.eig(K.A)
    
    else:
        
        # Finding singular values and singular vectors
        [u, s, vh] = nl.svd(K.A, full_matrices=False)
    
    # Storing eigenvalues and eigenvectors' matrix to do the computations
    eig_val = s ** 2
    
    # I have to choose the left singular value associated with the smallest
    # singular value, which is the last one of the list (thanks to the 
    # algorithm)
    q = (u[:,-1])
    q = np.reshape(q, (1, len(u)))
    qt = np.reshape(q, (len(u), 1))
    print(q)
    print(qt)
        
    # Plotting the numerical line that locates the position of the singular 
    # values / eigenvalues found inside the if clause    
    y = np.ones(len(s))
    
    plt.ion()
    style.use('ggplot')
    plt.figure()
    ax = plt.gca()
    ax.scatter(eig_val, y, c='blue', alpha=0.25, edgecolors='none')
    ax.set_xscale('log')
    ax.set_xlim([1e-8, 1e2])
    ax.set_ylim([0.8, 1.2])
    ax.set_title('Eigenvalues location in the number line')
    plt.draw()
    plt.pause(3)
    plt.close()
    
#    # This part has to be reevaluated. If there is any option to get the 
#    # solutions of the program this is not necessary
#    # Solving a linear system to find the eigenvector corresponding to the 
#    # smallest eigenvalue [A - (lambda) * I] * qqv = 0
#    nn = K.shape[0]
#    K1 = K - np.eye(nn) * np.max(eig_val)
#      
    
    # This is a MATRIX, if it is a dot product between the two vectors, the 
    # value of qqt would be 1.0, since the eigenvectors (or singular vectors),
    # are unitary
    qqt = np.matmul(qt, q)
    
#    print(qqt)
   
    return qqt