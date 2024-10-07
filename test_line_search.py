#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:40:11 2022

@author: hans-werner
"""

import numpy as np
from line_search_methods import steepest_descent, modified_newton, dfp, bfgs
import matplotlib.pyplot as plt

def test_rosenbrock():
    """
    Test convergence of the 
    """
    # Function 
    f = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    # Gradient 
    g = lambda x: np.array([-400*(x[1] - x[0]**2)*x[0] - 2*(1 - x[0]), 
                            200*(x[1] - x[0]**2)])
           
    # Hessian 
    H = lambda x: np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], 
                            [-400*x[0], 200]])
    
    tol = 1e-6
    k_max = 15000
    x0 = np.array([-1.2,2])
    
    x_star = np.array([1, 1])  # Ground truth global minimizer
    
    # Test steepest descent
    xk_sd, record_sd = steepest_descent(f,g,x0,tol,k_max)
    
    
    # Test modified Newton 
    xk_newton, record_newton = modified_newton(f,g,H,x0,tol,k_max)
    
    
    # Test DFP 
    H0 = np.linalg.inv(H(x0))
    xk_dfp, record_dfp = dfp(f,g,x0,H0,tol,k_max)
    
    
    # Test BFGS
    B0 = np.eye(2)
    xk_bfgs, record_bfgs = bfgs(f,g,x0,B0,tol,k_max)
    
    
    # Compute errors
    errors_sd = np.linalg.norm(record_sd['xk'] - x_star, axis=1)
    errors_newton = np.linalg.norm(record_newton['xk'] - x_star, axis=1)
    errors_dfp = np.linalg.norm(record_dfp['xk'] - x_star, axis=1)
    errors_bfgs = np.linalg.norm(record_bfgs['xk'] - x_star, axis=1)
    
    '''
    # Plot errors vs iterations
    plt.figure(figsize=(10, 6))
    plt.semilogy(record_sd['iteration_count'], errors_sd, label='Steepest Descent')
    plt.semilogy(record_newton['iteration_count'], errors_newton, label='Modified Newton')
    plt.semilogy(record_dfp['iteration_count'], errors_dfp, label='DFP')
    plt.semilogy(record_bfgs['iteration_count'], errors_bfgs, label='BFGS')
    
    plt.xlabel('Iterations')
    plt.ylabel('Error ||x_k - x*||')
    plt.title('Convergence of Optimization Methods on Rosenbrock Function')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    
    # Plot errors vs iterations
    plt.figure(figsize=(10, 6))
    plt.loglog(record_sd['iteration_count'], errors_sd, label='Steepest Descent')
    plt.loglog(record_newton['iteration_count'], errors_newton, label='Modified Newton')
    plt.loglog(record_dfp['iteration_count'], errors_dfp, label='DFP')
    plt.loglog(record_bfgs['iteration_count'], errors_bfgs, label='BFGS')

    plt.xlabel('Iterations (log scale)')
    plt.ylabel('Error ||x_k - x*|| (log scale)')
    plt.title('Convergence of Optimization Methods on Rosenbrock Function (Log-Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


    
if __name__ == '__main__':
    test_rosenbrock()
