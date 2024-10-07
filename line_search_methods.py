#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:18:24 2019

@author: hans-werner
"""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import line_search

def steepest_descent(f, g, x0, tol, k_max):
    """
    Description: Minimize the function f, over an unbounded domain using the 
        steepest descent algorithm.
    
     Inputs:
    
       f: function, returning the function value
       
       g: function, returning the gradient
    
       x0: double, (d,) initial guess
    
       tol: double >0, tolerance for convergence
    
       k_max: int, maximum number of iterations
    
     
     Outputs:
    
       xk: double, (d,) optimal value
    
       record: dict, iteration record with the following keys
            iteration_count: iteration index
            xk: double, x-iterates
            fk: double, function values
            gk_norm: double, norm of gradient
            n_evals: int, number of function/gradient evaluations
            converged: bool, true if the algorithm converged before the k_max
                
            
    Modified: 
        
        10/20/2022 (HW van Wyk)
    """
    #
    # Initialize arrays for recording iteration history
    #
    record = dict()
    record['iteration_count'] = np.zeros(k_max+1);
    record['xk'] = np.zeros((k_max+1,len(x0)));
    record['fk'] = np.zeros(k_max+1);
    record['gk_norm'] = np.zeros(k_max+1);
    record['n_evals'] = np.zeros(k_max+1);

    # 
    # Initial guesses
    # 
    xk = x0;  # initial iterate
    fk = f(xk);  # initial function value 
    gk = g(xk);  # initial gradient value
    g0_norm = norm(gk,2);
    
    #
    # Optimzation loop
    #
    for k in range(k_max):
    
        #
        # Compute the descent direction
        #
        pk = -gk;
            
        # Store current values
        record['iteration_count'][k] = k  # iteration count
        record['xk'][k,:] = xk  # current iterate
        record['fk'][k] = fk  # function value
        gk_norm = norm(gk,2)
        record['gk_norm'][k] = gk_norm  # gradient norm
        
        #
        # Check for convergence (relative gradient)
        #
        # if gk_norm/g0_norm < tol
        if gk_norm < tol:
            record['converged'] = True
            break
        
        #
        # Compute the step-length and update xk, fk, and gk using a 
        # 
        ak, n_a, dummy, fk, dummy, gk = line_search(f,g,xk,pk,gk,fk)
        
         #
        # Update x
        # 
        xk = xk + ak*pk
        
        # Store number of steplength iterations
        record['n_evals'][k] = n_a;


    # Discard unused entries
    record['iteration_count'] = record['iteration_count'][:k+1]
    record['xk'] = record['xk'][:k+1,:]
    record['fk'] = record['fk'][:k+1]
    record['gk_norm'] = record['gk_norm'][:k+1]
    record['n_evals'] = record['n_evals'][:k+1]

    if k == k_max-1:
        print('Maximum number of iterations reached')
        record['converged'] = False
    
    return xk, record



def modified_newton(f,g,h,x0,tol,k_max):
    """
    Description: Uncsontrained minimization of the function f using the modi-
        fied Newton method. 
        
    Inputs:
        
        f: function, objective function
        
        g: function, gradient function
        
        h: function, Hessian function
            
        x0: double, (d,) initial guess
        
        tol: double >0, convergence tolerance 
        
        k_max: int, maximum number of iterations
        
    
    Outputs
    
        xk: double, (d,) approximated optimal value
        
        record: dict, iteration record with the following keys
            iteration_count: iteration index
            xk: double, x-iterates
            fk: double, function values
            gk_norm: double, norm of gradient
            n_evals: int, number of function/gradient evaluations
            converged: bool, true if the algorithm converged before the k_max
                
            
    Modified: 
        
        10/20/2022 (HW van Wyk)
    """
    #
    # Initialize arrays for recording iteration history
    #
    record = dict()
    record['iteration_count'] = np.zeros(k_max+1);
    record['xk'] = np.zeros((k_max+1,len(x0)));
    record['fk'] = np.zeros(k_max+1);
    record['gk_norm'] = np.zeros(k_max+1);
    record['n_evals'] = np.zeros(k_max+1);

    # 
    # Initial guesses
    # 
    xk = x0  # initial iterate
    fk = f(xk)  # initial function value 
    gk = g(xk)  # initial gradient value
    Hk = h(xk)  # initial Hessian value
    g0_norm = norm(gk,2);
    
    #
    # Optimzation loop
    #
    for k in range(k_max):
    
        #
        # Modify the Hessian
        # 
        Bk = make_hessian_positive_definite(Hk)
        
        #
        # Compute the Newton direction
        #
        pk = -np.linalg.solve(Bk,gk)
            
        # Store current values
        record['iteration_count'][k] = k  # iteration count
        record['xk'][k,:] = xk  # current iterate
        record['fk'][k] = fk  # function value
        gk_norm = norm(gk,2)
        record['gk_norm'][k] = gk_norm  # gradient norm
        
        #
        # Check for convergence (relative gradient)
        #
        #if gk_norm/g0_norm < tol
        if gk_norm < tol:
            record['converged'] = True
            break
        
        #
        # Compute the step-length and update xk, fk, and gk using a 
        # 
        ak, n_a, dummy, fk, dummy, gk = line_search(f,g,xk,pk,gk,fk)
        
         #
        # Update x
        # 
        xk = xk + ak*pk
        
        #
        # Update the Hessian
        # 
        Hk = h(xk)
        
        # Store number of steplength iterations
        record['n_evals'][k] = n_a;


    # Discard unused entries
    record['iteration_count'] = record['iteration_count'][:k+1]
    record['xk'] = record['xk'][:k+1,:]
    record['fk'] = record['fk'][:k+1]
    record['gk_norm'] = record['gk_norm'][:k+1]
    record['n_evals'] = record['n_evals'][:k+1]

    if k == k_max-1:
        print('Maximum number of iterations reached')
        record['converged'] = False
    
    return xk, record


def make_hessian_positive_definite(H):
    """
    Description: Modify the computed Hessian by converting its negative eigen-
        values into positive ones, i.e. if H = V*D*VT, then
        
        B = V*(D + dD)VT, where D+dD is the perturbed diagonal matrix
        
        
    Inputs:
        
        H: double (d,d) symmetric Hessian matrix
        
        
    Output:
        
        B: double, (d,d) symmetric, positive definite perturbation of H
    """
    # Smallest positive eigenvalue allowed 
    dlt = 0.1
    
    # Compute the eigenvalues and eigenvectors of H
    d, V = np.linalg.eigh(H)
    
    # Modify diagonal entries
    dm = np.maximum(d, dlt)
    
    # Define modified Hessian B
    B = V.dot(np.diag(dm).dot(V.T))
    
    return B

    
def dfp(f,g,x0,H0,tol,k_max):
    """
    Description: Uncsontrained minimization of the function f using the 
        Davidon-Fletcher-Powell quasi-Newton method
        
    Inputs:
        
        f: function, objective function
        
        g: function, gradient function
            
        x0: double, (d,) initial iterate
        
        H0: double, (d,d) initial guess for inverse Hessian
        
        tol: double >0, convergence tolerance 
        
        k_max: int, maximum number of iterations
        
    
    Outputs
    
        xk: double, (d,) approximated optimal value
        
        record: dict, iteration record with the following keys
            iteration_count: iteration index
            xk: double, x-iterates
            fk: double, function values
            gk_norm: double, norm of gradient
            n_evals: int, number of function/gradient evaluations
            converged: bool, true if the algorithm converged before the k_max
                
            
    Modified: 
        
        10/20/2022 (HW van Wyk)
    """
    #
    # Initialize arrays for recording iteration history
    #
    record = dict()
    record['iteration_count'] = np.zeros(k_max+1);
    record['xk'] = np.zeros((k_max+1,len(x0)));
    record['fk'] = np.zeros(k_max+1);
    record['gk_norm'] = np.zeros(k_max+1);
    record['n_evals'] = np.zeros(k_max+1);

    # 
    # Initial guesses
    # 
    xk = x0  # initial iterate
    Hk = H0  # initial INVERSE Hessian 
    fk = f(xk)  # initial function value 
    gk = g(xk)  # initial gradient value
    g0_norm = norm(gk,2)
    
    #
    # Optimzation loop
    #
    Hk = make_hessian_positive_definite(Hk)
    for k in range(k_max):
        #
        # Compute the descent direction (Hk is the INVERSE Hessian approx)
        #
        pk = -np.dot(Hk, gk)
        
        # Check if it's a descent direction
        if np.dot(pk, gk) > 0:
            print('pk is not a descent direction')
        
        # Store current values
        record['iteration_count'][k] = k  # iteration count
        record['xk'][k,:] = xk  # current iterate
        record['fk'][k] = fk  # function value
        gk_norm = norm(gk,2)
        record['gk_norm'][k] = gk_norm  # gradient norm
        
        #
        # Check for convergence (relative gradient)
        #
        # if gk_norm / g0_norm < tol
        if gk_norm < tol:
            record['converged'] = True
            break
        
        # Remember the old values of x and g
        x_old = xk
        g_old = gk
        
        #
        # Compute the step-length and update xk, fk, and gk using a 
        # 
        ak, n_a, dummy, fk, dummy, gk = line_search(f,g,xk,pk,gk,fk,maxiter=20)
        
        #
        # Update x
        # 
        xk = xk + ak*pk
        
        #
        # Update Hk
        #
        
        # Compute the step size and the change in gradients 
        sk = xk - x_old
        yk = gk - g_old
              
        # Compute DFP update 
        Hk = Hk - np.outer(Hk @ yk, yk.T @ Hk) / np.dot(yk.T, Hk @ yk) + np.outer(sk, sk) / np.dot(yk.T, sk)

        
        # Store number of steplength iterations
        record['n_evals'][k] = n_a;


    # Discard unused entries
    record['iteration_count'] = record['iteration_count'][:k+1]
    record['xk'] = record['xk'][:k+1,:]
    record['fk'] = record['fk'][:k+1]
    record['gk_norm'] = record['gk_norm'][:k+1]
    record['n_evals'] = record['n_evals'][:k+1]

    if k == k_max-1:
        print('Maximum number of iterations reached')
        record['converged'] = False
    
    return xk, record


def bfgs(f,g,x0,B0,tol,k_max):
    """
    Description: Uncsontrained minimization of the function f using the
        Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method
        
    Inputs:
        
        f: function, objective function
        
        g: function, gradient function
                    
        x0: double, (d,) initial guess for optimizer
        
        B0: double, (d,d) initial guess for Hessian
        
        tol: double >0, convergence tolerance 
        
        k_max: int, maximum number of iterations
        
    
    Outputs
    
        xk: double, (d,) approximated optimal value
        
        record: dict, iteration record with the following keys
            iteration_count: iteration index
            xk: double, x-iterates
            fk: double, function values
            gk_norm: double, norm of gradient
            n_evals: int, number of function/gradient evaluations
            converged: bool, true if the algorithm converged before the k_max
                
            
    Modified: 
        
        10/20/2022 (HW van Wyk)
    """
        #
    # Initialize arrays for recording iteration history
    #
    record = dict()
    record['iteration_count'] = np.zeros(k_max+1)
    record['xk'] = np.zeros((k_max+1,len(x0)))
    record['fk'] = np.zeros(k_max+1)
    record['gk_norm'] = np.zeros(k_max+1)
    record['n_evals'] = np.zeros(k_max+1)

    # 
    # Initial guesses
    # 
    xk = x0  # initial iterate
    Bk = B0  # initial value for Hessian
    fk = f(xk)  # initial function value 
    gk = g(xk)  # initial gradient value
    g0_norm = norm(gk,2)
    
    #
    # Optimzation loop
    #
    for k in range(k_max):
    
        #
        # Compute the descent direction
        #
        pk = -np.linalg.solve(Bk,gk)
            
        # Store current values
        record['iteration_count'][k] = k  # iteration count
        record['xk'][k,:] = xk  # current iterate
        record['fk'][k] = fk  # function value
        gk_norm = norm(gk,2)
        record['gk_norm'][k] = gk_norm  # gradient norm
        
        #
        # Check for convergence (relative gradient)
        #
        # if gk_norm/ g0_norm < tol
        if gk_norm < tol:
            record['converged'] = True
            break
        
        x_old = xk
        g_old = gk
        
        #
        # Compute the step-length and update xk, fk, and gk using a 
        # 
        ak, n_a, dummy, fk, dummy, gk = line_search(f,g,xk,pk,gk,fk)
        
         #
        # Update x
        # 
        xk = xk + ak*pk
        
        sk = xk - x_old
        yk = gk - g_old
        
        # BFGS update for Bk (Hessian approximation)
        Bk_sk = Bk @ sk
        sk_T_Bk_sk = np.dot(sk, Bk_sk)
        yk_T_sk = np.dot(yk, sk)

        Bk = Bk - (np.outer(Bk_sk, Bk_sk) / sk_T_Bk_sk) + (np.outer(yk, yk) / yk_T_sk)

        # Store number of steplength iterations
        record['n_evals'][k] = n_a;


    # Discard unused entries
    record['iteration_count'] = record['iteration_count'][:k+1]
    record['xk'] = record['xk'][:k+1,:]
    record['fk'] = record['fk'][:k+1]
    record['gk_norm'] = record['gk_norm'][:k+1]
    record['n_evals'] = record['n_evals'][:k+1]

    if k == k_max-1:
        print('Maximum number of iterations reached')
        record['converged'] = False
    
    return xk, record

    
def backtrack(f, xk, pk, a, rho, c1):
    
    """
    Step length selection algorithm using backtracking
    
     Usage: a,n_iter = backtracking(f, x0, pk, a0, rho, c1)
    
     Inputs:
    
       f: function, to be minimized
    
       xk: double, (dim,) array current iterate
    
       pk: double, (dim,) descent direction
    
       a: double >0, initial guess
    
       rho: double in (0,1), scaling factor
    
       c1: double in (0,1), used in sufficient decrease condition.
     
     Outputs: 
    
       a: double >0, steplength satisfying the sufficient decrease condition
           phi(a) <= f(x) +  c*a*gk^T*p
    
       n_iter: int, number of 
   
    """
    # Compute function value and gradient at current iterate
    fk, gk, dummy = f(xk)
    
    count = 0
    while f(xk+a*pk)[0] > fk+c1*a*gk.dot(pk):
        a = a*rho
        count += 1
        
    return a, count
