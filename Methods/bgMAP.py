#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains Maximum A Posteriori (MAP) algorithms for Bernoulli-Gaussian (BG) signals with additive Gaussian noise.
@author: pierrebarbault
"""
import numpy as np

# =============================================================================
# Thresholding
# =============================================================================
def T_s(theta):
    if theta[2] == 0 :
        return 0
    rho = 2 * np.log( 1 - theta[0] ) - 2 * np.log( theta[0] )
    
    return np.sqrt( (theta[1] + theta[2])/ theta[1] * theta[2] * ( rho + np.log( theta[1] / theta[2] + 1 ) ) )

def T_x(theta):
    if theta[2] == 0 :
        return 0
    rho = 2 * np.log( 1 - theta[0] ) - 2 * np.log( theta[0] )
    
    return np.sqrt( theta[2] * (theta[1] + theta[2]) / theta[1] * (  rho + np.log( 2 * np.pi * theta[1]) ) )

def T_xs(theta):
    if theta[2] == 0 :
        return 0
    rho = 2 * np.log( 1 - theta[0] ) - 2 * np.log( theta[0] )
    
    return np.sqrt( theta[2] * (theta[1] + theta[2]) / theta[1] * (  rho + np.log( 2 * np.pi * theta[1]) ) )

# =============================================================================
# Marginal MAP estimation
# =============================================================================

def deb_marg(y, theta_in, gamma):
    return y

# =============================================================================
# Joint MAP estimation
# =============================================================================

def deb_joint(y, theta_in, gamma, fixed = False):
    
    N = len(y)
    
    ###
    y_2 = np.flip(np.sort( y**2 ))
    
    T_max = y_2[:N-1]
    T_min = y_2[1:]
    
    S_plus = np.cumsum(y_2)[:N-1]
    S_moins = np.sum(y_2) - S_plus + theta_in[2] * gamma
    
    K = np.arange(N-1) + 1
        
    # Estimation de p :
    p_est = K / N
    
    # Estimation de s_x, s_e selon Gassiat et Goussard :
    
    if fixed :
        delta = ( 2 * theta_in[2] - S_plus )**2 - 4 * K * theta_in[2]**2
        
        mask = delta >= 0
        
        s_x_est = 0.5 * ( S_plus[mask] - 2 * theta_in[2] + np.sqrt(delta[mask]) ) / K[mask]
        s_e_est = 0 * s_x_est + theta_in[2]
    else :
        a = S_moins + S_plus
        b = 2 * S_moins - S_plus / p_est
        
        delta = b**2 - 4 * a * S_moins
        
        mask = delta >= 0
        
        root = ( - b[mask] - np.sqrt(delta[mask]) ) / (2 * a[mask])
                
        s_x_est = (S_plus[mask] / K[mask]) / (1 + root)**2
        s_e_est = root * s_x_est
    
    #print(delta[mask])
    
    if any(mask) :
        p_est = p_est[mask]
        
        # Estimation corrigée :
        
        rho = 2 * np.log(1 / p_est - 1)
        
        T_est = ( s_x_est + s_e_est) * s_e_est / s_x_est * ( np.log( 2 * np.pi * s_x_est) + rho )
        
        Q_est = ( S_plus[mask] / (s_x_est + s_e_est) + S_moins[mask] / s_e_est 
                + N * np.log(s_e_est) + K[mask] * np.log(2 * np.pi * s_x_est) 
                - 2 * K[mask] * np.log( p_est ) - 2 * (N - K[mask]) * np.log( 1 - p_est ) )
        
        ## recherche de la solution :
        
        #cond = (T_est < T_max[mask]) * (T_est > T_min[mask]) #OLD
        cond_2 = (T_est - T_max[mask]) * (T_est > T_max[mask]) + (T_min[mask] - T_est) * (T_est < T_min[mask])
        cond = cond_2 == 0
        
        if any(cond) :
            
            min_Q = np.argmin(Q_est[cond])
            
            T_est = T_est[cond]
            s_e_est = s_e_est[cond]
            s_x_est = s_x_est[cond]
            p_est = p_est[cond]
        else :
            #print('crap')
            #min_Q = np.argmin(Q_est) #OLD
            if any(T_est > 0): # Ajout car bug parfois
                min_Q = np.argmin(cond_2[T_est > 0])

                s_e_est = s_e_est[T_est > 0]
                s_x_est = s_x_est[T_est > 0]
                p_est = p_est[T_est > 0]
                T_est = T_est[T_est > 0]
            else :
                print("Oula")
                min_Q = np.argmin(cond_2)
            
            
        
        x_out = ( y**2 > T_est[min_Q] ) * y * s_x_est[min_Q] / (s_x_est[min_Q] + s_e_est[min_Q])
        theta_out = np.array([p_est[min_Q], s_x_est[min_Q], s_e_est[min_Q]])
    else :
        #print('ouch')
        theta_out = 1 * theta_in
        x_out = 1 * y
    
    return theta_out, x_out