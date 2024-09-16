#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains Latent Expectation-Maximization (EM) algorithm for Bernoulli-Gaussian (BG) signals observed through linear operator H with additive Gaussian noise.
@author: pierrebarbault
"""
import numpy as np
import Methods.bgEM as em
import Methods.bgMAP as map

# =============================================================================
# Marginal LEM
# =============================================================================

def em_marg(H, y, x_init, theta_init, s_e_known = False, N_out = 50, N_in = 50):
    
    theta_est = 1. * theta_init
    
    x_est = 1. * x_init
        
    for k in range(N_out):
        
        z = x_est + H.T @ (y - H @ x_est)
        
        theta_est, phi_est = em.bg_em_x(z, theta_est, s_e_known)
        
        x_est = theta_est[1] / (theta_est[1] + theta_est[2]) * phi_est * z
        
    return theta_est, x_est * (phi_est > .5)

# =============================================================================
# Joint LEM
# =============================================================================

def em_joint(H, y, x_init, theta_init, s_e_known = False, N_step = 50):
    
    theta_est = 1 * theta_init
    x_est = 1. * x_init
    
    N = len(x_est)
    
    if H.all == np.eye(N).all :
        theta_est, x_est = map.deb_joint(y, theta_est, 0)
    else :
        gamma = np.trace( np.eye(N) - H.T @ H )
        #print(gamma)
        
        for k in range(N_step) :
            
            hat_z = x_est + H.T @ ( y - H @ x_est )
            
            theta_est, x_est = map.deb_joint(hat_z, theta_est, gamma, s_e_known)
        
    return theta_est, x_est