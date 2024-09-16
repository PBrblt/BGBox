#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains Expectation-Maximization (EM) algorithm for Bernoulli-Gaussian (BG) signals with additive Gaussian noise.
@author: pierrebarbault
"""
import numpy as np

# =============================================================================
# Compute posteriori distribution of the support p(s | y, \theta)
# =============================================================================
def compute_phi(y,theta):
    
    sig_on = theta[1] + theta[2] # variance when the support is 'on' (s=1)
    sig_off = 1 * theta[2] # variance when the support is 'off' (s=0)
    
    if sig_on <= 0 :
        print( 'sig_x + sig_e <= 0' )
        return np.ones(len(y))
    
    phi_on = theta[0] * np.exp(- 0.5 * y**2 / sig_on ) / np.sqrt(sig_on)
    
    if sig_off > 0 :
        phi_off = (1-theta[0]) * np.exp(-0.5 * y**2 / theta[2] ) / np.sqrt(theta[2])
    else :
        # When the noise variance is zero, phi_off become a dirac distribution
        phi_off = 0 * y
        phi_off[y == 0] = np.inf
    # If both phi_off and phi_on are zeros, we put the result to zero
    res = 0 * y
    mask = (phi_on + phi_off) != 0
    res[mask] = phi_on[mask] / ( phi_on + phi_off )[mask]
    return res

# =============================================================================
# Denoising EM with s as an hidden variable
# =============================================================================

def bg_em_s(y, theta_in, fixed = False):
    
    N_in = 50
    
    theta_out = 1 * theta_in
    
    for k in range(N_in) :
        phi = compute_phi(y,theta_out)
        esp_x =  theta_out[1]/(theta_out[1] + theta_out[2]) * y
        esp_x2 = esp_x**2 + theta_out[1]/(theta_out[1] + theta_out[2]) * theta_out[2]
        
        #coeff = theta[2]/(theta[1] + theta[2])
        
        theta_out[0] = np.mean(phi)
        
        if not fixed :
            theta_out[2] = np.mean( phi * (esp_x2 - 2 * y * esp_x ) + y**2 ) #np.mean(y**2) + new_p * theta[1] * coeff  + (coeff**2 - 1) * np.mean( phi * y**2 )
        
        if theta_out[0] == 0 :
            theta_out[1] = 1 * theta_out[2]
        else :
            theta_out[1] = np.mean( phi * esp_x2) / theta_out[0] #theta[1] * coeff + (1 - coeff)**2 * np.mean( phi * y**2 )/new_p
    
    return theta_out, phi

# =============================================================================
# Denoising EM with x as an hidden variable
# =============================================================================

def bg_em_x(y, theta_in, fixed = False):
    
    N_in = 50
    
    theta_out = 1 * theta_in
    
    for k in range(N_in) :
        phi = compute_phi(y,theta_out)
        esp_x =  theta_out[1]/(theta_out[1] + theta_out[2]) * y
        esp_x2 = esp_x**2 + theta_out[1]/(theta_out[1] + theta_out[2]) * theta_out[2]
        
        #coeff = theta[2]/(theta[1] + theta[2])
        
        theta_out[0] = np.mean(phi)
        
        if not fixed :
            theta_out[2] = np.mean( phi * (esp_x2 - 2 * y * esp_x ) + y**2 ) #np.mean(y**2) + new_p * theta[1] * coeff  + (coeff**2 - 1) * np.mean( phi * y**2 )
        
        if theta_out[0] == 0 :
            theta_out[1] = 1 * theta_out[2]
        else :
            theta_out[1] = np.mean( phi * esp_x2) / theta_out[0] #theta[1] * coeff + (1 - coeff)**2 * np.mean( phi * y**2 )/new_p
    
    return theta_out, phi

# =============================================================================
# Denoising EM with (x,s) as an hidden variable
# =============================================================================

def bg_em_xs(y, theta_in, fixed = False):
    
    N_in = 50
    
    theta_out = 1 * theta_in
    
    for k in range(N_in) :
        phi = compute_phi(y,theta_out)
        esp_x =  theta_out[1]/(theta_out[1] + theta_out[2]) * y
        esp_x2 = esp_x**2 + theta_out[1]/(theta_out[1] + theta_out[2]) * theta_out[2]
        
        #coeff = theta[2]/(theta[1] + theta[2])
        
        theta_out[0] = np.mean(phi)
        
        if not fixed :
            theta_out[2] = np.mean( phi * (esp_x2 - 2 * y * esp_x ) + y**2 ) #np.mean(y**2) + new_p * theta[1] * coeff  + (coeff**2 - 1) * np.mean( phi * y**2 )
        
        if theta_out[0] == 0 :
            theta_out[1] = 1 * theta_out[2]
        else :
            theta_out[1] = np.mean( phi * esp_x2) / theta_out[0] #theta[1] * coeff + (1 - coeff)**2 * np.mean( phi * y**2 )/new_p
    
    return theta_out, phi