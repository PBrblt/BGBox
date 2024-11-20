#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:35:01 2023

@author: pierrebarbault
"""
import numpy as np

#import Methods.bgEM as em
import Methods.bgMOM as mom
#import Methods.bgMAP as map
import Methods.bgLEM as lem

from scipy.io import savemat

# =============================================================================
# Read operator and generate data for general inverse problems
# =============================================================================

def read_op(name_op):
    H = np.load('exp_data/op/' + name_op + '.npy')
    return H / np.linalg.norm(H, 2)

def snr_to_sig(x, iSNR):
    """Noise generator"""
    sigma_e = np.mean(x**2)*10**(-iSNR/10)
    return sigma_e

def generate_data(H, p, iSNR):
    """Generate a noisy BG signal"""
    N_y, N_x = np.shape(H)
    s = np.random.rand(N_x) < p
    x = s * np.random.randn(N_x)
    
    y = H @ x 
    s_e = snr_to_sig(x, iSNR)
    
    y = y + np.random.randn(N_y) * s_e
    return y, x, s, np.array([p, 1., s_e])

# =============================================================================
# Iterative thresholding
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

def normes(x, s, x_test):
    
    nrm = np.empty(3)
    
    nrm[0] = np.sum( (x_test != 0) * (1 - s) )
    nrm[1] = np.sum( (x_test == 0) * s )
    nrm[2] = 10 * np.log10( np.mean(x**2) / np.mean( (x - x_test)**2) )
    
    return nrm

def fista(H, y, x_init, T, norme=0):
    
    x_est = np.copy(x_init)
    prev_x = x_est + 1
    
    it = 0
    
    while np.mean((x_est - prev_x)**2) > 10e-3 * np.mean(prev_x**2) :
        
        prev_x = 1 * x_est
        
        z = x_est + H.T @ ( y - H @ x_est )
        
        if norme == 0 :
            next_x = (abs(z) > T) * z#L0
        if norme == 1 :
            next_x = (abs(z) > T) * ( z - np.sign(z) * T )#L1
        
        x_est = next_x + it / (it+5) * ( next_x - prev_x)
        
        it += 1
        
    return(x_est)

def exp_l0(H, y, x, s):
    
    N_point = 100
    
    T = np.logspace(-6, 2, N_point)
    
    x_0 = 0 * x
    
    res_max = np.zeros(3)
    
    for k_t in range(N_point-1, -1, -1):
        t = T[k_t]
        
        x_0 = fista(H, y, x_0, t, 0)
        
        res = normes(x, s, x_0)
        
        if res_max[2] < res[2] :
            res_max = 1 * res
    return res_max

# =============================================================================
# Generation of H
# =============================================================================

N_x = 900
N_y = 300

create_H = False

if create_H:
    list_H = []

    mu = np.ones(N_x)
    #list_H.append(np.eye(N_x))
    #bars = ['Id']

    #list_H.append(np.random.randn(N_x,N_x))
    #list_H[-1] = list_H[-1] / np.linalg.norm(list_H[-1], 2)
    #bars.append("Rndn sqr")

    #list_H.append(np.random.randn(N_y,N_x))
    #list_H[-1] = list_H[-1] / np.linalg.norm(list_H[-1], 2)
    #bars.append("Rndn rec")

    ## list_H.append(np.random.randn(N_x,N_x)**2)
    ## list_H[-1] = list_H[-1] / np.linalg.norm(list_H[-1], 2)
    ## bars.append("Rnd sqr")
    #list_H.append(np.random.multivariate_normal(0.5 * mu, np.eye(N_x),N_x))
    #list_H[-1] = list_H[-1] / np.linalg.norm(list_H[-1], 2)
    #bars.append("Rndn + 1")

    ## list_H.append(np.random.randn(N_y,N_x)**2)
    ## list_H[-1] = list_H[-1] / np.linalg.norm(list_H[-1], 2)
    ## bars.append("Rnd rec")
    #list_H.append(np.random.multivariate_normal(0.5 * mu, np.eye(N_x),N_y))
    #list_H[-1] = list_H[-1] / np.linalg.norm(list_H[-1], 2)
    #bars.append("Rndn + 1")

    list_w = [0.0, 0.1, 0.3, 0.5]
    bars = []
    for w in list_w:
        list_H.append(np.random.multivariate_normal(w * mu, np.eye(N_x),N_y))
        list_H[-1] = list_H[-1] / np.linalg.norm(list_H[-1], 2)
        bars.append("Rnd " + str(w))

    # i = np.arange(900)
    # a,b = np.meshgrid(i,i)
    # for w in list_w :
    #    cov = w ** (abs(a-b))
    #    H = np.random.multivariate_normal(0*i,cov,N_y)
    #    H = H / np.linalg.norm(H, 2)
    #    list_H.append(H)
    #    bars.append('w=' + str(w))

    # list_w = [0.0, 0.05, 0.1, 0.2]
    # for w in list_w :
    #    H = np.random.multivariate_normal(w*np.ones(N_x),np.eye(N_x),N_y)
    #    H = H / np.linalg.norm(H, 2)
    #    list_H.append(H)
    #    bars.append('\mu=' + str(w))

    print(bars)
else:
    #list_H = np.load("Results/list_H.npy", allow_pickle=True)
    list_H = np.load("Results/list_H_sbl.npy", allow_pickle=True)

#H = read_op('meg')
#H = H / np.linalg.norm(H, 2)
#list_H.append(H)

# =============================================================================
# SCRIPT
# =============================================================================
# Number of samples :
N_samples = 100

def exp_trial(p, iSNR, fixed = False):
    res_em = np.zeros((4, len(list_H), 6))
    
    for n in range(N_samples):
        print(n)
        
        s = np.random.rand(N_x) < p
        x = s * np.random.randn(N_x)
        
        for k_op in range(len(list_H)) :
            
            H = list_H[k_op]
            
            y = H @ x 
            s_e = snr_to_sig(y, iSNR)
            y = y + np.random.randn(len(y)) * np.sqrt(s_e)
            
            theta = np.array([p, 1., s_e])
            rel = np.array([1., 1., s_e])
            x_naif = H.T @ y
            
            theta_mom = mom.moments(x_naif)
            
            if fixed :
                theta_mom[2] = s_e
            
            res_em[0, k_op, :3] += exp_l0(H, y, x, s) / N_samples
            res_em[0, k_op, 3:] += abs(theta_mom - theta) / rel / N_samples
            
            theta_m, x_m = lem.em_marg(H, y, 0 * x, theta_mom, fixed, N_out = 100)
            
            theta_j, x_j = lem.em_joint(H, y, 0 * x, theta_mom, fixed)
            theta_jm, x_jm = lem.em_marg(H, y, x_j, theta_j, fixed)
            theta_j, x_j = lem.em_joint(H, y, x_j, theta_j, fixed)
            
            
            res_em[1, k_op, :3] += normes(x, s, x_m) / N_samples
            res_em[1, k_op, 3:] += abs(theta_m - theta) / rel / N_samples
            
            res_em[2, k_op, :3] += normes(x, s, x_j) / N_samples
            res_em[2, k_op, 3:] += abs(theta_j - theta) / rel  / N_samples
            
            res_em[3, k_op, :3] += normes(x, s, x_jm) / N_samples
            res_em[3, k_op, 3:] += abs(theta_jm - theta) / rel / N_samples
    return res_em

# =============================================================================
# Plot results
# =============================================================================
# Input SNR :
iSNR = 20

p_list = [0.01, 0.05, 0.1]

save_op = False

if save_op:
    np.save("Results/list_H", list_H)
    savemat("Results/list_H.mat", {'H':list_H})
    #np.save("Results/list_H_sbl", list_H)
    #savemat("Results/list_H_sbl.mat", {'H':list_H})

for i in range(len(p_list)) :
    p = p_list[i]
    res_em = exp_trial(p, iSNR)

    np.save("Results/res_sbl_" + str(iSNR) + "_" + str(i), res_em)