#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:35:01 2023

@author: pierrebarbault
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
# Moments
# =============================================================================

def moments(y):
    
    y_2 = y**2
    N = len(y)
    m2 = np.mean(y_2)
    m4 = np.mean(y_2**2)
    m6 = np.mean(y_2**3)
    
    while m6 < 5 * m4**2 / (3 * m2) and N > 3 :
        
        m_2_alt = ( N * m2 - y_2 ) / ( N - 1 )
        m_4_alt = ( N * m4 - y_2**2 ) / ( N - 1 )
        m_6_alt = ( N * m6 - y_2**3 ) / ( N - 1 )
        
        crit = m_6_alt - 5 * m_4_alt**2 / (3 * m_2_alt)
        
        y_2 = np.delete(y_2, np.argmax(crit))
        m2 = np.mean(y_2)
        m4 = np.mean(y_2**2)
        m6 = np.mean(y_2**3)
        N -= 1
    
#    if N != len(y) :
#        print(len(y) - N)
    
    
    a = m4 - 3 * m2**2
    b = m2 * m4 - m6 / 5
    c = m2 * m6 / 5 - m4**2 / 3
        
    delta = b**2 - 4 * a * c
    if delta < 0 :
        print(delta)
        
    r_pos = -.5 * ( b + np.sqrt(delta) ) / a
    r_neg = -.5 * ( b - np.sqrt(delta) ) / a
    #print(r_pos, r_neg, m2)
    r_pos = max( 0, min(m2, r_pos ) ) # Necessary condition
    r_neg = max( 0, min(m2, r_neg ) ) # Necessary condition
    
   
        
    new_s_e = min(r_pos, r_neg)
    
    new_s_x = (m4/3 - new_s_e**2)/(m2 - new_s_e) - 2 * new_s_e
    new_p = (m2 - new_s_e)/new_s_x
    
    new_p = len(y_2)/len(y) * new_p
    
    return np.array([new_p, new_s_x, new_s_e])

# =============================================================================
# Compute posteriori distribution of the support
# =============================================================================
def compute_phi(y,theta):
    
    sig_on = theta[1] + theta[2]
    
    if sig_on <= 0 :
        print( 'sig_x + sig_e <= 0' )
        return np.ones(len(y))
    
    phi_on = theta[0] * np.exp(- 0.5 * y**2 / sig_on ) / np.sqrt(sig_on)
    
    if theta[2] > 0 :
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
# Denoising EM with x as an hidden variable
# =============================================================================

def deb_marg(y, theta_in, fixed = False):
    
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
# Marginal LEM
# =============================================================================

def em_marg(H, y, x_init, theta_init, s_e_known = False, N_out = 50, N_in = 50):
    
    theta_est = 1. * theta_init
    
    x_est = 1. * x_init
        
    for k in range(N_out):
        
        z = x_est + H.T @ (y - H @ x_est)
        
        theta_est, phi_est = deb_marg(z, theta_est, s_e_known)
        
        x_est = theta_est[1] / (theta_est[1] + theta_est[2]) * phi_est * z
        
    return theta_est, x_est * (phi_est > .5)

# =============================================================================
# Denoising Joint estimation
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
    
    if np.sum(mask) != 0 :
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
        
        if np.sum(cond) != 0 :
            
            min_Q = np.argmin(Q_est[cond])
            
            T_est = T_est[cond]
            s_e_est = s_e_est[cond]
            s_x_est = s_x_est[cond]
            p_est = p_est[cond]
        else :
            #print('crap')
            #min_Q = np.argmin(Q_est) #OLD
            min_Q = np.argmin(cond_2[T_est > 0])
            
            s_e_est = s_e_est[T_est > 0]
            s_x_est = s_x_est[T_est > 0]
            p_est = p_est[T_est > 0]
            T_est = T_est[T_est > 0]
            
            
        
        x_out = ( y**2 > T_est[min_Q] ) * y * s_x_est[min_Q] / (s_x_est[min_Q] + s_e_est[min_Q])
        theta_out = np.array([p_est[min_Q], s_x_est[min_Q], s_e_est[min_Q]])
    else :
        #print('ouch')
        theta_out = 1 * theta_in
        x_out = 1 * y
    
    return theta_out, x_out

# =============================================================================
# Joint LEM
# =============================================================================

def em_joint(H, y, x_init, theta_init, s_e_known = False, N_step = 50):
    
    theta_est = 1 * theta_init
    x_est = 1. * x_init
    
    N = len(x_est)
    
    if H.all == np.eye(N).all :
        theta_est, x_est = deb_joint(y, theta_est, 0)
    else :
        gamma = np.trace( np.eye(N) - H.T @ H )
        #print(gamma)
        
        for k in range(N_step) :
            
            hat_z = x_est + H.T @ ( y - H @ x_est )
            
            theta_est, x_est = deb_joint(hat_z, theta_est, gamma, s_e_known)
        
    return theta_est, x_est

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
    
    while np.mean((x_est - prev_x)**2)/np.mean(prev_x**2) > 10e-3 :
        
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

list_H = []

list_w = [0, 0.5, 0.7, 0.9]
list_H.append(np.eye(N_x))
bars = ['Id']

for w in list_w :
    i = np.arange(900)
    a,b = np.meshgrid(i,i)
    cov = w ** (abs(a-b))
    H = np.random.multivariate_normal(0*i,cov,N_y)
    H = H / np.linalg.norm(H, 2)
    list_H.append(H)
    bars.append('w=' + str(w))

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
            
            theta_mom = moments(x_naif)
            
            if fixed :
                theta_mom[2] = s_e
            
            res_em[0, k_op, :3] += exp_l0(H, y, x, s) / N_samples
            res_em[0, k_op, 3:] += (theta_mom - theta) / rel / N_samples
            
            theta_m, x_m = em_marg(H, y, 0 * x, theta_mom, fixed, N_out = 100)
            
            theta_j, x_j = em_joint(H, y, 0 * x, theta_mom, fixed)
            theta_jm, x_jm = em_marg(H, y, x_j, theta_j, fixed)
            theta_j, x_j = em_joint(H, y, x_j, theta_j, fixed)
            
            
            res_em[1, k_op, :3] += normes(x, s, x_m) / N_samples
            res_em[1, k_op, 3:] += (theta_m - theta) / rel / N_samples
            
            res_em[2, k_op, :3] += normes(x, s, x_j) / N_samples
            res_em[2, k_op, 3:] += (theta_j - theta) / rel  / N_samples
            
            res_em[3, k_op, :3] += normes(x, s, x_jm) / N_samples
            res_em[3, k_op, 3:] += (theta_jm - theta) / rel / N_samples
    return res_em

# =============================================================================
# Plot results
# =============================================================================
# Input SNR :
iSNR = 5

p_list = [0.01, 0.05, 0.1]

color_list = ['r', 'y', 'g','b']
hatch_list = [None,'/','-','x']

tag_list = ['False Positive', 'False Negative', 'output SNR', 'Absolute error of $p$', 'Absolute error of $\sigma_x^2$', 'Relative error of $\sigma_e^2$']

abcisse = np.arange(len(list_H))

L = 0.06

#handle_list = [Patch(color='red', label='p = 0.01'), Patch(color='yellow', label='p = 0.05'), Patch(color='green', label='p = 0.1'), 
#               Patch(facecolor='white', edgecolor ='black', label='L0'), Patch(facecolor='white', edgecolor ='black', hatch = '//', label='M'),
#               Patch(facecolor='white', edgecolor ='black', hatch = '--', label='J'), Patch(facecolor='white', edgecolor ='black', hatch = 'xx', label='LEMUR')]
#
#handle_list2 = [Patch(color='red', label='p = 0.01'), Patch(color='yellow', label='p = 0.05'), Patch(color='green', label='p = 0.1'), 
#               Patch(facecolor='white', edgecolor ='black', label='Mom'), Patch(facecolor='white', edgecolor ='black', hatch = '//', label='M'),
#               Patch(facecolor='white', edgecolor ='black', hatch = '--', label='J'), Patch(facecolor='white', edgecolor ='black', hatch = 'xx', label='LEMUR')]

handle_list = [Patch(color='r', label='L0'), Patch(color='y', label='M'), Patch(color='g', label='J'), 
               Patch(color='b', edgecolor ='black', label='LEMUR'), Patch(facecolor='white', edgecolor ='black', label='p=0.01'),
               Patch(facecolor='white', edgecolor ='black', hatch = '//', label='p=0.05'), Patch(facecolor='white', edgecolor ='black', hatch = '--', label='p=0.1')]

handle_list2 = [Patch(color='r', label='Mom'), Patch(color='y', label='M'), Patch(color='g', label='J'), 
               Patch(color='b', edgecolor ='black', label='LEMUR'), Patch(facecolor='white', edgecolor ='black', label='p=0.01'),
               Patch(facecolor='white', edgecolor ='black', hatch = '//', label='p=0.05'), Patch(facecolor='white', edgecolor ='black', hatch = '--', label='p=0.1')]

for i in range(len(p_list)) :
    p = p_list[i]
    res_em = exp_trial(p, iSNR)
    
    for k in range(3) :
        plt.figure(k)
        for j in range(4):
            #plt.bar(abcisse + j * 0.2 + i * 0.06, res_em[j, :, k], width = L, color = color_list[i], hatch = hatch_list[j]) #Reaggae bar
            plt.bar(abcisse + i * 0.26 + j * 0.06, res_em[j, :, k], width = L, color = color_list[j], hatch = hatch_list[i])
        plt.ylabel(tag_list[k])
        plt.title('Input SNR = ' + str(iSNR) + ' dB')
        plt.xticks(abcisse + 0.4, bars, rotation=45, fontweight='bold')
        
        plt.legend(handles = handle_list)
        
        if k < 2 :
            plt.yscale('log')
        plt.xlim(-0.2,6.5)
        plt.show()
#        plt.figure(k)
#        plt.plot(abcisse + i * 0.1, res_em[0, :, k], color_list[i] + '^', label = 'L0')
#        plt.plot(abcisse + i * 0.1, res_em[1, :, k], color_list[i] + 'v', label = 'M')
#        plt.plot(abcisse + i * 0.1, res_em[2, :, k], color_list[i] + '<', label = 'J')
#        plt.plot(abcisse + i * 0.1, res_em[3, :, k], color_list[i] + '>', label = 'J+M')
#        plt.title(tag_list[k])
#        plt.xticks(abcisse, bars, rotation=45, fontweight='bold')
#        plt.legend()
#        
#        if k < 2 :
#            plt.yscale('log')
#        plt.show()
    for k in range(3,6) :
        plt.figure(k)
        for j in range(4):
            #plt.bar(abcisse + j * 0.2 + i * 0.06, abs(res_em[j, :, k]), width = L, color = color_list[i], hatch = hatch_list[j]) #Reggea bars
            plt.bar(abcisse + i * 0.26 + j * 0.06, abs(res_em[j, :, k]), width = L, color = color_list[j], hatch = hatch_list[i])
        plt.ylabel(tag_list[k])
        plt.title('Input SNR = ' + str(iSNR) + ' dB')
        plt.xticks(abcisse + 0.4, bars, rotation=45, fontweight='bold')
        
        plt.legend(handles = handle_list2)
        plt.yscale('log')
        plt.xlim(-0.2,6.5)
        plt.show()
        
#    for k in range(3,6) :
#        plt.figure(k)
#        plt.plot(abcisse + i * 0.1, abs(res_em[0, :, k]), color_list[i] + '^', label = 'Mom')
#        plt.plot(abcisse + i * 0.1, abs(res_em[1, :, k]), color_list[i] + 'v', label = 'M')
#        plt.plot(abcisse + i * 0.1, abs(res_em[2, :, k]), color_list[i] + 'x', label = 'J')
#        plt.plot(abcisse + i * 0.1, abs(res_em[3, :, k]), color_list[i] + '+', label = 'J+M')
#        plt.title(tag_list[k])
#        plt.legend()
#        plt.yscale('log')
#        plt.xticks(abcisse, bars, rotation=45, fontweight='bold')
#        plt.show()