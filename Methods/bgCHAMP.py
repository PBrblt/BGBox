#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains Champagne algorithm for Sparse Bayesian Learning (SBL) models with additive Gaussian noise.
@author: pierrebarbault
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def champagne(H, y, sigma_init, fixed = False):
    
    # Initialisation
    gamma = 0 * H.T @ y + 1
    Id = np.eye(len(y))
    sigma = 1 * sigma_init

    for t in range(20):
        #print(t)

        Gamma = np.diag(gamma)

        Sigma_y_inv = np.linalg.inv(sigma * Id + H @ Gamma @ H.T)

        x_hat = Gamma @ H.T @ Sigma_y_inv @ y

        Sigma_x = Gamma - Gamma @ H.T @ Sigma_y_inv @ H @ Gamma

        # Noise update
        if fixed :
            sigma = np.mean((y - H @ x_hat)**2) + sigma * np.sum( np.diag(Sigma_x) / gamma ) / len(y)
        
        # gamma update
        gamma = np.diag(Sigma_x) + x_hat**2 # EM update
    return x_hat

#####################################################################

list_H = np.load("Results/list_H_sbl.npy", allow_pickle=True)

list_iSNR = [20, 10, 5]

list_p = [0.01, 0.05, 0.1]

T = 1e-3

N = 1#00

res = np.zeros((len(list_H),len(list_iSNR), len(list_p),3))

for k in range(len(list_H)):

    H = list_H[k]
    N_y, N_x = np.shape(H)

    for l in range(len(list_iSNR)):
        iSNR = list_iSNR[l]
        for m in range(len(list_p)):
            p = list_p[m]
            print(k,l,m)
            for n in range(N):
                s = np.random.rand(N_x) < p
                x = s * np.random.randn(N_x)

                y = H @ x 
                s_e = np.mean(x**2)*10**(-iSNR/10)
            
                y = y + np.random.randn(N_y) * s_e

                #tic = time.time()
                #x_est = champagne(H, y, s_e**2, True)
                x_est = champagne(H, y, np.mean(y**2), True)
                #print(time.time()-tic)

                fp = np.sum( (abs(x_est) > T) * (1 - s) )
                fn = np.sum( (abs(x_est) <= T) * s )
                oSNR = 10 * np.log10( np.mean(x**2) / np.mean( (x - x_est)**2) )

                res[k,l,m,0] = res[k,l,m,0] + fp / N 
                res[k,l,m,1] = res[k,l,m,1] + fn / N 
                res[k,l,m,2] = res[k,l,m,2] + oSNR / N

print(res)

np.save("Results/res_champ_uns_2", res)

plt.figure()
plt.scatter(np.arange(N_x), x)
plt.scatter(np.arange(N_x), x_est)
plt.show()