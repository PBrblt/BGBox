#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file plot the comparisaon between LEMUR and SBL.
@author: pierrebarbault
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

##

list_title = ["False Positive", "False Negative", "SNR"]

list_iSNR = [20, 10, 5]
list_w = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

list_p = [0.01, 0.05, 0.1]
list_line = ['-', '--', '-.']

res_champ = np.load("Results/res_champ.npy")

for i in range(len(list_iSNR)):
    iSNR = list_iSNR[i]

    res_amp = loadmat("Results/Res_AMP_sbl_" + str(iSNR))['Res']

    plt.figure()
    

    for j in range(len(list_p)):
        p = list_p[j]
        res_lem = np.load("Results/res_sbl_" + str(iSNR) + "_" + str(j) + ".npy")

        for k in range(3):
            plt.subplot(1, 3, 1+k).set_title(list_title[k])
            
            plt.plot(list_w, res_lem[0,:,k], 'r', linestyle = list_line[j])
            plt.plot(list_w, res_champ[:,i,j,k], 'b', linestyle = list_line[j])
            plt.plot(list_w, res_amp[j,:,k], 'c', linestyle= list_line[j])
            plt.plot(list_w, res_lem[3,:,k], 'g', linestyle = list_line[j])
            if k < 2 :
                plt.yscale('log')
            #if j == 0 :
            #    plt.plot([], 'r', label = '$\ell_0$')
            #    plt.plot([], 'b', label = 'SBL')
            #    plt.plot([], 'c', label = 'EMGAMP')
            #    plt.plot([], 'g', label = 'LEMUR')
            #plt.plot([], 'k', linestyle = list_line[j], label = 'p = ' + str(p))
            #plt.legend()
            plt.xlabel("$\mu$")
        plt.subplot(1, 3, 3)
        if j == 0:
            plt.plot([], 'r', label = '$\ell_0$')
            plt.plot([], 'b', label = 'SBL')
            plt.plot([], 'c', label = 'EMGAMP')
            plt.plot([], 'g', label = 'LEMUR')
        plt.plot([], 'k', linestyle = list_line[j], label = 'p = ' + str(p))
        #plt.axis("off")
        plt.legend()
    plt.suptitle(str(iSNR) + " dB")
plt.show()