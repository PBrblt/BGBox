import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from matplotlib.patches import Patch

iSNR = 10
L = 0.06

p_list = [0.01, 0.05, 0.1]
#bars = ['Id', 'w=0', 'w=0.5', 'w=0.7', 'w=0.9']
bars = ['Id', 'sqr $\mu = 0$', 'rec $\mu = 0$', 'sqr $\mu = 0.4$', 'rec $\mu = 0.4$']

color_list = ['r', 'y', 'g','b','c']
hatch_list = [None,'/','-','x']

tag_list = ['False Positive', 'False Negative', 'output SNR', 'Absolute error of $p$', 'Absolute error of $\sigma_x^2$', 'Relative error of $\sigma_e^2$']

abcisse = np.arange(5)#5

#handle_list = [Patch(color='red', label='p = 0.01'), Patch(color='yellow', label='p = 0.05'), Patch(color='green', label='p = 0.1'), 
#               Patch(facecolor='white', edgecolor ='black', label='L0'), Patch(facecolor='white', edgecolor ='black', hatch = '//', label='M'),
#               Patch(facecolor='white', edgecolor ='black', hatch = '--', label='J'), Patch(facecolor='white', edgecolor ='black', hatch = 'xx', label='LEMUR')]
#
#handle_list2 = [Patch(color='red', label='p = 0.01'), Patch(color='yellow', label='p = 0.05'), Patch(color='green', label='p = 0.1'), 
#               Patch(facecolor='white', edgecolor ='black', label='Mom'), Patch(facecolor='white', edgecolor ='black', hatch = '//', label='M'),
#               Patch(facecolor='white', edgecolor ='black', hatch = '--', label='J'), Patch(facecolor='white', edgecolor ='black', hatch = 'xx', label='LEMUR')]

handle_list = [Patch(color='r', label='$\ell_0$'), Patch(color='y', label='$J_z$'), Patch(color='g', label='$J_x$'), 
               Patch(color='b', edgecolor ='black', label='LEMUR'), Patch(color='c', edgecolor ='black', label='BGAMP'), Patch(facecolor='white', edgecolor ='black', label='p=0.01'),
               Patch(facecolor='white', edgecolor ='black', hatch = '//', label='p=0.05'), Patch(facecolor='white', edgecolor ='black', hatch = '--', label='p=0.1')]

handle_list2 = [Patch(color='r', label='Mom'), Patch(color='y', label='$J_z$'), Patch(color='g', label='$J_x$'), 
               Patch(color='b', edgecolor ='black', label='LEMUR'), Patch(color='c', edgecolor ='black', label='BGAMP'), Patch(facecolor='white', edgecolor ='black', label='p=0.01'),
               Patch(facecolor='white', edgecolor ='black', hatch = '//', label='p=0.05'), Patch(facecolor='white', edgecolor ='black', hatch = '--', label='p=0.1')]

for i in range(len(p_list)) :
    p = p_list[i]

    res_em = np.load("Results/res_" + str(iSNR) + "_" + str(i) + '.npy')
    
    for k in range(3) :
        plt.figure(k)
        for j in range(4):
            #plt.bar(abcisse + j * 0.2 + i * 0.06, res_em[j, :, k], width = L, color = color_list[i], hatch = hatch_list[j]) #Reaggae bar
            plt.bar(abcisse + i * 0.32 + j * 0.06, res_em[j, :, k], width = L, color = color_list[j], hatch = hatch_list[i])
        plt.ylabel(tag_list[k])
        plt.title('Input SNR = ' + str(iSNR) + ' dB')
        plt.xticks(abcisse + 0.4, bars, rotation=15, fontweight='bold')
        
        if k == 2:
            plt.legend(handles = handle_list)
        
        if k < 2 :
            plt.yscale('log')
        plt.xlim(-0.2,5)
        #plt.show()
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
    print(res_em[:,:,4])
    for k in range(3,6) :
        plt.figure(k)
        for j in range(4):
            #plt.bar(abcisse + j * 0.2 + i * 0.06, abs(res_em[j, :, k]), width = L, color = color_list[i], hatch = hatch_list[j]) #Reggea bars
            plt.bar(abcisse + i * 0.32 + j * 0.06, res_em[j, :, k], width = L, color = color_list[j], hatch = hatch_list[i])
        plt.ylabel(tag_list[k])
        plt.title('Input SNR = ' + str(iSNR) + ' dB')
        plt.xticks(abcisse + 0.4, bars, rotation=15, fontweight='bold')
        if k==5 :
            plt.legend(handles = handle_list2)
        plt.yscale('log')
        plt.xlim(-0.2,5)
        #plt.show()
        
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

res_amp = loadmat("Results/Res_AMP_" + str(iSNR))['Res']
print(np.shape(res_amp))

print(res_amp[:, :, 4])

for k in range(6) :
    plt.figure(k)
    for i in range(len(p_list)) :
        plt.bar(abcisse + i * 0.32 + 4 * 0.06, res_amp[i, :, k], width = L, color = 'c', hatch = hatch_list[i])

plt.show()