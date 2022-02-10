#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 18:01:39 2021

@author: pierrebarbault
"""
import numpy as np
from numpy.random import rand
from numpy.random import randn
from scipy.special import erf

# =============================================================================
# MISC
# =============================================================================

def SNR(u,v):
    """ Signal to Noise Ratio with u the pure signal and v the noisy one"""
    return(10*np.log10(np.mean(u**2)/np.mean((u-v)**2)))

# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def genBG(p,sigma_x,Nx,Ny):
    return (rand(Nx,Ny)<p)*(randn(Nx,Ny))*sigma_x

def noise(x,iSNR):
    """ Noise generator """

def genBG(p, sigma_x, Nx, Ny):
    """Generate a signal of shape (Nx,Ny) ~ BG(p,sigma_x^2)"""
    return (rand(Nx, Ny) < p) * (randn(Nx, Ny)) * sigma_x


def noise(x, iSNR):
    """Noise generator"""
    N = np.shape(x)
    sigma_e = np.sqrt(np.mean(x**2)*10**(-iSNR/10))
    y = x + sigma_e*randn(N[0],N[1])
    
    return(y,sigma_e)

def Phi(z,eta):
    return( (1 + erf(z/np.sqrt(2*eta)))/2 )

def genK(eta):
<<<<<<< Updated upstream
    
=======
    """Generate a simple discretized Gaussian convolution kernel"""
>>>>>>> Stashed changes
    N_e = 50
    i = np.arange(2*N_e+1)
    
    h_large = Phi(i- N_e + 0.5,eta) - Phi(i - N_e -0.5,eta)
    h = h_large[h_large!=0]
    
    Ix,Iy = np.meshgrid(h,h)
    K = Ix*Iy
    return(K)

# =============================================================================
# METHODS
# =============================================================================

<<<<<<< Updated upstream
def seuil(theta):
    p,sig_x,sig_e = theta
    s_x,s_e = sig_x**2,sig_e**2
    T_map = np.sqrt( 2* s_e/s_x * (s_e + s_x) * np.log( (1-p)/p * np.sqrt((s_e+s_x)/s_e) ) )
    T_mmse = np.sqrt( T_map**2 + 2* s_e/s_x * (s_e + s_x) * np.log((s_e+s_x)/(s_x - s_e) ) )
    return(T_map)
=======

def tresh(theta,mmse=False):
    """Compute treshold value"""
    p, sig_x, sig_e = theta
    s_x, s_e = sig_x ** 2, sig_e ** 2
    T_map = np.sqrt(
        2 * s_e / s_x * (s_e + s_x) * np.log((1 - p) / p * np.sqrt((s_e + s_x) / s_e))
    )
    T_mmse = np.sqrt(
        T_map ** 2 + 2 * s_e / s_x * (s_e + s_x) * np.log((s_e + s_x) / (s_x - s_e))
    )
    if mmse :
        return T_mmse
    else :
        return T_map
>>>>>>> Stashed changes

def grad(Hty,x,alpha,HtH,kerArgs):
    return( x + alpha * (Hty - HtH(x,kerArgs)) )

<<<<<<< Updated upstream
#def grad(Hty,K,x,alpha):
=======
def grad(Hty, x, alpha, HtH, kerArgs):
    """Compute the gradient descent of ||y-Hx||^2 with a weight alpha"""
    return x + alpha * (Hty - HtH(x, kerArgs))


# def grad(Hty,K,x,alpha):
>>>>>>> Stashed changes
#    return( x + alpha * (Hty - wbx.sub( signal.convolve2d(signal.convolve2d( wbx.up(x) ,K,'same'),K,'same') ) ) )

def prox0(w,T):
    return( w*(abs( w )>T) )

<<<<<<< Updated upstream
def prox1(w,T):
    return( (w - w/abs(w)*T)*(abs( w )>T) )

def moments(Y):
        """ Moments identification method for gaussian mixture. """
        
        m2 = np.mean(Y**2)
        m4 = np.mean(Y**4)/3
        m6 = np.mean(Y**6)/15
        
        A = m4 - m2**2
        B = m6/m2 - m2**2
        C = (B/A - 3)*m2
        
        D = (C**2)/4 + A
        D = max(0,D)
        X = -C/2 + np.sqrt(D)
        
        sigma_b = np.sqrt(abs(m2 - X))
#        sigma_x = np.sqrt(abs(A/X + X) )
        
        #sigma_b = np.sqrt(m2 - X)
        sigma_x = np.sqrt( A/X + X )
        p = X**2/(A+X**2)
        #p = min(p,1)
        return(p,sigma_x,sigma_b)

def em_step(Y,theta):
        """ EM method with x as complete data. """
        p,sigma_x,sigma_b = theta
        N = np.size(Y)
        #print(N)
        
#        q1 = p/(np.sqrt(2*np.pi*(sigma_x**2+sigma_b**2)))*np.exp(-Y**2/(2*(sigma_x**2+sigma_b**2)))
#        q0 = (1-p)/(np.sqrt(2*np.pi*(sigma_b**2)))*np.exp(-Y**2/(2*(sigma_b**2)))
#        Q = q0+q1
#        q1,q0 = q1/Q,q0/Q
        
        sig_2 = (sigma_x**2 + sigma_b**2) * sigma_b**2/sigma_x**2
        q1 = 1/ ( 1 + (1-p)/p * np.sqrt(1+sigma_x**2/sigma_b**2) * np.exp( -Y**2/(2*sig_2) ) )
        q0 = 1 - q1
        
        sigma_n = sigma_x**2 * sigma_b**2/(sigma_x**2 + sigma_b**2)
        X_hat = Y * sigma_x**2/(sigma_x**2 + sigma_b**2)
        Phi = q1/(q1+q0)
        
        
        p = np.sum(Phi)/N
        sigma_x = np.sqrt( np.sum(Phi*X_hat**2)/np.sum(Phi) + sigma_n )
        sigma_b = np.sqrt( np.sum(Phi*((Y-X_hat)**2 + sigma_n ))/N + np.sum((1-Phi)*Y**2)/N )
        
        X_eap = Y*Phi*sigma_x**2/(sigma_x**2+sigma_b**2)
        X_map = (q1>q0)*Y*np.sqrt( sigma_x**2/(sigma_x**2+sigma_b**2) )
        
        return([p,sigma_x,sigma_b],X_eap)

def ista(Hty,x_init,t,HtH,kerArgs,norme=0):
    
=======
def prox0(w, T):
    """Proximal descent of l_0 norm"""
    return w * (abs(w) > T)


def prox1(w, T):
    """Proximal descent of l_1 norm"""
    return (w - w / abs(w) * T) * (abs(w) > T)


def moments(Y):
    """Moments identification method for gaussian mixture."""

    m2 = np.mean(Y ** 2)
    m4 = np.mean(Y ** 4) / 3
    m6 = np.mean(Y ** 6) / 15

    A = m4 - m2 ** 2
    B = m6 / m2 - m2 ** 2
    C = (B / A - 3) * m2

    D = (C ** 2) / 4 + A
    D = max(0, D)
    X = -C / 2 + np.sqrt(D)

    sigma_b = np.sqrt(abs(m2 - X))
    #        sigma_x = np.sqrt(abs(A/X + X) )

    # sigma_b = np.sqrt(m2 - X)
    sigma_x = np.sqrt(A / X + X)
    p = X ** 2 / (A + X ** 2)
    # p = min(p,1)
    return (p, sigma_x, sigma_b)


def em_step(Y, theta):
    """EM update with x as complete data."""
    p, sigma_x, sigma_b = theta
    N = np.size(Y)
    # print(N)

    #        q1 = p/(np.sqrt(2*np.pi*(sigma_x**2+sigma_b**2)))*np.exp(-Y**2/(2*(sigma_x**2+sigma_b**2)))
    #        q0 = (1-p)/(np.sqrt(2*np.pi*(sigma_b**2)))*np.exp(-Y**2/(2*(sigma_b**2)))
    #        Q = q0+q1
    #        q1,q0 = q1/Q,q0/Q

    sig_2 = (sigma_x ** 2 + sigma_b ** 2) * sigma_b ** 2 / sigma_x ** 2
    q1 = 1 / (
        1
        + (1 - p)
        / p
        * np.sqrt(1 + sigma_x ** 2 / sigma_b ** 2)
        * np.exp(-(Y ** 2) / (2 * sig_2))
    )
    q0 = 1 - q1

    sigma_n = sigma_x ** 2 * sigma_b ** 2 / (sigma_x ** 2 + sigma_b ** 2)
    X_hat = Y * sigma_x ** 2 / (sigma_x ** 2 + sigma_b ** 2)
    Phi = q1 / (q1 + q0)

    p = np.sum(Phi) / N
    sigma_x = np.sqrt(np.sum(Phi * X_hat ** 2) / np.sum(Phi) + sigma_n)
    sigma_b = np.sqrt(
        np.sum(Phi * ((Y - X_hat) ** 2 + sigma_n)) / N + np.sum((1 - Phi) * Y ** 2) / N
    )

    X_eap = Y * Phi * sigma_x ** 2 / (sigma_x ** 2 + sigma_b ** 2)
    #X_map = (q1 > q0) * Y * np.sqrt(sigma_x ** 2 / (sigma_x ** 2 + sigma_b ** 2))

    return ([p, sigma_x, sigma_b], X_eap)


def ista(Hty, x_init, t, HtH, kerArgs, norme=0):

>>>>>>> Stashed changes
    x = np.copy(x_init)
    
    norm = 1.0
    
    c = 1/norm
    
    go = True
    while go:
        
        w = grad(Hty,x,c,HtH,kerArgs)
        
        if norme == 0 :
            u = prox0(w,t)#L0
        if norme == 1 :
            u = prox1(w,t)#L1
        
        critere = np.mean((u-x)**2)/np.mean(x**2)
        #print(critere)
        x = 1*u
        
        go = critere > 10.0**(-6)
            
    return(x)

def fista(Hty,x_init,t,HtH,kerArgs,norme=0):
    
    x = np.copy(x_init)
    u = 1*x
    
    norm = 1.0
    
    c = 1/norm
    
    go = True
    b = 1.0
    
    while go:
        
        w = grad(Hty,x,c,HtH,kerArgs)
        
        if norme == 0 :
            next_u = prox0(w,t)#L0
        if norme == 1 :
            next_u = prox1(w,t)#L1
        
        next_b = ( 1 + np.sqrt(1 + 4*b**2))/2
        x = next_u + (b - 1)/next_b * ( next_u - u)
        
        u = 1*next_u
        critere = np.mean((u-x)**2)/np.mean(x**2)
        
        go = critere > 10.0**(-6)
    return(u)

#def fista_t(t,img_y,K,x_init,norme=0):
#    
#    Hty = wbx.sub( signal.convolve2d( img_y ,K,'same') )
#    x = np.copy(x_init)
#    v = 1*x
#    
#    norm = 1.0
#    
#    c = 1/norm
#    
#    go = True
#    k = 1
#    
#    while go:
#        
#        #t = 2/(k + 1)
#        t = 0.9
#        #a,b = b,1 + np.sqrt(0.25 + b**2)
#        
#        z = x + t*( v - x ) #(1-t)*x + t*v
#        
#        w = grad(Hty,K,z,c )
#        
#        if norme == 0 :
#            u = prox0(w,t)#L0
#        if norme == 1 :
#            u = prox1(w,t)#L1
#        
#        v = u + 1/t * ( u - x )
#        #v = u + 0.9 * ( x - u )
#        
#        critere = np.mean((u-x)**2)/np.mean(x**2)
#        #print(cpt)
#        k = k + 1
#        
#        x = 1*u
#        
#        go = critere > 10.0**(-6)
#    #print(cpt)
#    #print(np.mean(abs(x_init-x)) )
#    return(x)

<<<<<<< Updated upstream
def emista(Hty,HtH,kerArgs):
    
    #Hty = wbx.sub( signal.convolve2d( img_y ,K,'same') )
=======

def emista(Hty, HtH, kerArgs, em_init = moments):

    # Hty = wbx.sub( signal.convolve2d( img_y ,K,'same') )
>>>>>>> Stashed changes
    x = 1 * Hty
    
    norm = 1.0#14.28
    #print(norm)
    
    alpha = 1/norm
    #print("Alpha : " +str(alpha) )
    
    theta_p = [0,0,0]
    go = True
    
    while go:
<<<<<<< Updated upstream
        
        z = grad(Hty,x,alpha,HtH,kerArgs)
        
        theta = moments(z)
        while np.mean((np.array(theta_p)/np.array(theta)-1)**2)> 10.0**(-12):
            theta_p = 1*theta
            theta,u = em_step(z,theta)
        
        critere = np.mean((u-x)**2)/np.mean(x**2)
        x = 1*u
        go = critere>10.0**(-6)
    
    return(theta,x)

#def fista(Hty,T,x_true,HtH,kerArgs):
=======

        z = grad(Hty, x, alpha, HtH, kerArgs)

        theta = em_init(z)
        while np.mean((np.array(theta_p) / np.array(theta) - 1) ** 2) > 10.0 ** (-12):
            theta_p = 1 * theta
            theta, u = em_step(z, theta)

        critere = np.mean((u - x) ** 2) / np.mean(x ** 2)
        x = 1 * u
        go = critere > 10.0 ** (-6)

    return theta, x


# def fista(Hty,T,x_true,HtH,kerArgs):
>>>>>>> Stashed changes
#    N_t = len(T)
#    res_0 = np.zeros(N_t)
#    res_1 = 0*res_0#np.zeros(N_t)
#    sup_0 = 0*res_0#np.zeros(N_t)
#    sup_1 = 0*res_0#np.zeros(N_t)
#    supp = 1*(x_true!=0)
#    x_0 = 0*x_true
#    x_1 = 0*x_true
#    for j in range(N_t):
#        t = T[j]
#        #print(j*100/N_t)
#        x_0 = bg.fista(Hty,x_0,t,HtH,kerArgs,0)
#        res_0[j] = bg.SNR(x_true,x_0)
#        sup_0[j] = np.mean( abs( supp-1*(x_0!=0) ) )
#        
#        x_1 = bg.fista(Hty,x_1,t,HtH,kerArgs,1)
#        res_1[j] = bg.SNR(x_true,x_1)
#        sup_1[j] = np.mean( abs( supp-1*(x_1!=0) ) )
#    return(res_0,res_1,sup_0,sup_1)

<<<<<<< Updated upstream
def courbe(Hty,T,x_true,HtH,kerArgs,mode='fista'):
=======

def em_vs_ista(Hty, T, x_true, HtH, kerArgs, mode="fista"):
    """Compute the result of (F)ISTA for an array of T treshold values."""
>>>>>>> Stashed changes
    N_t = len(T)
    res_0 = np.zeros(N_t)
    res_1 = 0*res_0#np.zeros(N_t)
    sup_0 = 0*res_0#np.zeros(N_t)
    sup_1 = 0*res_0#np.zeros(N_t)
    supp = 1*(x_true!=0)
    x_0 = 0*x_true
    x_1 = 0*x_true
    if mode == 'fista' :
        for j in range(N_t):
            t = T[j]
            #print(j*100/N_t)
            x_0 = fista(Hty,x_0,t,HtH,kerArgs,0)
            res_0[j] = SNR(x_true,x_0)
            sup_0[j] = np.mean( abs( supp-1*(x_0!=0) ) )
            
            x_1 = fista(Hty,x_1,t,HtH,kerArgs,1)
            res_1[j] = SNR(x_true,x_1)
            sup_1[j] = np.mean( abs( supp-1*(x_1!=0) ) )
    elif mode == 'ista' :
        for j in range(N_t):
            t = T[j]
<<<<<<< Updated upstream
            #print(j*100/N_t)
            x_0 = fista(Hty,x_0,t,HtH,kerArgs,0)
            res_0[j] = SNR(x_true,x_0)
            sup_0[j] = np.mean( abs( supp-1*(x_0!=0) ) )
            
            x_1 = fista(Hty,x_1,t,HtH,kerArgs,1)
            res_1[j] = SNR(x_true,x_1)
            sup_1[j] = np.mean( abs( supp-1*(x_1!=0) ) )
    return(res_0,res_1,sup_0,sup_1)
=======
            # print(j*100/N_t)
            x_0 = fista(Hty, x_0, t, HtH, kerArgs, 0)
            res_0[j] = SNR(x_true, x_0)
            sup_0[j] = np.mean(abs(supp - 1 * (x_0 != 0)))

            x_1 = fista(Hty, x_1, t, HtH, kerArgs, 1)
            res_1[j] = SNR(x_true, x_1)
            sup_1[j] = np.mean(abs(supp - 1 * (x_1 != 0)))
    return res_0, res_1, sup_0, sup_1


if __name__ == '__main__':
    from scipy.signal import convolve2d
    p = 0.05
    sigma_x = 1.
    Nx = 30
    Ny = 20
    x = genBG(p, sigma_x, Nx, Ny)
    
    eta = 0.25 #Blurr parameter (0 = no blurr)
    Kernel = genK(eta)
    
    iSNR = 10.
    y, sigma_e = noise(convolve2d(x, Kernel, 'same'), iSNR)
    
    #Estimation example :
    x_0 = 0*x
    
    Hty = convolve2d(y,Kernel,'same')
    def HtH(x, K):
        return convolve2d(convolve2d(x,K,'same'),K,'same')
    theta, x_em = emista(Hty, HtH, Kernel)
    T = tresh(theta)
    x_fista_0 = fista(y, x_0, T, HtH, Kernel, 0)
    x_fista_1 = fista(y, x_0, T, HtH, Kernel, 1)
    
    plt.figure()
    plt.subplot(232)
    plt.imshow(x)
    plt.title("Source signal")
    plt.subplot(233)
    plt.imshow(y)
    plt.title("Blurred noisy signal iSNR="+str(iSNR))
    plt.subplot(234)
    plt.imshow(x_em)
    plt.title("EM estimation SNR="+str(round(SNR(x,x_em),2)))
    plt.subplot(235)
    plt.imshow(x_fista_0)
    plt.title("ISTA L0 estimation (T estimated via EM) SNR="+str(round(SNR(x,x_fista_0),2)))
    plt.subplot(236)
    plt.imshow(x_fista_1)
    plt.title("ISTA L1 estimation (T estimated via EM) SNR="+str(round(SNR(x,x_fista_1),2)))
    plt.show()
>>>>>>> Stashed changes
