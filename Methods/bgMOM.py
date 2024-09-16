#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pierrebarbault
"""
import numpy as np

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