import numpy as np
from copy import copy

def cal_dE00(gt_lab, rec_lab):
    
    def cosd(angle):
        return np.cos(angle*np.pi/180)
    def sind(angle):
        return np.sin(angle*np.pi/180)
    
    L_1, a_1, b_1 = np.split(gt_lab, 3, axis=1)
    L_2, a_2, b_2 = np.split(rec_lab, 3, axis=1)

    C_1 = np.sqrt(a_1**2 + b_1**2)
    C_2 = np.sqrt(a_2**2 + b_2**2)
    
    dL = L_2 - L_1
    L_bar = (L_1 + L_2)/2
    C_bar = (C_1 + C_2)/2
    
    a_1 = a_1 + (a_1/2)*(1 - np.sqrt(C_bar**7/(C_bar**7 + 25**7)))
    a_2 = a_2 + (a_2/2)*(1 - np.sqrt(C_bar**7/(C_bar**7 + 25**7)))    
    C_1 = np.sqrt(a_1**2 + b_1**2)
    C_2 = np.sqrt(a_2**2 + b_2**2)
    C_bar = (C_1 + C_2)/2
    dC = C_2 - C_1

    h_1 = (np.arctan2(b_1, a_1) / np.pi*180) % 360
    h_2 = (np.arctan2(b_2, a_2) / np.pi*180) % 360
    
    zeros_1 = (a_1==0) * (b_1==0)
    zeros_2 = (a_2==0) * (b_2==0)    
    if np.sum(zeros_1) != 0:
        h_1[zeros_1] = 0
    if np.sum(zeros_2) != 0:
        h_2[zeros_2] = 0
    
    dh = np.zeros(h_1.shape)
    h_dist = np.abs(h_1 - h_2)
    cond_1 = C_1*C_2 == 0
    cond_2 = (C_1*C_2 != 0) * (h_dist <= 180)    
    cond_3 = (C_1*C_2 != 0) * (h_dist > 180) * (h_2 <= h_1)
    cond_4 = (C_1*C_2 != 0) * (h_dist > 180) * (h_2 > h_1)
    
    dh[cond_1] = 0
    dh[cond_2] = h_2[cond_2] - h_1[cond_2]
    dh[cond_3] = h_2[cond_3] - h_1[cond_3] + 360
    dh[cond_4] = h_2[cond_4] - h_1[cond_4] - 360
    
    dH = 2*np.sqrt(C_1*C_2) * sind(dh/2)
    
    H_bar = np.zeros(h_1.shape)
    cond_3 = (C_1*C_2 != 0) * (h_dist > 180) * ((h_1+h_2) < 360)
    cond_4 = (C_1*C_2 != 0) * (h_dist > 180) * ((h_1+h_2) >= 360)
    
    H_bar[cond_1] = h_1[cond_1] + h_2[cond_1]
    H_bar[cond_2] = (h_1[cond_2] + h_2[cond_2])/2
    H_bar[cond_3] = (h_1[cond_3] + h_2[cond_3] + 360)/2
    H_bar[cond_4] = (h_1[cond_4] + h_2[cond_4] - 360)/2
    
    T = 1 - 0.17*cosd(H_bar-30) + 0.24*cosd(2*H_bar) + 0.32*cosd(3*H_bar+6) - 0.2*cosd(4*H_bar-63) 
    S_L = 1 + (0.015*(L_bar-50)**2)/np.sqrt(20+(L_bar-50)**2)
    S_C = 1 + 0.045*C_bar
    S_H = 1 + 0.015*C_bar*T
    R_T = -2*np.sqrt((C_bar**7)/(C_bar**7+25**7)) * sind(60*np.exp(-((H_bar-275)/25)**2))
    
    return np.sqrt((dL/S_L)**2 + (dC/S_C)**2 + (dH/S_H)**2 + R_T*(dC/S_C)*(dH/S_H) )

def dE00(gt, rec, exposure=1):
    return cal_dE00(gt['lab'], rec['lab'])

def spec2lab(orig_spec, cmf, orig_wp_spectrum, new_wp_spectrum=()):
    def f(t):
        d = 6/29
        
        case_1 = t>d**3
        case_2 = ~case_1
        
        out = np.zeros(t.shape)
        out[case_1] = t[case_1]**3
        out[case_2] = t[case_2]/(3*d**2)+4/29
        
        return out
    
    spec = copy(orig_spec)
    if len(new_wp_spectrum):
        spec = spec / orig_wp_spectrum.reshape(1, -1) * new_wp_spectrum.reshape(1, -1)
        Xw, Yw, Zw = new_wp_spectrum.reshape(-1) @ cmf
    else:
        Xw, Yw, Zw = orig_wp_spectrum.reshape(-1) @ cmf
    
    XYZ = spec @ cmf    
    X, Y, Z = XYZ[:,0], XYZ[:,1], XYZ[:,2]
    
    Lab = np.zeros((spec.shape[0], 3))
    Lab[:, 0] = 116*f(Y/Yw)-16  # L
    Lab[:, 1] = 500*(f(X/Xw)-f(Y/Yw)) # a
    Lab[:, 2] = 200*(f(Y/Yw)-f(Z/Zw)) # b
    
    return Lab