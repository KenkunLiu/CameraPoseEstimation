# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:25:09 2019

@author: Kenkun Liu
"""
# some useful functions for calculation
import numpy as np

# calculate rotation matrix given rotation axis
def expm(w):
    omega = w/np.linalg.norm(w)
    theta = np.linalg.norm(w)
    omega_hat = np.array([[0,-omega[2],omega[1]],[omega[2],0,-omega[0]],[-omega[1],omega[0],0]])
    R = np.identity(3)+omega_hat*np.sin(theta)+np.dot(omega_hat,omega_hat)*(1-np.cos(theta))
    return R

# calculate rigid body transformation matrix given rotation matrix and translation vector
def g_ab(R, P):
    P_temp = np.array([[P[0]],[P[1]],[P[2]]])
    RP = np.hstack((R,P_temp))
    g = np.vstack((RP,np.array([0,0,0,1])))
    return g

# calculate exponential coordinates given rigid body transformation matrix
def exp_coor(g):
    R = g[0:3,0:3]
    P = g[0:3,3]
    theta = np.arccos((np.trace(R)-1)/2)
    omega = (1/(2*np.sin(theta)))*np.array([[R[2][1]-R[1][2]],[R[0][2]-R[2][0]],[R[1][0]-R[0][1]]])
    omega_hat = np.array([[0,-omega[2][0],omega[1][0]],[omega[2][0],0,-omega[0][0]],[-omega[1][0],omega[0][0],0]])
    v = np.dot(np.linalg.inv(np.dot(np.identity(3)-R,omega_hat)+theta*np.dot(omega,np.transpose(omega))),P)
    return np.vstack((np.array([[v[0]],[v[1]],[v[2]]]),omega))