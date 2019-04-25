#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:02:03 2018

@author: raphaelromero
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.integrate as integrate
import time
from scipy.optimize import minimize

def extract_data(filename = "data.csv",plot_data = False):
    #extract and read the data
    df = pd.read_csv(filename)
    data = df.values
    data = data[:,1:].T
    # show the data
    n,d = np.shape(data)    
    # we now subsample the data so it fit our model
    new_index = np.sort(np.random.choice(d, 31, replace=False))
    datas = data[:,new_index]
    
    if(plot_data):
        plt.figure(figsize=(10, 7))
        for k in range(n-1):
            plt.plot(data[n-1,:],data[k,:], alpha=0.2)
            plt.title("Full Dataset")
        plt.show()
        # we plot the subsampled data
        for k in range(n-1):
            plt.plot(datas[n-1,:],datas[k,:], alpha=0.2)
            plt.title("Subsampled dataset")
        plt.show()
    return datas

def S(X,Y):
    """
    Computes the sufficient statistics associated with the observation X and the hidden variables stored in X
    X[:-2] is the beta parameter (deformation)
    X[-2] is the lambda parameter (scaling)
    X[-1] is the class index
    """
    I = X[-1] # Class index
    lam = X[-2] # lambda parameter
    beta = X[:-2] # deformation parameter
    phi_beta = PHI(beta) # deformation matrix
    tmp = {}
    
    
    tmp[1] = np.ones(2)
    tmp[2] = lam*np.dot(phi_beta.T,Y)
    tmp[3] = lam**2 *np.dot(phi_beta.T,phi_beta)
    tmp[4] = np.outer(beta,beta)
    tmp[5] = np.linalg.norm(Y)**2
    tmp[6] = lam
    tmp[7] = np.log(lam)
    s = {}
    for i in range(1,8):
        
        s[i] = np.zeros(np.shape(tmp[i])+(2,))
        s[i][...,int(I)] = tmp[i]
    
    return s

def init_s():
    s = {}

    s[1] = np.ones(2)
    s[2] = np.zeros((35, 2))
    s[3] = np.zeros((35,35,2))
    s[3][...,0] = np.eye(35)
    s[3][...,1] = np.eye(35)

    s[4] = np.zeros((2,2))
    s[5] = np.ones((1,2))
    s[6] = np.ones((1,2))
    s[7] = np.ones((1,2))
    return s
    
def MCoEM(dat,C):
    """
    Computes parameters for the hierachical model using the Monte Carlo online EM
    
    Parameters
    ----------
    dat : observations (N arrays of dimension 31)
    C : Number of classes
    Returns
    -------
    theta: The updated parameters
    """
    N = np.shape(dat)[0] # Number of samples
    # Initialization
    # Sufficient stats
    
    theta = init_theta()
    s = init_s()
        
    for n in range(3):
        
        index = np.random.randint(N)
        obs = dat[index] # randomly chosen observation
        mit = 300
        # simulation step
        X = MCMC_simulate(obs,theta,miter=mit,iteration=100, C=2)
        
        # Stochastic approximation step
        rho = np.exp(-0.6*np.log(n+1))
        s = SA(obs,rho,X,s,mit)
    
        # Maximization step
        #
        theta = M(obs,s)


    return theta
def init_theta():
    theta= {}
    theta['alpha0'] = 5*np.random.rand(35)
    theta['alpha1'] = 3*np.random.rand(35) + 3
    theta['omega0'] = 0.4
    theta['omega1'] = 0.6
    theta['cov0'] = 0.08
    theta['cov1'] = 0.08
    theta['sigma'] = 1
    return theta
def SA(Y,rho,X,s,mit):
    """
    Performs the stochastic approximation step
    Parameters
    ----------
    Y : observation
    rho : learning rate of the stochastic approximation
    X : Missing data sampled from the markov chain
    s : current fit of the sufficient statistic
    mit : number of iterations of the markov chain
    Returns :
    snew : Updated sufficient statistic (dictionnary type)
    """
    snew = s
    for ind in range(1,8):
            for j in range(2):
                tmp = np.array([S(X[k],Y)[ind] for k in range(mit)])
                tmp1 = s[ind][...,j]*(1-rho) + np.mean(tmp)*rho
                snew[ind][...,j] = tmp1
    return snew

def M(Y,s):
    """
    Performs the maximization step of the MCoEM
    Parameters
    ----------
    Y : observation
    s : Updated fit of sufficient statistics
    Returns 
    -------
    theta : new fit of parameters
    """
    card = np.shape(Y)[0] # Dimension of the observation
    theta = {}
    theta['alpha0'] = (np.linalg.solve(s[3][...,0],s[2][...,0]))
    alpha0 = theta['alpha0']
    theta['alpha1'] = (np.linalg.solve(s[3][...,1],s[2][...,1]))
    alpha1 = theta['alpha1']
    theta['omega0'] = s[1][...,0]/np.sum(s[1][...,0])
    theta['omega1'] = s[1][...,1]/np.sum(s[1][...,1])
    d_beta = np.shape(s[4][...,0])[0]
    theta['cov0'] = (s[4][...,0]/(d_beta * s[1][...,0]))[0]
    theta['cov1'] = (s[4][...,1]/(d_beta * s[1][...,1]))[0]
    theta['sigma'] = ((1.0/card) * (-2*(np.dot(alpha0,s[2][...,0])
                                       +np.dot(alpha1,s[2][...,1]))
                                   +np.trace(np.dot(np.outer(alpha0,alpha0),s[3][...,0]))
                                   +np.trace(np.dot(np.outer(alpha1,alpha1),s[3][...,1]))
                                   +s[5][...,0]
                                   +s[5][...,1]))[0]
    return theta


s = init_s()
theta = M(y,s)
print(theta)