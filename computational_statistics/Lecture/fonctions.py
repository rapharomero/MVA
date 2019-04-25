# -*- coding: utf-8 -*-
"""
Created on Sat Jan 06 16:55:37 2018

@author: Alexandre Philbert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.integrate as integrate
import time
from scipy.optimize import minimize

#preprocess our data_set so we can work with it 

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
    plt.figure(figsize=(10, 7))
    
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

datas = extract_data(filename = "data.csv",plot_data = False)

n = np.shape(datas)[0]
u = datas[n-1,:] # our index of sample

#we now define the model
m = 35 # the size of our basis,
r = np.linspace(2,18,m,endpoint = False) # the centers of our basis function phi
eps = 0.1 #value of epsilon
dbeta = 20
#prior parameters
a = 10 
b = 10

# we define our vector nu so we can create our basis phi
nu = np.zeros(m)
utmp = np.zeros(31)
for idx, k in enumerate(r) :
    utmp[0:] = u
    utmp[utmp == k] = -18
    tmp = min((utmp-k)**2)
    nu[idx] = -tmp/np.log(eps)
    
 # we define the parameters of the function D
mm = 0 # borne inf de l'intégrale H
ss = 20 # borne sup de l'intégrale H
mi = 2 # borne inf de l'espaces du temps U
su = 18 # borne sup de l'espca du temps U
q = np.linspace(mm,ss,20)  # centre des B_spline d'ordre 1 nous permettant de définir w 

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

def pseudoint(beta,x,i):
    """
    affin de faire les calculs dans un temps raisonnable, les noyaux gaussiens
    dans l'expression de H ont étés remplacé par des B-spline d'ordre 1 ce qui
    permet de calculer H sans passer par un calcul approcimé d'intégrale.
    pour calculer H directement, on subdivise l'itervalle d'intégration en 
    sous intervalles où l'intégrale à une closed form
    ---
    beta = vecteur des coefficients beta
    x = valeur de la borne sup de l'intégrale
    i = indice du sous intervalle sur lequel on effectue le calcul
    ---
    retourne
    la valeur dde l'intégrale
    
    """
    if i == 0:
        a = (beta[i+1]+beta[i+2]-beta[i])/2.1
        b = (beta[i+1]+beta[i+2]+beta[i])+(beta[i]*q[i]-beta[i+1]*q[i+1]-beta[i+2]*q[i+2])/2.1
        tmp = x*a + b
    elif i == 18:
        a = (beta[i+1]-beta[i]-beta[i-1])/2.1
        b = (beta[i+1]+beta[i-1]+beta[i])+(beta[i]*q[i]-beta[i+1]*q[i+1]+beta[i-1]*q[i-1])/2.1
        tmp = x*a + b
    else :
        a = (beta[i+1]+beta[i+2]-beta[i]-beta[i-1])/2.1
        b = (beta[i+1]+beta[i+2]+beta[i-1]+beta[i])+(beta[i]*q[i]-beta[i+1]*q[i+1]+beta[i-1]*q[i-1]-beta[i+2]*q[i+2])/2.1
        tmp = x*a + b
    if a == 0:
        return (x-q[i])*np.exp(b)
    else :
        return (np.exp(tmp)-np.exp(a*q[i]+b))/a
    
    
def H(u,beta):
    """
    calcul H avec l'approximation suivante : les gaussian kernels sont 
    remplacés par des B splines d'ordre 1
    ---
    u = la valeur en laquelle on estime H
    beta = paramètres beta
    ---
    retourne la valeur de H
    """
    k = 0
    tmp = 0
    while(u>q[k+1]):
        tmp += pseudoint(beta,q[k+1],k)
        #print(tmp,k)
        k+=1
        
    num = tmp +pseudoint(beta,u,k)
    while(k<=dbeta-2):
        tmp+=pseudoint(beta,q[k+1],k)
        k+=1
    return num/tmp

def D(u,beta):
    """
    calcule D, la fonction de déformation temporelle
    """
    return mi + (su - mi)*H(u,beta)


def phi(u,l):
    """
    calcule phi_l(u), avec phi_l le l-ième elements de la base de fonction du
    modèle: ici des noyaux gaussiens
    """
    return np.exp(-(1./nu[l])*(u-r[l])**2)

def PHI(beta):
    """
    calcule la matrice phi_beta 
    """
    #d = np.shape(beta)[0]
    n = np.shape(u)[0]
    res = np.zeros([n,m])
    tmp = np.array([D(u[i],beta) for i in range(n)])
    for j in range(m):
        for i in range(n):
            res[i,j] = phi(tmp[i], j)
    return res

def logg(theta,Y,I,X):
    """
    retourne la densité g_theta(y|X,I) à une constante de normalisation près
    """
    #define parameters
    alpha = np.array([theta['alpha0'],theta['alpha1']])
    sigma = theta['sigma']
    alpha = np.reshape(alpha[I],[35,1])
    
    #define our variables
    lamb = X[0]
    beta = X[1:]
    mat = PHI(beta)
    vectortemp = np.reshape(Y,[31,1]) - lamb*np.dot(mat,alpha)
    #result
    res = -(np.linalg.norm(vectortemp)**2)/(2*sigma**2)
    
    return res

def logp(theta,I,X):
    """
    retourne la densité p_theta(y|X,I) à une constante de normalisation près
    """
    #our variables
    lamb = X[0]
    beta = np.reshape(X[1:],[dbeta,1])
    
    #our parameters
    cov = np.array([theta['cov0'],theta['cov1']])
    cov = (1/cov[I])*np.eye(dbeta)
    
    #derive result
    tmp = np.dot(beta.T,np.dot(cov,beta))[0,0]
    """if lamb <= 0.0001:
        res = -10**8
    else :"""
    res = -0.5*tmp + (a-1)*np.log(lamb) - b*lamb
    
    return res

def pseudo_priors_centers_f(sigma,alpha,a,b,cov,y, lam, beta):
    """
    fonction objective à minimiser pour obtenir les paramètres du pseudo prior
    
    paramètres
    ---
    sigma, alpha, a, b, cov = paramètres actuels du problème ( correspondant à 
    un I donnée)
    y = observations
    lam, beta = variables latentes
    """
    invcov = (1/cov) *np.eye(dbeta)
    M = PHI(beta)
    n = np.shape(alpha)[0]
    ALPHA = np.reshape(alpha,[n,1])*np.reshape(alpha,[1,n])
    MMt = np.dot(M.T, M)
    tmp = 2*(lam/sigma**2) * np.dot(alpha,np.dot(M.T,y))
    tmp -= (lam/sigma)**2 * np.trace(np.dot(ALPHA, MMt))
    tmp -= np.dot(beta, np.dot(invcov, beta)) 
    tmp = tmp - 2*b*lam + 2*(a-1)*np.log(lam)
    
    return -tmp


bound = [(None,None) for k in range(dbeta+1)]
bound[0] = (0.001, None)

def pseudo_priors_centers_fun(theta,Y,X,j):
    return -(logp(theta,j,X)+logg(theta,Y,j,X))

def pseudo_prior_centers2(theta,j,Y,Xinit):
    obj = lambda x : pseudo_priors_centers_fun(theta,Y,x,j)
    res = minimize(obj, Xinit,method = "L-BFGS-B", bounds = bound,
                   options={'maxiter':100, 'disp': True})
    print(res)
    return res.x
    


def pseudo_prior_centers(sigma,alpha,a,b,cov,y,j, Xinit):
    """
    donne la moyenne du psudo prior j
    """
    n = np.shape(Xinit)[0] - 1
    v = cov[j]
    alpha = alpha[j]
    obj = lambda x : pseudo_priors_centers_f(sigma,alpha,a,b,v,y, x[0] , x[1:])
    res = minimize(obj, Xinit,method = "L-BFGS-B", bounds = bound,
                   options={ 'maxiter': 100, 'disp': True})
    print(res)
    return res.x

def sample_class(Y,theta,X,C,pseudo_prior):
    """
    Gibbs sampler for the class index
    
    Parameters
    ----------
    Y : observation
    theta : Current fit of parameters
    X : missing data sampled at the previous iteration
        X[i*21+1: (i+1)*21] is beta_i
        X[21*i]lambda_i
    
    pseudo_priors : the linked densities 
    Returns
    -------
    i : The class sampled 
    """
    
    xsize = dbeta+1
    omega = np.array([theta['omega0'],theta['omega1']])
    logprob = np.ones(C)
    for j in range(C):
        x = X[j*xsize:(j+1)*xsize]
        logprobtmp = logg(theta,Y,j,x)+logp(theta,j,x)+np.log(omega[j])
        
        #here we only considere C = 2
        if j==0 :
            logprobtmp = logprobtmp+pseudo_prior[1].logpdf(X[0:xsize])
        else :
            logprobtmp = logprobtmp+pseudo_prior[0].logpdf(X[xsize:2*xsize])
        logprob[j] = logprobtmp
    k = np.argmax(logprob)
    #print(logprob)
    print "logprob", logprob
    norm = logprob[k] + np.log(1+np.exp(sum(logprob)-2*logprob[k]))
    print "weights", logprob - norm
    weights = np.exp(logprob-norm)
    
    
        
    return np.random.choice(range(C), p=weights)


def RWMH(Y,theta,j, iteration , X):
    """
    Random Walk Metropolis-Hastings move 
    Parameters
    ----------
    Y : observation
    theta : Current fit of parameters
    j : index of the variable beeing updated
    iteration : number of iterations
    X : value of X to initialize the random walk
        
    Returns
    -------
    X : The updated hidden parameters
    """
    
    x = X
    alpha_moy = 0
    for k in range(iteration):
        #sample a new value
        lock = True
        while(lock):
            y = multivariate_normal.rvs(x,0.5*np.eye(dbeta+1))
            if (y [0] > 0):
                lock = False
        
        #y[y<=0] = 0.01
        #compute coefficient alpha
        logpiy = logg(theta,Y,j,y)+logp(theta,j,y)
        logpix = logg(theta,Y,j,x)+logp(theta,j,x)
        alpha = min (1, np.exp(logpiy-logpix))
        alpha_moy+=alpha
        u = np.random.rand()
        if (u <= alpha):
            x = y
            
    print(alpha_moy/iteration)
    return x

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
    s = {}

    s[1] = np.zeros(2)
    s[2] = np.zeros((35, 2))
    s[3] = np.zeros((35,35,2))
    s[4] = np.zeros((2,2))
    s[5] = np.ones((1,2))
    s[6] = np.ones((1,2))
    s[7] = np.ones((1,2))
    theta= {}
    beta = np.ones(20) + 0.6*np.random.randn(20)
    phi_beta  = PHI(beta)    

    test= np.dot(np.linalg.pinv(phi_beta),dat[0])
    theta['alpha0'] = test + 0.05*np.random.rand(35)
    theta['alpha1'] = test + 0.5*np.random.rand(35)
    theta['omega0'] = 0.4
    theta['omega1'] = 0.6
    theta['cov0'] = 0.08
    theta['cov1'] = 0.07
    theta['sigma'] = 1
    
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
def MCMC_simulate(Y,theta,miter,iteration, C=2):
    """
    Updates the sequence of parameters using the Monte Carlo online EM
    
    Parameters
    ----------
    theta: Current fit of parameters
    Y : observation (array of size 31)
    C : Number of classes
    miter : Length of the Markov Chain that will be using to sample the missing data
    Returns
    r : Number of iterations for the RWMH
    -------
    alphas,gammas,omegas,sigma : The updated parameters 
    """
    #get parameters
    
    alpha = np.array([theta['alpha0'],theta['alpha1']])
    omega = np.array([theta['omega0'],theta['omega1']])
    cov = np.array([theta['cov0'],theta['cov1']])
    sigma = theta['sigma']
    
    xsize = dbeta+1
    X = np.zeros([miter,2*xsize+1])
    
    #set the pseudo priors densities
    pseudo_prior = {}
    for k in range(C):
        Xinit = np.ones(xsize) + 0.6*np.random.randn(xsize)
        moy = pseudo_prior_centers2(theta,k,Y,Xinit)
        #moy = pseudo_prior_centers(sigma,alpha,a,b,cov,Y,k,Xinit)
        #moy = np.ones(dbeta+1)
        #moy[1:] = np.zeros(dbeta)
        pseudo_prior[k] = multivariate_normal(moy,cov[k]*np.eye(xsize))
    
    
    
    #initialize sample
    for k in range(C):
        law = pseudo_prior[k]
        lock = True
        while(lock):
            tmp = law.rvs()
            if ( tmp[0] > 0):
                lock = False
            
        X[0,k*xsize:(k+1)*xsize] = tmp
    
    for k in range(1,miter):
        X[k,2*xsize] = sample_class(Y,theta,X[k-1,:42],C, pseudo_prior)
        print(k,X[k,2*xsize])
        for j in range(C):
            if j != X[k,2*xsize] :
                law = pseudo_prior[j]
                X[k,j*xsize:(j+1)*xsize] = law.rvs()
            else :
                X[k,j*xsize:(j+1)*xsize] = RWMH(Y,theta,j,iteration,X[k-1,j*xsize:(j+1)*xsize])
    
    return X


#X = np.zeros([m,43])#
"""
Xnew = MCMC_simulate(Y,theta,300,60, C=2)"""
#
#set the pseudo priors densities
"""
pseudo_prior = {}
Xtest = np.zeros([m,43])

for k in range(C):
    Xinit = np.ones(21) + 0.6*np.random.randn(21)
    moy = pseudo_prior_centers(sigma,alpha,a,b,cov,Y,k, Xinit)
    print(moy)
    pseudo_prior[k] = multivariate_normal(moy,cov[k]*np.eye(21))

for k in range(C):
        law = pseudo_prior[k]
        tmp = law.rvs()
        tmp[tmp<=0]= 0.1
        Xtest[0,k*21:k*21+21] = tmp
Xtest[1,42] = sample_class(Y,theta,Xtest[0,:42],C, pseudo_prior)
for j in range(C):
            if j != Xtest[1,42] :
                law = pseudo_prior[j]
                Xtest[1,j*21:(j+1)*21] = law.rvs()
            else :
                Xtest[1,j*21:(j+1)*21] = RWMH(Y,theta,j,300,Xtest[0,j*21:(j+1)*21])

i = sample_class(Y,theta,Xtest[k-1,:42],C,pseudo_prior)
print i 


beta = np.ones(20) + 0.6*np.random.randn(20)
t = [u[k] for k in range(31)]
t0 = time.time()
be = [ D(u[k],beta) for k in range(31) ]
print( time.time()-t0)
plt.plot(t, be)
plt.show()




  
# let 's test the function logp
trys = [t for t in np.linspace(0,20,100)]
res0 = np.zeros(100)
res1 = np.zeros(100)
for k in range(100):
    X[0] = trys[k]
    res0[k] = logg(theta,Y,0,X)
    res1[k] = logg(theta,Y,1,X)
plt.plot(np.linspace(0,20,100),res0)
plt.show()
plt.plot(np.linspace(0,20,100),res1)
plt.show()"""
    

    
    
    
    