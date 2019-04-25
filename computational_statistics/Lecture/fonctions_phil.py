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

#extract and read the data
df = pd.read_csv("data.csv")
data = df.values
data = data[:,1:].T
# show the data
n,d = np.shape(data)
plt.figure(figsize=(10, 7))
for k in range(n-1):
    plt.plot(data[n-1,:],data[k,:], alpha=0.2)
plt.show()

# we now subsample the data so it fit our model
new_index = np.sort(np.random.choice(d, 31, replace=False))
datas = data[:-1,new_index]

plt.figure(figsize=(10, 7))

# we plot the results
for k in range(n-1):
    plt.plot(datas[n-1,:],datas[k,:], alpha=0.2)
plt.show()

#we now define the model
m = 35 # the size of our basis
r = np.linspace(2,18,m, endpoint = False) # the centers of our basis function phi
u = datas[n-1,:] # our index of sample
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
    la valeur de l'intégrale
    
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

def Dphi(u,l):
    """
    calcule la dérivée de phi_l(u), avec phi_l le l-ième elements de la base de fonction du
    modèle: ici des noyaux gaussiens
    """
    return -2*(u-r[l])*phi(u,l)/np.sqrt(nu[l])
    
    
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
    #plt.plot(lamb*np.dot(mat,alpha))
    #plt.plot(Y)
    #plt.show()
    #result
    res = -(np.linalg.norm(vectortemp)**2)/(2*sigma**2)
    
    return res

def logp(theta,I,X):
    """
    retourne la densité p_theta(y|X,I) à une constante de normalisation près
    """
    #our variables
    lamb = X[0]
    beta = X[1:]
    
    #our parameters
    cov = np.array([theta['cov0'],theta['cov1']])
    cov = (1/cov[I])*np.eye(dbeta)
    
    #derive result
    tmp = np.dot(beta,np.dot(cov,beta))
    """if lamb <= 0.0001:
        res = -10**8
    else :"""
    if lamb <= 0 :
        return -np.Inf
    
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
    if lamb <= 0 :
        return -np.Inf
    tmp = tmp - 2*b*lam + 2*(9)*np.log(lam)
    
    return -tmp


bound = [(None,None) for k in range(dbeta+1)]
bound[0] = (0.001, None)


def pseudo_priors_centers_fun(theta,Y,X,j):
    return -(logp(theta,j,X)+logg(theta,Y,j,X)) #, -0.5*gradOBJ(theta,I,Y,X)

def pseudo_prior_centers2(theta,j,Y,Xinit):
    obj = lambda x : pseudo_priors_centers_fun(theta,Y,x,j)
    res = minimize(obj, Xinit, jac = False, method = "BFGS",
                   options={'maxiter':70, 'disp': True})
    print(res)
    return res
    
"""moi"""
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
    
    theta = init_theta()
    
        
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
"""fin moi"""

def pseudo_prior_centers(sigma,alpha,a,b,cov,y,j, Xinit):
    """
    donne la moyenne du psudo prior j
    """
    n = np.shape(Xinit)[0] - 1
    v = cov[j]
    alpha = alpha[j]
    obj = lambda x : pseudo_priors_centers_f(sigma,alpha,a,b,v,y, x[0] , x[1:])
    res = minimize(obj, Xinit,  method = "L-BFGS-B", bounds = bound,
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
    
    norm = logprob[k] + np.log(1+np.exp(sum(logprob)-2*logprob[k]))

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
            if (y [0] > 0.0001):
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
        Xinit[0] = 1
        res = pseudo_prior_centers2(theta,k,Y,Xinit)
        #moy = pseudo_prior_centers(sigma,alpha,a,b,cov,Y,k,Xinit)
        #moy = Xinit
        covM = 0.05*np.eye(21)
        pseudo_prior[k] = multivariate_normal(res.x,covM)
    
    
    
    #initialize sample
    for k in range(C):
        law = pseudo_prior[k]
        lock = True
        while(lock):
            tmp = law.rvs()
            if ( tmp[0] > 0.0001):
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

"""
On code les fonctions nécessaires pour calculer le gradient de pi(0|y)
"""
def fbidon(x,a,b,c,d):
    return (a*x-a/(c**2) + b/c)*np.exp(c*x+d)

def Intexpaffine(a,b,c,d,sup,inf):
    if c == 0 :
        return b * np.exp(d)*(sup-inf)+ 0.5*a*(sup**2-inf**2)*np.exp(d)
    else :
        return fbidon(sup,a,b,c,d)-fbidon(inf,a,b,c,d)

def betacoef(i):
    if i == 0:
        a = (beta[i+1]+beta[i+2]-beta[i])/2.1
        b = (beta[i+1]+beta[i+2]+beta[i])+(beta[i]*q[i]-beta[i+1]*q[i+1]-beta[i+2]*q[i+2])/2.1
        #tmp = x*a + b
    elif i == 18:
        a = (beta[i+1]-beta[i]-beta[i-1])/2.1
        b = (beta[i+1]+beta[i-1]+beta[i])+(beta[i]*q[i]-beta[i+1]*q[i+1]+beta[i-1]*q[i-1])/2.1
        #tmp = x*a + b
    else :
        a = (beta[i+1]+beta[i+2]-beta[i]-beta[i-1])/2.1
        b = (beta[i+1]+beta[i+2]+beta[i-1]+beta[i])+(beta[i]*q[i]-beta[i+1]*q[i+1]+beta[i-1]*q[i-1]-beta[i+2]*q[i+2])/2.1
        #tmp = x*a + b
    return a, b

def pseudoIntDer(u,beta,k):
    id0 = max(k-2,0)
    id1 = max(k-1,0)
    id2 = min(k,dbeta-2)
    id3 = min(dbeta-2,k+1)
    id4 = min(dbeta-2,k+2)
    c0, d0 = betacoef(id0)
    c1, d1 = betacoef(id1)
    c2, d2 = betacoef(id2)
    c3, d3 = betacoef(id3)
    a0 = a1 = 1/2.1
    a2 = a3 = -1/2.1
    b0 = b1 = 1-q[k]/2.1
    b2 = b3 = 1+q[k]/2.1
    res0 = Intexpaffine(a0,b0,c0,d0,min(q[id1],u),min(q[id0],u))
    res1 = Intexpaffine(a1,b1,c1,d1,min(q[id2],u),min(q[id1],u))
    res2 = Intexpaffine(a2,b2,c2,d2,min(q[id3],u),min(q[id2],u))
    res3 = Intexpaffine(a3,b3,c3,d3,min(q[id4],u),min(q[id3],u))
    
    return res0+res1+res2+res3

def Hnum(u,beta):
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
    return num,tmp
    

def DD(u,beta,k):
    denom, num = Hnum(u,beta)
    num1 = pseudoIntDer(u,beta,k)
    num2 = pseudoIntDer(q[dbeta-1],beta,k)
    
    return 16*(num1*denom-num*num2)/denom**2

def DPHI(beta,k):
    tmp1 = np.array([D(u[i],beta) for i in range(31)])
    tmp2 = np.array([DD(u[j],beta,k) for j in range(31)])
    res = np.zeros([31,35])
    for j in range(35):
        for i in range(31):
            res[i,j] = Dphi(tmp1[i], j)*tmp2[i]
    return res
    
    
    

def gradcoordinate(theta,I,Y,X,k, M):
    alpha = np.array([theta['alpha0'],theta['alpha1']])
    alpha = alpha[I]
    sigma = theta['sigma']
    cov = np.array([theta['cov0'],theta['cov1']])
    cov= cov[I] 
    beta = X[1:]
    lamb = X[0]
    if k == 0:
        #M = PHI(beta)
        ALPHA = np.reshape(alpha,[35,1])*np.reshape(alpha,[1,35])
        MMt = np.dot(M.T, M)
        res = 2*np.dot(alpha,np.dot(M.T,Y))/sigma**2
        res = res -  2 * lamb * np.trace(np.dot(ALPHA,MMt)) - 2*b -2*(a-1)/lamb
    else :
        DM = DPHI(beta,k-1)
        res = 2*lamb*np.dot(alpha,np.dot(DM.T,Y))/sigma**2 - 2*X[k]/cov
        
        tmp = 2*np.dot(np.dot(DM,alpha),np.dot(M,alpha))
        
        res = res - tmp*(lamb/sigma)**2
    return res
        
       
def gradOBJ(theta,I,Y,X):
    M = PHI(beta)
    res = np.array([gradcoordinate(theta,I,Y,X,k, M) for k in range(21)])
    return res
    
    
"""
        
# ON TESTE LE CODE
 
#essai :
#variable
beta = np.ones(20) + 0.6*np.random.randn(20)
X = np.ones(21)
X[1:] = beta
Y = datas[5,:]
I = 0
C = 2

# paramètres
theta= {}
test= np.dot(np.linalg.pinv(PHI(beta)),Y)
theta['alpha0'] = 5*np.random.rand(35)
theta['alpha1'] = 3*np.random.rand(35) + 3
theta['omega0'] = 0.4
theta['omega1'] = 0.6
theta['cov0'] = 0.08
theta['cov1'] = 0.08
theta['sigma'] = 1

res = logg(theta,Y,I,X)
print res

res = logp(theta,I,X)
print res

#pseudo_prior_centers2(theta,I,Y,X)
Xnew = MCMC_simulate(Y,theta,300,60, C=2)
"""
"""
for k in range(5):
    #abstest = np.linspace(2,18,200)
    #test = np.array([phi(tt,k) for tt in abstest])
    beta = np.zeros(21) + 0.6*np.random.randn(21)
    X = np.ones(21)
    X[1:] = beta[1:]
    X[0] = max(0.2,beta[0])
    
    #plt.show()
    print  "fonction", pseudo_priors_centers_fun(theta,Y,X,I)
    print  "logg", logg(theta,Y,I,X)
    print  "logp", logp(theta,I,X)
    
tmp = np.array([D(u[i],beta) for i in range(31)])
plt.plot(u,tmp)
plt.show()
    
for k in range(35) :
    ttt = np.array([phi(t,k) for t in tmp])
    plt.plot(tmp,ttt)
    plt.show()
    



plt.plot(range(31),Y)
plt.show()
lamb = X[0]
random_curve = lamb*np.dot(PHI(beta),np.reshape(theta['alpha0'],[35,1]))
plt.plot(range(31), random_curve)
plt.show()

def ftmp(x):
    tmp = [phi(x,l) for l in range(35)]
    return sum(tmp)
te = np.array([ftmp(x) for x in r])
plt.plot(r,te)

plt.show()

res = logg(theta,Y,I,X)
print res

res = logp(theta,I,X)
print res

print gradOBJ(theta,I,Y,X)


alpha = np.array([theta['alpha0'],theta['alpha1']])
omega = np.array([theta['omega0'],theta['omega1']])
cov = np.array([theta['cov0'],theta['cov1']])
sigma = theta['sigma']

#X = np.zeros([m,43])
Xnew = MCMC_simulate(Y,theta,300,60, C=2)

#set the pseudo priors densities

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
    

    
    
    
