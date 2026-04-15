# EXERCISE 10 - EM FOR BAYESIAN NETWORKS
# EXACT CODE AS PER MANUAL (PAGES 74-81)

import math
import numpy as np
import matplotlib.pyplot as plt

# Helper Functions

def nCr(n,r):
    # A naive implementation for calculating the combination number C_n^r.
    # Args:
    # n: int, the total number
    # r: int, the number of selected
    # Returns: C_n^r, the combination number
    f = math.factorial
    return f(n)/f(r)/f(n-r)

def binomial(x,n,p):
    # The binomial distribution C_n^x p^x (1-p)^(n-x)
    # Args:
    # n: int, the total number
    # x: int, the number of selected
    # p: float, the probability
    # Returns: The binomial probability C_n^x p^x (1-p)^(n-x)
    return nCr(n,x) * (p**x) * ((1-p)**(n-x))

def log_likelihood(X,n,theta):
    # Calculates the log likelihood function for the two coins problem.
    # Args:
    # X: np.array of shape (n_trials,), dtype int,
    # the observations (number of heads at each trial).
    # n: int, total number of tosses per trial.
    # theta: tuple of (lambda, pA, pB), where
    # - lambda: float, the prior probability of selecting coin A (=1/2)
    # - pA: float, coin A's probability of showing head
    # - pB: float, coin B's probability of showing head
    # Returns: log-likelihood f(theta) = sum_i log sum_zi P(xi, zi; theta)
    # = sum_i log[ lam ( nCr(10, xi) pA^x1 (1-pA)^(10-xi) ) + (1-lam) ( nCr(10, xi) pB^x1 (1-pB)^(10-xi) ]
    (lam,p1,p2)=theta
    ll=0
    for x in X:
        ll+=np.log( lam*binomial(x,n,p1)+(1-lam)*binomial(x,n,p2) )
    return ll

def ELBO(X,n,Q,theta):
    # Calculates the ELBO for the two coins problem.
    # Args:
    # X: np.array of shape (n_trials,), dtype int,
    # the observations (number of heads at each trial).
    # n: int, total number of tosses per trial.
    # Q: np.array of shape (n_trials, 2), dtype float,
    # the hidden posterior q(z) (z = A, B) computed in the E-step.
    # theta: tuple of (lambda, pA, pB), where
    # - lambda: float, the prior probability of selecting coin A (=1/2)
    # - pA: float, coin A's probability of showing head
    # - pB: float, coin B's probability of showing head
    # Returns: ELBO (Evidence Lower Bound)
    (lam,p1,p2)=theta
    elbo=0
    for i,x in enumerate(X):
        elbo+=Q[i,0]*np.log(lam*binomial(x,n,p1)/Q[i,0])
        elbo+=Q[i,1]*np.log((1-lam)*binomial(x,n,p2)/Q[i,1])
    return elbo

def plot_coin_function(grid_fn,title,path=[]):
    # Plots a function wrt pA, pB using 2D contours.
    # Reference: https://nbviewer.jupyter.org/github/eecs445-f16/umich-eecs445-f16/blob/master/handsOn_lecture17_clustering-mixtures-em/handsOn_lecture17_clustering-mixtures-em.ipynb#Problem:-implement-EM-for-Coin-Flips
    # Args:
    # grid_fn: callable, a function that takes pA, pB as inputs and returns the function value at that point.
    # title: string, title of the plot.
    # path: (optional) A list of tuple of (pA, pB) that are visited in the EM iterations.
    # Visualized as line segments if not empty.
    # Returns: Shows the figure and returns None.
    xvals=np.linspace(0.01,0.99,100)
    yvals=np.linspace(0.01,0.99,100)
    xx,yy=np.meshgrid(xvals,yvals)
    grid=np.zeros([len(xvals),len(yvals)])
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            grid[j,i]=grid_fn(xvals[i],yvals[j])
    plt.figure(figsize=(6,4.5),dpi=100)
    C=plt.contour(xx,yy,grid,1000)
    cbar=plt.colorbar(C)
    plt.title(title,fontsize=15)
    plt.xlabel(r"$p_A$",fontsize=12)
    plt.ylabel(r"$p_B$",fontsize=12)
    if path:
        p1,p2=zip(*path)
        plt.plot(p1,p2,'g+-')
        plt.text(p1[0]-0.15,p2[0]-0.05,'start:\n$p_A$={}\n$p_B$={}'.format(p1[0],p2[0]),color='green',size=10)
        plt.text(p1[-1]+0.0,p2[-1]+0.02,'end:\n$p_A$={:.3f}\n$p_B$={:.3f}'.format(p1[-1],p2[-1]),color='green',size=10)
    plt.show()

def plot_coin_likelihood(X,n,path=[]):
    # Plots the coin likelihood wrt pA, pB using 2D contours.
    # Args:
    # X: np.array of shape (n_trials,), dtype int,
    # the observations (number of heads at each trial).
    # n: int, total number of tosses per trial.
    # path: (optional) A list of tuple of (pA, pB) that are visited in the EM iterations.
    # Visualized as line segments if not empty.
    # Returns: Shows the figure and returns None.
    grid_fn=lambda pA,pB: log_likelihood(X,n,(0.5,pA,pB))
    return plot_coin_function(
        grid_fn=grid_fn,
        title=r"Log-Likelihood $\log p(\mathcal{X}|p_A,p_B)$",
        path=path)

def plot_coin_ELBO(X,n,Q,path=None):
    # Plots the coin ELBO wrt pA, pB using 2D contours.
    # Reference: https://nbviewer.jupyter.org/github/eecs445-f16/umich-eecs445-f16/blob/master/handsOn_lecture17_clustering-mixtures-em/handsOn_lecture17_clustering-mixtures-em.ipynb#Problem:-implement-EM-for-Coin-Flips
    # Plots the coin ELBO wrt pA, pB using 2D contours.
    # Reference: https://nbviewer.jupyter.org/github/eecs445-f16/umich-eecs445-f16/blob/master/handsOn_lecture17_clustering-mixtures-em/handsOn_lecture17_clustering-mixtures-em.ipynb#Problem:-implement-EM-for-Coin-Flips
    # Args:
    # X: np.array of shape (n_trials,), dtype int,
    # the observations (number of heads at each trial).
    # n: int, total number of tosses per trial.
    # Q: np.array of shape (n_trials, 2), dtype float,
    # the hidden posterior q(z) (z = A, B) computed in the E-step.
    # path: (optional) A list of tuple of (pA, pB) that are visited in the EM iterations.
    # Visualized as line segments if not empty.
    # Returns: Shows the figure and returns None.
    grid_fn=lambda pA,pB: ELBO(X,n,Q,(0.5,pA,pB))
    return plot_coin_function(grid_fn=grid_fn,title="ELBO",path=path)

# Starts the EM algorithm.

n=10 # number of tosses per trial
X=[5,9,8,4,7] # observation
lam=0.5 # prior
p1=0.6 # parameter: pA
p2=0.5 # parameter: pB
n_trials=len(X) # number of trials
n_iters=10 # number of EM iterations
path=[(p1,p2)]
print('Init: theta = ')
print(p1,p2)
for i in range(n_iters):
    print(f'=== EM Iter: {i+1} ===')
    # E-step
    q=np.zeros([n_trials,2])
    for trial in range(n_trials):
        x=X[trial]
        q[trial,0]=lam*binomial(x,n,p1)
        q[trial,1]=(1-lam)*binomial(x,n,p2)
        q[trial,:]=q[trial,:]/np.sum(q[trial,:])
    print('E-step: q(z) = ')
    print(q)
    # M-step
    p1=sum((np.array(X)/n)*q[:,0])/sum(q[:,0])
    p2=sum((np.array(X)/n)*q[:,1])/sum(q[:,1])
    path.append([p1,p2])
    print('M-step: theta = ')
    print(p1,p2)

plot_coin_likelihood(X,n,path)
plot_coin_ELBO(X,n,q,path)

print("\n" + "="*60)
print("Result: Thus, the python program for EM for Bayesian networks was executed successfully.")
print("="*60)