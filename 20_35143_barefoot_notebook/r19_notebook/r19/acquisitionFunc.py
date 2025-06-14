# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:51:02 2020

@author: richardcouperthwaite
"""

import numpy as np
from scipy.stats import norm
from scipy import random

def knowledge_gradient(M, sn, mu, sigma):
    """
    This is the method used to determine the knowledge gradient of the fused model
    for a given set of test data points. The aim is to calculate the best possible
    point to be used in the next iteration. This function will be called after the
    fused model is calculated for each of the lower order models being assumed to
    be the truth model.
    
    Implementation based on the work by Frazier, Powell, Dayanik
    [1]P. Frazier, W. Powell, and S. Dayanik, “The Knowledge-Gradient Policy for 
    Correlated Normal Beliefs,” INFORMS Journal on Computing, vol. 21, no. 4, pp. 
    599–613, May 2009.
    
    M: the number of samples
    sn: the noise of the model
    mu: mean of the model for all M samples
    sigma: covariance matrix of the model
    
    The function returns:
    NU: Knowledge Gradient values for all the samples
    nu_star: the maximum knowledge gradient value
    x_star: the index of the value with the maximum knowledge gradient (0 as first index)
    """
    from scipy.stats import norm
    
    def algorithm1(a, b, M):
        c = [np.inf]
        A = [0]
        for i in range(M-1):
            c.append(np.inf)
            t = 0
            while t == 0:
                j = A[-1]
                c[j] = (a[j]-a[i+1])/(b[i+1]-b[j])
                if (len(A)!=1) and (c[j]<=c[A[-2]]):
                    A = A[0:-1] 
                else:
                    t = 1
            A.append(i+1)
        c = np.array(c)
        A = np.array(A)
        return c, A
    
    NU = []
    
    for i in range(M):   
        a = mu
        try:
            b = sigma[:,i]/np.sqrt(sn**2+sigma[i,i])
        except IndexError:
            b = sigma/np.sqrt(sn**2+sigma[i])
            
        I = np.argsort(b)
        a = a[I]
        b = b[I]
        bb, indexes, inverse = np.unique(b, return_index=True, return_inverse=True)
        aa = []
        for ii in range(len(indexes)):
            aa.append(np.max(a[np.where(b == b[indexes[ii]])]))
            
        MM = len(aa)
        aa = np.array(aa)
        c, A = algorithm1(aa, bb, MM)
        aa = aa[A]
        bb = bb[A]
        c = c[A]
        MM = A.shape[0]
        sig = 0
        for ii in range(MM-1):
            sig += (bb[ii+1]-bb[ii])*(norm.pdf(-abs(c[ii]))+ (-abs(c[ii])) * norm.cdf(-abs(c[ii])))
        nu = np.log(sig)
        NU.append(nu)
        
        try:
            if nu>nu_star:
                nu_star = nu
                x_star = i
        except NameError:
            nu_star = nu
            x_star = i
    
    return nu_star, x_star, NU

def expected_improvement(curr_max, xi, y, std):
    """
    This function calculates the maximum expected improvement for a selection of
    test points from the surrogate model of an objective function with a mean and variance.
    
    J. Mockus, V. Tiesis, and A. Zilinskas. Toward Global Optimization, volume 2,
    chapter The Application of Bayesian Methods for Seeking the Extremum, pages
    117{128. Elsevier, 1978.
    
    Parameters
    ----------
    curr_max : float
        This value is the best value of the objective function that hase been obtained.
    xi : float
        This parameter defines how much the algorithm exploits, or explores.
    y : 1D vector Numpy Array
        The mean of the surrogate model at all test points used in the optimization.
    std : 1D vector Numpy Array
        The standard deviation from the surrogate model at all test points 
        used in the optimization.

    Returns
    -------
    max_val : float
        The maximum expected improvement value.
    x_star : integer
        The index of the test point with the maximum expected improvement value.
    EI : TYPE
        Expected improvement values for all test points.

    """
    
    pdf = norm.pdf(y)
    cdf = norm.cdf(y)

    EI = (y-curr_max-xi)*pdf + std*cdf
        
    max_val = np.max(EI)
    x_star = np.where(EI == max_val)[0]
    
    return max_val, x_star[0], EI

def probability_improvement(curr_max, xi, y, std):
    """
    This function calculates the maximum probability improvement for a selection of
    test points from the surrogate model of an objective function with a mean and variance.
    
    Kushner, H. J. “A New Method of Locating the Maximum Point of an Arbitrary 
    Multipeak Curve in the Presence of Noise.” Journal of Basic Engineering 86, 
    no. 1 (March 1, 1964): 97–106. https://doi.org/10.1115/1.3653121.
    
    Parameters
    ----------
    curr_max : float
        This value is the best value of the objective function that hase been obtained.
    xi : float
        This parameter defines how much the algorithm exploits, or explores.
    y : 1D vector Numpy Array
        The mean of the surrogate model at all test points used in the optimization.
    std : 1D vector Numpy Array
        The standard deviation from the surrogate model at all test points 
        used in the optimization.

    Returns
    -------
    max_val : float
        The maximum probability of improvement value.
    x_star : integer
        The index of the test point with the maximum probability of improvement value.
    PI : TYPE
        Probability of improvement values for all test points.

    """
    
    PI = norm.cdf((y-curr_max-xi)/std)
    max_val = np.max(PI)
    x_star = np.where(PI == max_val)[0]
    
    return max_val, x_star[0], PI

def upper_conf_bound(kt, y, std):
    """
    This function calculates the Upper Confidence Bound for a selection of
    test points from the surrogate model of an objective function with a mean and variance.
    
    D. D. Cox and S. John. SDO: A statistical method for global optimization. In
    M. N. Alexandrov and M. Y. Hussaini, editors, Multidisciplinary Design Opti-
    mization: State of the Art, pages 315{329. SIAM, 1997.
        
    Parameters
    ----------
    curr_max : float
        This value is the best value of the objective function that hase been obtained.
    kt : float
        This parameter is a combined parameter for the sqrt(beta*nu).
    y : 1D vector Numpy Array
        The mean of the surrogate model at all test points used in the optimization.
    std : 1D vector Numpy Array
        The standard deviation from the surrogate model at all test points 
        used in the optimization.

    Returns
    -------
    max_val : float
        The maximum upper confidence bound value.
    x_star : integer
        The index of the test point with the maximum upper confidence bound value.
    UCB : TYPE
        Upper confidence bound values for all test points.

    """
    
    UCB = y+kt*std
    max_val = np.max(UCB)
    x_star = np.where(UCB == max_val)[0]
    
    return max_val, x_star[0], UCB

def thompson_sampling(y, std):
    """
    Thompson sampling was first described by Thompson in 1933 as a solution to
    the multi-arm bandit problem.
    
    Thompson, W. 1933. “On the likelihood that one unknown probability
    exceeds another in view of the evidence of two samples”. Biometrika.
    25(3/4): 285–294.

    Parameters
    ----------
    y : 1D vector Numpy Array
        The mean of the surrogate model at all test points used in the optimization.
    std : 1D vector Numpy Array
        The standard deviation from the surrogate model at all test points 
        used in the optimization.

    Returns
    -------
    nu_star : float
        The maximum value from the Thompson Sampling.
    x_star : integer
        The index of the test point with the maximum value.
    tsVal : TYPE
        Sampled values for all test points.
    """
    
    tsVal = random.normal(loc=y, scale=std)
    nu_star = np.max(tsVal)
    
    x_star = int(np.where(tsVal == nu_star)[0])
    
    return nu_star, x_star, tsVal