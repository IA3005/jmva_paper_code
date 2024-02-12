import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

import numpy as np

from scipy.stats import wishart,beta

import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient


from manifold import SPD

import numpy.linalg as la



### simulate t-Wishart samples

def t_wishart_rvs(n,scale,df,size):
    p,_=scale.shape
    L = la.cholesky(scale)
    ws = wishart.rvs(scale=np.eye(p),df=n,size=size)
    qs = beta.rvs(a=df/2,b=n*p/2,size=size)
    vec = df*(1/qs-1)/np.trace(ws,axis1=-1,axis2=-2)
    return np.einsum('...,...ij->...ij',vec,L@ws@L.T) 



### cost and grad for t- Wishart 

def t_wish_cost(R,S,n,df):
    k, p, _ = S.shape
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    return 1/2 * np.log(la.det(R)) - np.sum(-(df+n*p)/2*np.log(1+a/df))/n/k


def t_wish_egrad(R,S,n,df):
    k, p, _ = S.shape
    # psi
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    psi = np.einsum('k,kij->ij',(df+n*p)/(df+a),S)
    return la.solve(R,la.solve(R.T,((R  - psi/n/k) /2).T).T)


def t_wish_est(S,n,df):
    
    p = S.shape[1]
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    
    @pymanopt.function.numpy(manifold)
    def cost(R):
        return t_wish_cost(R,S,n,df)
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(R):
        return t_wish_egrad(R,S,n,df)
    #
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    init = np.eye(S.shape[-1])
    optimizer = ConjugateGradient(verbosity=0)
    return optimizer.run(problem, initial_point=init).point

