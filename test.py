# Author: Werley Cordeiro
# werleycordeiro@gmail.com

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import os

url = "https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1"
df = pd.read_csv(url,sep=';',index_col=0);
df.head();
df.tail();

para = np.array([0.0609,
0.14170940,0.07289485,0.11492339,0.11120008,0.09055795,0.07672075,0.07222108,0.07076431,0.07012891,0.07267366,0.10624206,0.09029621,0.10374527,0.09801215,0.09122014,0.11794190,0.13354418,
0.99010443,0.02496842,-0.002294319,
-0.02812401,0.94256154, 0.028699387,
0.05178493,0.01247332, 0.788078795,
8.345444,-1.572442,0.2029919,  
0.3408764,
-0.07882772,0.62661018,
-0.21351036,-0.00425989,1.08802059])

prev = False;
ahead = 12;
lik = True;
matu =  np.array([3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]);

def kalman(para,Y,lik,prev,ahead,matu):
 lam = para[0];
 M = ahead;
 if prev:
 	T = Y.shape[0]
 	Yf = Y
 	Yf.iloc[(T-M):T,] = np.nan
 	Y = Y.iloc[1:(T-M),]
 	T = Y.shape[0]
 else:
 	T = Y.shape[0]
 	Yf = 1
 W = Y.shape[1];
 N = 3;
 mu = np.zeros(N)
 phi = np.identity(N)
 H = np.identity(W)
 Q = np.identity(N)
 #import Nelson_Siegel_factor_loadings as Nelson_Siegel_factor_loadings 
 Z = Nelson_Siegel_factor_loadings(lam=lam,matu=matu)
 for i in range(0,W):
 	H[i,i] = para[i+1]
 H = H.dot(H)
 phi[0,0] = para[18]
 phi[0,1] = para[19]
 phi[0,2] = para[20]
 phi[1,0] = para[21]
 phi[1,1] = para[22]
 phi[1,2] = para[23]
 phi[2,0] = para[24]
 phi[2,1] = para[25]
 phi[2,2] = para[26]
 mu[0] = para[27]
 mu[1] = para[28]
 mu[2] = para[29]
 Q[0,0] = para[30]
 Q[1,0] = para[31]
 Q[1,1] = para[32]
 Q[2,0] = para[33]
 Q[2,1] = para[34]
 Q[2,2] = para[35]
 Q = Q.dot(Q.T)
 v1 = np.zeros((T,W))
 v2 = np.zeros((T,W))
 if prev:
 	a_tt = np.zeros(((T+M),N))
 	a_t = np.zeros(((T+M+1),N))
 	P_tt = np.zeros((N,N,(T+M)))
 	P_t = np.zeros((N,N,(T+M+1)))
 else:
 	a_tt = np.zeros((T,N))
 	a_t = np.zeros(((T+1),N))
 	P_tt = np.zeros((N,N,T))
 	P_t = np.zeros((N,N,(T+1)))
 a_t[0,:] = mu
 #import lyapunov as lyapunov
 P_t[:,:,0] = lyapunov(N=N,phi=phi,Q=Q)
 logLik =-0.5*T*W*np.log(2*np.pi)
 #import Kfilter as Kfilter
 return(Kfilter(logLik=logLik,N=N,T=T,Y=Y,Z=Z,a_t=a_t,P_t=P_t,H=H,a_tt=a_tt,P_tt=P_tt,v2=v2,v1=v1,phi=phi,mu=mu,Q=Q,prev=prev,M=M,Yf=Yf,lik=lik))

results = kalman(para=para,Y=df,lik=lik,prev=prev,ahead=ahead,matu=matu)
results #-2887.3


bnds = ((0.00001,0.99999),
	(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),(0,0.9),
	(0.10,1.05),(-0.15,1.0),(-0.15,1.0),
	(-0.15,1.0),(0.10,1.0),(-0.15,1.0),
	(-0.15,1.0),(-0.15,1.0),(0.10,1.0),
	(-14,14),(-14,14),(-14,14),
	(0,1.25),
	(-0.99,1.25),(0,1.25),
	(-0.99,1.25),(-0.99,1.25),(0,1.25))

from scipy import optimize

optimize.minimize(fun=kalman,
	x0=para,
	args=(df,lik,prev,ahead,matu),
	method='L-BFGS-B',
	bounds=bnds,
	options={'disp':True})