# Author: Werley Cordeiro
# werleycordeiro@gmail.com
import numpy as np
import pandas as pd
import lyapunov
import Nelson_Siegel_factor_loadings
import Kfilter
from lyapunov import lyapunov
from Nelson_Siegel_factor_loadings import Nelson_Siegel_factor_loadings
from Kfilter import Kfilter
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
 Z = Nelson_Siegel_factor_loadings(lam=lam,matu=matu)
 for i in range(0,W):
 	H[i,i] = np.exp(para[i+1])
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
 P_t[:,:,0] = lyapunov(N=N,phi=phi,Q=Q)
 logLik =-0.5*T*W*np.log(2*np.pi)
 return(Kfilter(logLik=logLik,N=N,T=T,Y=Y,Z=Z,a_t=a_t,P_t=P_t,H=H,a_tt=a_tt,P_tt=P_tt,v2=v2,v1=v1,phi=phi,mu=mu,Q=Q,prev=prev,M=M,Yf=Yf,lik=lik))
