import numpy as np
def Kfilter(logLik,N,T,Y,Z,a_t,P_t,H,a_tt,P_tt,v2,v1,phi,mu,Q,prev,M,Yf,lik):
 for t in range(0,T):
 	v = Y.iloc[t,:] - Z.dot(a_t[t,:])
 	F = Z.dot(P_t[:,:,t]).dot(Z.T) + H
 	if (np.linalg.det(F)<=1e-32) or (np.isneginf(np.linalg.det(F))) or (np.isinf(np.linalg.det(F))) or (np.isinf(np.linalg.det(F))) or np.isnan(np.log(np.linalg.det(F))):
 		logLik = -1000000000000;break
 	else:
 		F_inv = np.linalg.inv(F)
 		logLik = logLik - 0.5 * (np.log(np.linalg.det(F)) + v.T.dot(F_inv).dot(v))
 		a_tt[t,:] = a_t[t,:] + P_t[:,:,t].dot(Z.T).dot(F_inv).dot(v)
 		P_tt[:,:,t] = P_t[:,:,t] - P_t[:,:,t].dot(Z.T).dot(F_inv).dot(Z).dot(P_t[:,:,t])
 		v1[t,:] = Z.dot(a_tt[t,:])
 		v2[t,:] = Y.iloc[t,:] - Z.dot(a_tt[t,:])
 		a_t[t + 1,:] = phi.dot(a_tt[t,:]) + (np.identity(N) - phi).dot(mu)
 		P_t[:,:,t + 1] = phi.dot(P_tt[:,:,t]).dot(phi.T) + Q
 	if prev:
 		if t>(T-1):
 			for m in range(0,M):
 				Yf[t + m,:] = Z.dot(a_t[t + m,:])
 				a_tt[t + m,:] = a_t[t + m, ]
 				P_tt[:,:,t + m] = P_t[:,:,t + m]
 				a_t[t + m + 1,:] = phi.dot(a_tt[t + m,:]) + (np.identity(N) - phi).dot(mu)
 				P_t[:,:,t + m + 1] = phi.dot(P_tt[t + m,:,:]).dot(phi.T) + Q
 if lik:
 	return(-logLik)
 else:
 	return(a_tt,a_t,P_tt,P_t,v2,v1,Yf)
