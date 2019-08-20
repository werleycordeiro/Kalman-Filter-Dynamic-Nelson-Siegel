import numpy as np
def Nelson_Siegel_factor_loadings(lam,matu):
 lam = np.exp(lam)
 A = lam*matu
 c1 = np.ones(matu.shape[0])
 c2 = (1-np.exp(-A))/(A)
 c3 = c2-np.exp(-A)
 lambmat = np.vstack((c1,c2,c3)).T
 return(lambmat)