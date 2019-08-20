import numpy as np
def lyapunov(N,phi,Q):
 return(np.matmul(np.linalg.inv(np.identity(N**2) - np.kron(phi,phi)), np.reshape(Q,(N**2),1)).reshape((N,N)))
