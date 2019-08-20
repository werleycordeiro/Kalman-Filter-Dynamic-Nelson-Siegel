# Author: Werley Cordeiro
# werleycordeiro@gmail.com
import numpy as np
import pandas as pd
from scipy import optimize
import DNS_baseline
from DNS_baseline import kalman
url = "https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1"
df = pd.read_csv(url,sep=';',index_col=0);
para = np.array([0.0609,
0.14170940,0.07289485,0.11492339,0.11120008,0.09055795,0.07672075,0.07222108,0.07076431,0.07012891,0.07267366,0.10624206,0.09029621,0.10374527,0.09801215,0.09122014,0.11794190,0.13354418,
0.99010443,0.02496842,-0.002294319,
-0.02812401,0.94256154, 0.028699387,
0.05178493,0.01247332, 0.788078795,
8.345444,-1.572442,0.2029919,  
0.3408764,
-0.07882772,0.62661018,
-0.21351036,-0.00425989,1.08802059])
para[0:18] = np.log(para[0:18])
prev = False;
ahead = 12;
lik = True;
matu =  np.array([3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]);
bnds = ((-np.Inf,np.Inf),
	(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf),
	(0.10,1.05),(-0.15,1.0),(-0.15,1.0),
	(-0.15,1.00),(0.10,1.0),(-0.15,1.0),
	(-0.15,1.00),(-0.15,1.0),(0.10,1.0),
	(-14,14),(-14,14),(-14,14),
	(-np.Inf,np.Inf),
	(-np.Inf,np.Inf),(-np.Inf,np.Inf),
	(-np.Inf,np.Inf),(-np.Inf,np.Inf),(-np.Inf,np.Inf))
optimize.minimize(fun=kalman,
	x0=para,
	args=(df,lik,prev,ahead,matu),
	method='L-BFGS-B',
	bounds=bnds,
	options={'disp':True})