from mpmath import *
import numpy as np

mp.dps = 64
beta = mpf(40)

lmax, Nomega = int(80), int(800)
omegac = 8
domega = mpf(16/799)
K = matrix(lmax, Nomega)
I = matrix(lmax, Nomega)
err = matrix(lmax, Nomega)
serr = matrix(lmax, Nomega)

for l in range(lmax):
    print(l)
    for N in range(Nomega):
        I[l, N], err[l, N] = quad(lambda tau: legendre(
            l, 2*tau/beta - 1) * exp(-(-omegac + domega*N)*tau), [0, beta], error=True)
        if N != 0 and N != Nomega-1:
            K[l, N] = domega * (-sqrt(2*l + 1)) * I[l, N] / \
                (1 + exp(-beta * (-omegac + domega*N)))
            serr[l, N] = domega * (-sqrt(2*l + 1)) * err[l, N] / \
                (1 + exp(-beta * (-omegac + domega*N)))
 
        else:
            K[l, N] = domega/2 * (-sqrt(2*l + 1)) * I[l, N] / \
                (1 + exp(-beta * (-omegac + domega*N)))
            
            serr[l, N] = domega/2 * (-sqrt(2*l + 1)) * err[l, N] / \
                (1 + exp(-beta * (-omegac + domega*N)))
u, s, vh = svd_r(K, compute_uv=True)

K_arr = np.array(K.tolist(), dtype=np.float64)
I_arr = np.array(I.tolist(), dtype=np.float64)
err_arr = np.array(err.tolist(), dtype=np.float64)
serr_arr = np.array(serr.tolist(), dtype=np.float64)

u_arr = np.array(u.tolist(), dtype=np.float64)
s_arr = np.array(s.tolist(), dtype=np.float64)
vh_arr = np.array(vh.tolist(), dtype=np.float64)



np.save("K.npy", K_arr)
np.save("I.npy", I_arr)
np.save("err.npy", err_arr)
np.save("serr.npy", serr_arr)
np.save("u_mp.npy", u_arr)
np.save("s_mp.npy", s_arr)
np.save("vh_mp.npy", vh_arr)



