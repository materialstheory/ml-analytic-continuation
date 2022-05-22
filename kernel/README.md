* Aw2Gl.ipynb: construct kernel matrix convert Aw to Gl
* Aw2Gl-precise.ipynb: construct precise version of the above with multi-precision library
* o2l_b40.npy: an example convert Aw (o) to Gl (l) at beta = 40, [-8, 8], N = 800
* u_mp.npy, s_mp.npy, vh_mp.npy: singular value decomposition of the EXACT version of the kernel matrix labeled by o2l_b40, computed with multi-precision library (mp).
* readme.md: how to generate other kernel matrices

# Generate kernel matrices

* For other inverse temperatures/number of Legendre basis, in Aw2Gl.ipynb (convert $A(\omega)$ to $G_{l}$) cell [2], change the beta/lmax variable
* For other omega grid, in Aw2Gl.ipynb, cell [1], change the omega assignment (uniform grid only)
* To generate the exact kernel matrix for beta = 40, l = 80, use Aw2Gl-precise.py
