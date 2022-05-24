# Files in the folder

* `Aw2Gl.ipynb`: constructs kernel matrix to convert A(omega) ("`Aw`")
to G_l ("`Gl`")
* `Aw2Gl-precise.py`: same as above but using a multi-precision library.
In our tests that did not seem to influence results much.
* `o2l_b40.npy`: the pre-computed kernel to convert Aw ("`o`") to Gl ("`l`")
at an inverse temperature beta = 40,
a frequency range of [-8, 8], on a grid of N = 800 points
* `u_mp.npy`, `s_mp.npy`, `vh_mp.npy`: singular value decomposition of the
multi-precision version of the above kernel matrix

# Generate different kernel matrices

* For other inverse temperatures/size of Legendre basis, in `Aw2Gl.ipynb`
(convert A(omega) to G_l) cell [2], change the beta/lmax variable
* For other omega grid, in `Aw2Gl.ipynb`, cell [1], change the omega assignment
(uniform grid only)
* To generate the exact kernel matrix for beta = 40, l = 80, use `Aw2Gl-precise.py`
