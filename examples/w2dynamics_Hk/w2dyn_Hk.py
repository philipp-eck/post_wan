#! /bin/python

import numpy as np
from hamiltonian import hamiltonian
from k_space     import k_space



#### Setting up Hamiltonian

inputfile = "../../test_ham/hr_In_soc.dat"
spin      = False

Ham = hamiltonian(inputfile,SPIN=spin)

#### Fourier-transformation
grid = np.array([[2,3,4]]) # FT on a 2x3x4 grid
w2name = "Hk_w2"
Ham.w2dynamics_hk(grid,w2name)



