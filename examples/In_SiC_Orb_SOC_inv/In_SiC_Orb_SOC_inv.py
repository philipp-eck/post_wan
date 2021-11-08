#! /bin/python

import numpy as np
from observables import observables
from hamiltonian import hamiltonian
from k_space     import k_space



#### Setting up Hamiltonian

inputfile = "../../test_ham/In_SiC_soc.dat"
bra_vec   = np.array([
   [ 3.0730000,    0.0000000,    0.0000000], 
   [-1.5365000,    2.6612960,    0.0000000],
   [ 0.0000000,    0.0000000,   20.0000000]
   ])

spin      = True
basis     = np.array([0,1]) # {In:s, In:p}
n_elec    = 4

Ham = hamiltonian(inputfile,bra_vec,spin,basis,N_ELEC=n_elec)



#### Setting up k-space

ktype     = "path"
kbasis    = "red"
vecs      = np.array([
                      [ 2/3,-1/3,0],
                      [ 1/2,   0,0],
                      [   0,   0,0],
                      [-2/3, 1/3,0]
                      ])
npoints   = 100
K_space = k_space(ktype,kbasis,vecs,bra_vec,npoints)
#vecs = np.array([[0,0,0],[3,0,0],[0,3,0]])
#ktype="plane"
#kbasis="car"
K_space = k_space(ktype,kbasis,vecs,bra_vec,npoints)

#### Defining operators

op_types  =["S","L","J"]
op_types_k=["BC","BC_S","BC_mag","Orb_SOC_inv"]
#op_types_k = ["E_triang"]

#### Running calculation

# Initializing observables
Observables = observables(Ham,K_space,op_types,op_types_k)

# Calculating observables
Observables.calculate_ops() 


