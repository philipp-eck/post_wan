#! /bin/python

import numpy as np
from observables import observables
from hamiltonian import hamiltonian
from k_space     import k_space



#### Setting up Hamiltonian

inputfile = "../../test_ham/TaAs.dat"
bra_vec   = np.array([[-1.7300404064779764, 1.7300378524931532,-5.8652233679604544],[1.7300406436001201,-1.7300378524931532,-5.8652234410777879],[-1.7300406436001201,-1.7300378524931532, 5.8652234410777879]])
spin      = True
basis     = np.array([2,2,1,1]) # {Ta1:d,Ta2:d,As1:p,As2:p}
ef        = 7.05179641

Ham = hamiltonian(inputfile,bra_vec,spin,basis,ef)



#### Setting up k-space

ktype     = "sphere"
#ktype     = "sphere_ster_proj"
kbasis    = "car"
vecs      = np.array([[ 0.50930216,   -0.03717546,   -0.31628638]]) # In this calculation: center of the sphere
radius    = 0.030
npoints   = 60

K_space = k_space(ktype,kbasis,vecs,bra_vec,npoints,radius)


#### Defining operators

op_types  =["S","L","J"]
op_types_k=["BC"]



#### Running calculation

# Initializing observables
Observables = observables(Ham,K_space,op_types,op_types_k)

# Calculating observables
Observables.calculate_ops()

# Calculating Pontryagin-index for all observables
#Observables.sphere_winding()

