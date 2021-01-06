#! /bin/python
### super-cell class, builds-up a super-cell from a primitive Hamiltonian 

import numpy as np
from hamiltonian import hamiltonian
from copy import deepcopy
class super_cell:
    ''' super-cellclass, builds up a super cell from a given "primitive" Wannier Hamiltonian.
        Instance attributes:
        bulk_ham    # Python hamiltonian class, bulk Hamiltonian input
        sup_vec     # Vectors spanning the unit cells (given in terms of primitive unit cell vectors)
        spin        # Boolean, if True, the Hamiltonian is spin-full
        n_orb       # Total number of orbitals
        n_bands     # Total number of bands
        hr          # Array containing the H(R)
        hr_spinless # Spin-less Hamiltonian, created by function hr_spinless
        R           # Array containing the R-vectors
        R_cart      # R-vectors in cartesian coordinates
        bra_vec     # Array containing the Bravais-vectors
        basis       # Array containing the l-quantum numbers of the subspaces
        ef          # Float containing the Fermi-energy
        n_elec      # Number of electrons in the system
    '''

    def __init__(self,BULK_HAM,SUP_VEC,PBC=None):
        '''Initializes the super-cell Hamiltonian, by starting from a bulk model'''
        self.bulk_ham = deepcopy(BULK_HAM)
        self.bulk_ham.R = self.bulk_ham.R.astype(float)
        self.sup_vec  = SUP_VEC#.astype(int)
        self.pbc      = PBC

        self.spin     = self.bulk_ham.spin
        self.ef       = self.bulk_ham.ef
       # print(int(self.uc_vol(self.bra_vec)/self.uc_vol(self.bulk_ham.bra_vec)))
        self.sup_dim = int(abs(self.triple_product(self.sup_vec)))
        self.n_orb    = self.sup_dim*self.bulk_ham.n_orb
        self.n_bands  = self.sup_dim*self.bulk_ham.n_bands
        self.n_elec   = self.sup_dim*self.bulk_ham.n_elec
        self.basis    = np.kron(np.ones(self.sup_dim),self.bulk_ham.basis).astype(int)
        self.set_super_cell()
        if type(self.bulk_ham.bra_vec) != np.ndarray:
            print("Define Bravais vectors of the bulk model!!!")
        else:
            self.bra_vec = np.einsum("ij,jk",self.sup_vec,self.bulk_ham.bra_vec)
            self.R_cart = np.einsum("...ij,...i",self.bra_vec[None],self.R[:,0:3])


        print("Super-cell contains "+str(self.sup_dim)+" primitive unit cells.")
        if self.spin == True:
            print("H(R) is spin-full and contains "+str(self.R.shape[0])+" R-vectors and "+str(self.n_orb)+" orbitals.")
        else:
            print("H(R) is spin-less and contains "+str(self.R.shape[0])+" R-vectors and "+str(self.n_orb)+" orbitals.")
        if not self.pbc == None:
            print("PBCs are broken along "+str(self.pbc+1)+". super-cell Bravais vector: ",self.bra_vec[self.pbc])


    def triple_product(self,vec):
        return np.dot(np.cross(vec[0],vec[1]),vec[2])

    def set_super_cell(self):
        '''Builds up the slab-Hamiltonian:
           1. Generate bulk Hamiltonian in the super-cell geometry.
           2. Build-up slab.
        '''
        print("Constructing super-cell H(R)...")
        self.R_frac = []
        for R in self.bulk_ham.R:
            self.R_frac.append(np.einsum("ji,j",np.linalg.inv(self.sup_vec),R[0:3]))
        self.R_frac = np.array(self.R_frac)
        self.R_prim = np.abs(np.around(np.einsum("ji,...j->...i",self.sup_vec,np.mod(self.R_frac,1)))).astype(int)

        #set up bulk Hamiltonian
        self.hr = np.zeros((np.shape(self.bulk_ham.R)[0],self.sup_dim*self.bulk_ham.n_bands,self.sup_dim*self.bulk_ham.n_bands),dtype=complex)
        #create lists containing all "basis" positions in the new super-cell, do this symmetrically to the origin
        basis_p = []
        basis_m = [] 
        for R_0 in range(-np.amax(self.sup_vec)*0,np.amax(self.sup_vec)):
            for R_1 in range(-np.amax(self.sup_vec)*0,np.amax(self.sup_vec)):
                for R_2 in range(-np.amax(self.sup_vec)*0,np.amax(self.sup_vec)):
                    R = np.around(np.einsum("ji,j",self.sup_vec,np.mod(np.einsum("ji,j",np.linalg.inv(self.sup_vec),np.array([R_0,R_1,R_2])),1)))#.astype(int)
                    if R.tolist() not in basis_p and R.tolist() not in basis_m:
                        basis_p.append(R.tolist())
                    R = np.around(np.einsum("ji,j",self.sup_vec,np.mod(np.einsum("ji,j",np.linalg.inv(self.sup_vec),np.array([-R_0,-R_1,-R_2])),1)))#.astype(int)
                    if R.tolist() not in basis_p and R.tolist() not in basis_m:
                        basis_m.append(R.tolist())
        basis = np.append(np.array(basis_p),np.array(basis_m)[::-1]).reshape((self.sup_dim,3)).astype(int)
        self.R = np.zeros_like(self.bulk_ham.R)
        o = self.bulk_ham.n_orb
        O = o*self.sup_dim
        for i,R in enumerate(self.R_frac):
            self.R[i,3] = self.bulk_ham.R[i,3]
            self.R[i,:3]= self.R_frac[i]
            # get basis site in super-cell
            for j,vec in enumerate(basis):
                if np.allclose(vec,self.R_prim[i]):
                    #use numpy roll to generate shift, do this separately on the spin-sectors
                    if self.pbc == None:
                       diag = np.eye(self.sup_dim)
                    else:
                       if abs(self.R[i,self.pbc]) >=1:
                          diag = np.diag(np.zeros(self.sup_dim))
                       elif self.R[i,self.pbc] >=0:
                          diag = np.diag(np.append(np.ones(self.sup_dim-j),np.zeros(j)))
                       elif self.R[i,self.pbc] <0:
                          diag = np.diag(np.append(np.zeros(self.sup_dim-j),np.ones(j)))
                    if self.spin == False:
                       self.hr[i]= np.roll(np.kron(np.eye(self.sup_dim),self.bulk_ham.hr[i]),+o*j,axis=0)
                    else:
                       self.hr[i,0:  O,0:  O] = np.roll(np.kron(diag,self.bulk_ham.hr[i][0:  o,0:  o]),+o*j,axis=0)
                       self.hr[i,O:2*O,O:2*O] = np.roll(np.kron(diag,self.bulk_ham.hr[i][o:2*o,o:2*o]),+o*j,axis=0)
                       self.hr[i,O:2*O,0:  O] = np.roll(np.kron(diag,self.bulk_ham.hr[i][o:2*o,0:  o]),+o*j,axis=0)
                       self.hr[i,0:  O,O:2*O] = np.roll(np.kron(diag,self.bulk_ham.hr[i][0:  o,o:2*o]),+o*j,axis=0)


        self.hk_path     = np.einsum_path("ikl,ai->akl",self.hr, np.exp(1j*2*np.pi*np.einsum("ib,ab",self.R[:,:3],np.array([[0,0,0]])))/self.R[:,3], optimize='optimal')[0]

    def hk(self,k_red):
        '''Performs Fouriert-transform at given point in reduced coordinates.'''
        hk = hamiltonian.hk(self,k_red)
#       print("Hamiltonian is hermitian?:",np.allclose(hk,np.conjugate(hk.transpose(0,2,1))))
        return hk


    def del_hk(self,k_red):
        '''Returns nabla_k H_k in cartesian coordinates.'''
        del_hk = hamiltonian.del_hk(self,k_red)
        return del_hk


    def set_hr_spinless(self):
        '''Averages spin-blocks to obtain a spin-less Hamiltonian without SOC interaction.'''
        self.hr_spinless = (self.hr[:,:self.n_orb,:self.n_orb]+self.hr[:,self.n_orb:,self.n_orb:])/2.0



### Testing section
if __name__ == "__main__":
    print('Testing class "super_cell":')
    print("Creating bulk Hamiltonian...")
    bra_vec   = np.array([[ 3.0730000,    0.0000000,    0.0000000],
                          [-1.5365000,    2.6612960,    0.0000000],
                          [ 0.0000000,    0.0000000,   20.0000000]])
    spin      = True
    n_elec    = 4
    inputfile = "test_ham/In_SiC_soc.dat"
    Basis     = np.array([0,1]) 
    Ham_bulk = hamiltonian(inputfile,bra_vec,spin,Basis,N_ELEC=n_elec)

    print("Initializing super_cell Hamiltonian")
    sup_vec = np.array([[ 1, 0, 0],
                        [10,20, 0],
                        [ 0, 0, 1]])
    Ham_super_cell = super_cell(Ham_bulk,sup_vec)

    print("Testing H(R) FT...")
    k = np.array([[0,0,0],[0.0,0.7,0]])
    Ham_super_cell.hk(k)

    print("Initializing slab...")
    Ham_slab = super_cell(Ham_bulk,sup_vec,1)
    Ham_slab.hk(k)

