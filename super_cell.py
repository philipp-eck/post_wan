#! /bin/python
### super-cell class, builds-up a super-cell from a primitive Hamiltonian 

import numpy as np
from hamiltonian import hamiltonian
from copy import deepcopy
import time
class super_cell(hamiltonian):
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
        sup_basis   # Basis position of the primitive unit cell in the super cell
    '''

    def __init__(self,BULK_HAM,SUP_VEC,PBC=None):
        '''Initializes the super-cell Hamiltonian, by starting from a bulk model'''
        self.bulk_ham = deepcopy(BULK_HAM)
        self.bulk_ham.R = self.bulk_ham.R.astype(float)
        self.sup_vec  = SUP_VEC#.astype(int)
        self.sup_vec_inv = np.linalg.inv(self.sup_vec)
        self.pbc      = PBC

        self.spin     = self.bulk_ham.spin
        self.ef       = self.bulk_ham.ef
       # print(int(self.uc_vol(self.bra_vec)/self.uc_vol(self.bulk_ham.bra_vec)))
        self.sup_dim = int(abs(self.triple_product(self.sup_vec)))
        self.n_orb    = self.sup_dim*self.bulk_ham.n_orb
        self.n_bands  = self.sup_dim*self.bulk_ham.n_bands
        self.n_elec   = self.sup_dim*self.bulk_ham.n_elec
        self.ctype    = BULK_HAM.hr.dtype
        
        if self.bulk_ham.basis is not None:
            self.basis    = np.kron(np.ones(self.sup_dim),self.bulk_ham.basis).astype(int)
        else:
            self.basis = None
        self.hr_spinless = None
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
           R_frac    : R-vectors in the basis of the super-cell lattice vectors
           R_prim    : Position of the unit cell in the super-cell, i.e. the basis
        '''
        print("Constructing super-cell H(R)...")
        self.R_frac = []
        for R in self.bulk_ham.R:
            self.R_frac.append(np.einsum("ji,j",self.sup_vec_inv,R[0:3]))
        self.R_frac = np.array(self.R_frac)
        eps = 0.001
        self.R_prim = np.abs(np.around(np.einsum("ji,...j->...i",self.sup_vec,np.mod(self.R_frac+eps,1)-eps))).astype(int)
        #set up bulk Hamiltonian
        self.hr = np.zeros((np.shape(self.bulk_ham.R)[0],self.sup_dim*self.bulk_ham.n_bands,self.sup_dim*self.bulk_ham.n_bands),dtype=self.ctype)
        #create lists containing all "basis" positions in the new super-cell, do this symmetrically to the origin
        basis_p = []
        basis_m = []
        t0 = time.time()
        for R_0 in range(0,np.amax(np.abs(self.sup_vec))+1):
            for R_1 in range(0,np.amax(np.abs(self.sup_vec))+1):
                for R_2 in range(0,np.amax(np.abs(self.sup_vec))+1):
                   #R = np.around(np.einsum("ji,j",self.sup_vec,np.mod(np.einsum("ji,j",self.sup_vec_inv,np.array([R_0,R_1,R_2])),1)))#.astype(int)
                    R = self.calc_R_basis(R_0,R_1,R_2)
                    if R.tolist() not in basis_p and R.tolist() not in basis_m:
                        basis_p.append(R.tolist())
        print("Time for setting-up basis positions:", time.time()-t0)
        basis = np.append(np.array(basis_p),np.array(basis_m)[::-1]).reshape((self.sup_dim,3)).astype(int)
        basis_red = np.einsum("ji,bj->bi",self.sup_vec_inv,basis)
        self.R = np.zeros_like(self.bulk_ham.R)

        def add_hr_old(self,r,i,j):
            '''Not needed any more, can be deleted...
               Function sets the Hamiltonian elements.
            '''
            o = self.bulk_ham.n_orb
            O = o*self.sup_dim
            if self.spin == False:
                self.hr[r,i*o:(i+1)*o,j*o:(j+1)*o]+=self.bulk_ham.hr[r]
            else:
                for s1 in range(2):
                    for s2 in range(2):
                        self.hr[r,s1*O+i*o:s1*O+(i+1)*o,s2*O+j*o:s2*O+(j+1)*o]+=self.bulk_ham.hr[r,s1*o:s1*o+o,s2*o:s2*o+o]

        def add_hr(self,r,d12):
            '''Function sets the Hamiltonian elements.
            '''
            o = self.bulk_ham.n_orb
            O = o*self.sup_dim
            if self.spin == False:
                self.hr[r]+=np.kron(d12,self.bulk_ham.hr[r])
            else:
                for s1 in range(2):
                    for s2 in range(2):
                        self.hr[r,s1*O:(s1+1)*O,s2*O:(s2+1)*O]+=np.kron(d12,self.bulk_ham.hr[r,s1*o:(s1+1)*o,s2*o:(s2+1)*o])
        
        self.sup_basis = basis

        o = self.bulk_ham.n_orb
        O = o*self.sup_dim
#       hr_new = np.zeros_like(self.hr)
        t0 = time.time()
        r12 = np.zeros((self.sup_dim,self.sup_dim,3))
        r12 += basis_red[None] - basis_red[:,None]
        eps = 0.00001
        if self.pbc == None:
             def set_d12(R,r12):
                 eps = 0.001
                 return np.all(np.mod(abs(r12-R)+eps,1)<2*eps,axis=2).astype(int)
        else:
             def set_d12(R,r12):
                 eps = 0.001
                 return np.all(np.mod(abs(r12-R)+eps,1)<2*eps,axis=2).astype(int) * (abs(r12-R)[:,:,self.pbc]<eps).astype(int) 
        for r,R in enumerate(self.R_frac):
            self.R[r,3] = self.bulk_ham.R[r,3]
            self.R[r,:3]= self.R_frac[r]
            d12 = set_d12(R,r12)
            add_hr(self,r,d12)
        print("Time for seting-up super-cell H(R)", time.time()-t0)
        #probably not needed anymore
        self.hk_path     = np.einsum_path("ikl,ai->akl",self.hr, np.exp(1j*2*np.pi*np.einsum("ib,ab",self.R[:,:3],np.array([[0,0,0]])))/self.R[:,3], optimize='optimal')[0]


    def calc_R_basis(self,R0,R1,R2):
        eps = 0.001
        R = np.around(np.einsum("ji,j",self.sup_vec,np.mod(np.einsum("ji,j",self.sup_vec_inv,np.array([R0,R1,R2]))+eps,1)-eps))#.astype(int)
        return R




### Testing section
if __name__ == "__main__":
    print('Testing class "super_cell":')
    print("Creating bulk Hamiltonian for a 2D Hamiltonian ...")
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

    print("Testing H(R) spinless FT...")
    Ham_super_cell.hk_spinless(k)

    print("Initializing slab...")
    Ham_slab = super_cell(Ham_bulk,sup_vec,1)
    Ham_slab.hk(k)

    print('Testing class "super_cell":')
    bra_vec   = np.array([[ 3.9050000,    0.0000000,    0.0000000],
                          [ 0.0000000,    3.9050000,    0.0000000],
                          [ 0.0000000,    0.0000000,    3.9050000]])
    spin      = False
    basis     = np.array([2])
    n_elec    = 6
    inputfile = "test_ham/SRO_Ru_d_wo_soc.dat"
    Ham_bulk = hamiltonian(inputfile,bra_vec,spin,Basis,N_ELEC=n_elec)

    print("Initializing super_cell Hamiltonian")
    sup_vec = np.array([[ 1, 0, 0],
                        [ 1, 1, 0],
                        [ 0, 0, 5]])
    Ham_super_cell = super_cell(Ham_bulk,sup_vec)

    print("Testing H(R) FT...")
    k = np.array([[0,0,0],[0.0,0.7,0]])
    Ham_super_cell.hk(k)


    print("Initializing slab...")
    Ham_slab = super_cell(Ham_bulk,sup_vec,2)
    Ham_slab.hk(k)

