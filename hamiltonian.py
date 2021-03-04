#! /bin/python
### Hamiltonian class, reads the WANNIER90 H(R)

import numpy as np
from k_space import k_space
import time

#Parallelization
parallel = False
if parallel == True:
   while True:
      try:
          from joblib import Parallel, delayed
      except ModuleNotFoundError:
          print("module 'joblib' is not installed")
          break
      try:
         import multiprocessing
         num_cores = multiprocessing.cpu_count()
         parallel = True
         break
      except ModuleNotFoundError:
         print("module 'multiprocessing' is not installed")
         break
 
class hamiltonian:
    ''' Hamiltonian class, reads the WANNIER90 H(R), the real space vectors and creates reciprocal vectors.
        Instance attributes:
        hr_file     # Filename of the input wannier90_hr.dat
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

        Attributes for computational optimization
        hk_path     # variable storing the best path for calculating H(k), calculated at the gamma point
        del_hk_path # variable storing the best path for contraction the del_hk derivative, calculated at the gamma point
    '''

    def __init__(self,HR_FILE,BRA_VEC=None,SPIN=False,BASIS=None,EF=None,N_ELEC=None):
        '''Initializes the Hamiltonian class.'''
        self.hr_file = HR_FILE
        self.spin    = SPIN
        if type(BASIS) == np.ndarray:
            self.basis   = np.array(BASIS,dtype="int32")
        else:
            self.basis = None
        self.ef      = EF
        self.n_elec  = N_ELEC

        self.read_HR(self.hr_file)

        self.set_lattice_vec(BRA_VEC)

        self.hk_path     = np.einsum_path("ikl,ai->akl",self.hr, np.exp(1j*2*np.pi*np.einsum("ib,ab",self.R[:,:3],np.array([[0,0,0]])))/self.R[:,3], optimize='optimal')[0]
        self.del_hk_path = np.einsum_path("ij,ikl,ai->ajkl",self.R[:,:3],self.hr, np.exp(1j*2*np.pi*np.einsum("ib,ab",self.R[:,:3],np.array([[0,0,0]])))/self.R[:,3], optimize='optimal')[0]

    def read_HR(self,HR_file):
        '''Reads the wannier90_hr.dat and the R vectors, sets n_bands and n_orb.
        '''
        print("Reading Hamiltonian...")
        hr_file = open(HR_file, "r")
        hr_file.readline()
        line = hr_file.readline()
        self.n_bands = int(line)                     # Number of bands
        if self.spin == True:
           self.n_orb = self.n_bands//2
        else:
           self.n_orb = self.n_bands
        line = hr_file.readline()
        nR = int(line)                     # Number of R-vectors
        sk = (nR-1) % 15
        skipline = int((nR-sk)/15 + 1)

        #### Read the weights in the header of the hamiltonian
        
        weight = np.array([])
        for i in range(skipline):
           line=hr_file.readline().split()
           weight=np.append(weight,line)
        weights = weight.astype(np.int)

        #### Read full H(R) information
        time0 = time.time()
        hr_full = np.loadtxt(HR_file,skiprows=3+skipline)
        print("Time for reading hr-file: ",time.time()-time0)
        hr_full = np.reshape(hr_full,(nR,self.n_bands,self.n_bands,7))
        hr_file.close()

        #### Build H(R)
        hr = np.array(hr_full[:,:,:,5] + 1j * hr_full[:,:,:,6])
        # is this transpose correct? Yes, since it reproduces the DFT spin-texture correctly!!! This might not be considererd in wannier tools!!! 
        self.hr = np.transpose(hr,axes=(0,2,1))

        #### Create R-vector array, last entry contains the weights
        R = np.zeros((nR,4))
        R[:,0:3] = hr_full[:,0,0,0:3]
        R[:,3]   = weights
        self.R   = R.astype(int)

        #### Print summary
        print("Hamiltonian from file "+HR_file+" successfully loaded.")
        if self.spin == True:
            print("H(R) is spin-full and contains "+str(nR)+" R-vectors and "+str(self.n_orb)+" orbitals.")
        else:
            print("H(R) is spin-less and contains "+str(nR)+" R-vectors and "+str(self.n_orb)+" orbitals.")


    def set_lattice_vec(self,bra_vec):
        '''Checks if the defined Bravais vectors are complete
           Doesn't check if vectors span R3... '''
        if np.shape(bra_vec) == (3,3):
            self.bra_vec = bra_vec
            self.R_cart = np.einsum("...ij,...i",self.bra_vec[None],self.R[:,0:3])
        else:
            print("Bravais vectors have the wrong shape or are not defined!!!")


    def hk(self,k_red):
        '''Performs Fouriert-transform at given point in reduced coordinates.'''
       #hk_out = np.sum(self.hr[:,:,:] * (np.exp(1j*2*np.pi*np.sum(self.R[:,:3]*k_red[:],axis=1))/self.R[:,3])[:,None,None], axis=0)
        if k_red.ndim == 1:
            hk_out = np.einsum("ikl,i",self.hr, np.exp(1j*2*np.pi*np.einsum("ib,b",self.R[:,:3],k_red))/self.R[:,3])
        elif k_red.ndim == 2:
            hk_out = np.einsum("ikl,ai->akl",self.hr, np.exp(1j*2*np.pi*np.einsum("ib,ab",self.R[:,:3],k_red))/self.R[:,3],optimize=self.hk_path)
        return hk_out

    def hk_parallel(self,k_red):
        '''Performs FT k-parallelized.
        '''
        hk_out = np.zeros((np.shape(k_red)[0],self.n_bands,self.n_bands),dtype="complex")

        def hk_k(i_k):
            hk_out[i_k] = np.einsum("ikl,i",self.hr, np.exp(1j*2*np.pi*np.einsum("ib,b",self.R[:,:3],k_red[i_k]))/self.R[:,3])
        
        Parallel(num_cores,prefer="threads",require='sharedmem')(delayed(hk_k)(i_k) for i_k in range(np.shape(k_red)[0]))
        return hk_out


    def del_hk(self,k_red):
        '''Returns nabla_k H_k in cartesian coordinates.'''
        #del_hk_out  = 1j* np.sum(self.R_cart[:,:,None,None]*self.hr[:,None,:,:] * (np.exp(1j*2*np.pi*np.sum(self.R[:,:3]*k_red,axis=1))/self.R[:,3])[:,None,None,None], axis=0)
        if k_red.ndim == 1:
            del_hk_out = 1j* np.einsum("ij,ikl,i->jkl",self.R_cart,self.hr, np.exp(1j*2*np.pi*np.einsum("ib,b",self.R[:,:3],k_red))/self.R[:,3],optimize=False)
        elif k_red.ndim == 2:
            del_hk_out = 1j* np.einsum("ij,ikl,ai->ajkl",self.R_cart,self.hr, np.exp(1j*2*np.pi*np.einsum("ib,ab",self.R[:,:3],k_red))/self.R[:,3],optimize=True)#self.del_hk_path)
        return del_hk_out


    def set_hr_spinless(self):
        '''Averages spin-blocks to obtain a spin-less Hamiltonian without SOC interaction.'''
        self.hr_spinless = (self.hr[:,:self.n_orb,:self.n_orb]+self.hr[:,self.n_orb:,self.n_orb:])/2.0

    def hk_spinless(self,k_red):
        '''Performs Fouriert-transform at given point in reduced coordinates.'''
        #hk_out = np.sum(self.hr_spinless[:,:,:] * (np.exp(1j*2*np.pi*np.sum(self.R[:,:3]*k_red[:],axis=1))/self.R[:,3])[:,None,None], axis=0)
        if type(self.hr_spinless) != np.ndarray:
           self.set_hr_spinless()
        if k_red.ndim == 1:
            hk_out = np.einsum("ikl,i",self.hr_spinless, np.exp(1j*2*np.pi*np.einsum("ib,b",self.R[:,:3],k_red))/self.R[:,3])
        elif k_red.ndim == 2:
            hk_out = np.einsum("ikl,ai->akl",self.hr_spinless, np.exp(1j*2*np.pi*np.einsum("ib,ab",self.R[:,:3],k_red))/self.R[:,3],optimize=self.hk_path)
        return hk_out

    def w2dynamics_hk(self,grid,filename):
        '''Performs Fourier transformation on a Gamma-centered Monkhorst grid
           and writes output in w2-readable format.'''

        def write_Hk(kpoint,hk):
            output.write('{k[0]:9.4f}{k[1]:9.4f}{k[2]:9.4f} \n'.format(k=kpoint))
            for i in range(self.n_bands):
                for j in range(self.n_bands):
                    output.write('{:11.4f}{:9.4f}'.format(hk[j,i].real,hk[j,i].imag)) # j,i since j:line, i: column
                output.write("\n")
        mh_grid = k_space("monkhorst","red",grid,)

        output = open(filename+".dat","w")
        #write header
        output.write("{:9d}{:9d}{:9d}\n".format(np.shape(mh_grid.k_space_red)[0],self.n_bands,self.n_bands))
        for i_k in range(np.shape(mh_grid.k_space_red)[0]):
            hk = self.hk(mh_grid.k_space_red[i_k])
            write_Hk(mh_grid.k_space_red[i_k],hk)

        output.close()

    def get_H_R(self,R,part="complex",f="6.3f"):
        '''Returns H(R) for given R.
           part = {"complex","real","imag"}
        '''
        R_index = np.argwhere(np.all((self.R[:,:3]-R)==0, axis=1))
        H_R = self.hr[R_index][0,0]

        if part == "complex":
            def print_H_R(i,j):
                fs = "{:"+f+"}{:"+f+"} "
                print(fs.format(H_R[i,j].real,H_R[i,j].imag),end="")
        elif part == "real":
            def print_H_R(i,j):
                fs = "{:"+f+"} "
                print(fs.format(H_R[i,j].real),end="")
        elif part == "imag":
            def print_H_R(i,j):
                fs = "{:"+f+"} "
                print(fs.format(H_R[i,j].imag),end="")

        for i in range(self.n_bands):
            for j in range(self.n_bands):
                print_H_R(i,j)
            print("")
        return H_R

    def mod_H_R(self,R,mat):
        '''Modifies the H(R) for given R by adding the matrix.
        '''
        R_index = np.argwhere(np.all((self.R[:,:3]-R)==0, axis=1))
        self.hr[R_index[0,0]] += mat



if __name__== "__main__":
    print("Testing class hamiltonian...")
    real_vec = np.array([[3.0730000,    0.0000000,    0.0000000],[-1.5365000,    2.6612960,    0.0000000],[0.0000000,    0.0000000,   20.0000000]])
    my_ham = hamiltonian("test_ham/TaAs.dat",real_vec,True)
    print("Instance attributes of the generated Hamiltonian")
    print(my_ham.__dict__.keys())
    print('Testing function "get_H_R".')
    R = np.array([0,0,0])
    my_ham.get_H_R(R)


    print('''Testing function "mod_H_R"''')
    mat = np.eye(my_ham.n_bands)
    my_ham.mod_H_R(R,mat)
    print("New hamiltonian at R=",R)
    my_ham.get_H_R(R)

    print("Testing Fourier transform at the Gamma-point")
    gamma = np.array([0,0,0])
    hk = my_ham.hk(gamma)
    print("Shape H_k",np.shape(hk))
    print('Testing function "hr_spinless"...')
    my_ham.set_hr_spinless()
    print('Testing function "w2dynamics_hk"...')
    grid = np.array([[2,3,4]])
    w2name = "Hk_w2"
    time0 = time.time()
    my_ham.w2dynamics_hk(grid,w2name)
    print(time.time()-time0)
    print("Testing attribute basis")
    basis = np.array([0,1])
    basis_ham = hamiltonian("test_ham/hr_In_soc.dat",real_vec,True,basis)
    print("Testing k-derivative of the Hamiltonian")
    my_ham.del_hk(np.array([0.1,0.3,0.6]))
    print("Testing h_k for more dimensional k_array")
    k = np.array([[0,0,0],[1,0,0]])
    print(np.shape(my_ham.hk(k)))
    print("Testing single and all k-point h_k")
    time0s = time.time()
    k = np.random.rand(1000,3)
    h_single = np.zeros((1000,my_ham.n_bands,my_ham.n_bands),dtype="complex")
    for i in range(1000):
        h_single[i] = my_ham.hk(k[i])
    print("Time for single hk:", time.time()-time0s)
    print("Testing all k-point FT")
    time0a = time.time()
    h_all = my_ham.hk(k)
    print("Time for all-k hk:", time.time()-time0a)
    print("Both methods return same H(k):",np.allclose(h_single,h_all))
    if parallel == True:
        print("Testing k-parallelized FT:")
        time0p = time.time()
        h_all = my_ham.hk_parallel(k)
        print("Time for all-k hk:", time.time()-time0p)

    print("Testing single and all k-point del_h_k")
    time0s = time.time()
    k = np.random.rand(1000,3)
    del_hk_single = np.zeros((1000,3,my_ham.n_bands,my_ham.n_bands),dtype="complex")
    for i in range(1000):
        del_hk_single[i] = my_ham.del_hk(k[i])
    print("Time for single hk:", time.time()-time0s)
    print("Testing all k-point FT")
    time0a = time.time()
    del_hk_all = my_ham.del_hk(k)
    print("Time for all-k hk:", time.time()-time0a)
    print("Both methods return same H(k):",np.allclose(del_hk_single,del_hk_all))
