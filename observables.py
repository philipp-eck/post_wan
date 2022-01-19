#! /bin/python
### observables class, takes care of calculating the observables of the corresponding operators. 
import numpy as np
from operators   import operator
from hamiltonian import hamiltonian
from k_space     import k_space
import time
#import matplotlib.pyplot as plt

# Parallelization
parallel = True
if parallel:
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


class observables:
    ''' observables class, takes care of calculating the observables of the corresponding operators.
        When called, creates a dictionary with all operators.
        Generates output arrays and calculates the expectation value.
        Instance attributes:
        hamiltonian   # Hamiltonian
        k_space       # k_space on which the properties are calculated
        op_types      # List containing all operators to be calculated: {S,L,J}
        op_types_k    # List containing all k-dependent operators to be calculated: {BC,BC_mag,Orb_SOC_inv,E_triang}
        ops           # Dictionary containing the operators
        evals         # Array containing all eigen-values
        evecs         # Array containing all eigen-vectors
        prefix        # String, prefix of output files
    '''
       

    def __init__(self,HAMILTONIAN,K_SPACE,OP_TYPES=[],OP_TYPES_K=[],PREFIX=""):
        '''Initializes the observables object'''

        self.ham         = HAMILTONIAN
        self.k_space     = K_SPACE
        self.prefix      = PREFIX
        self.op_types    = OP_TYPES
        self.op_types_k  = OP_TYPES_K
        self.ops         = {}
     
        self.initialize_ops()
    

    def initialize_ops(self):
        '''Calls operator class, initializes all required attributes for calculating the exp val.
        '''
        for op_type in self.op_types:
            print("Initializing k-independent operator "+op_type+".")
            self.ops[op_type]     = operator(op_type,self.ham)

        for op_type_k in self.op_types_k:
            print("Inititalizing k-dependent operator "+op_type_k+".")
            self.ops[op_type_k]     = operator(op_type_k,self.ham)

    def initialize_op_val(self):
        '''
        Calls operator class, initializes array containing the expectation values.
        '''
        for op_type in self.op_types:
            self.ops[op_type].initialize_val(np.shape(self.k_space.k_space_red)[:-1])

        for op_type_k in self.op_types_k:
            self.ops[op_type_k].initialize_val_k(np.shape(self.k_space.k_space_red)[:-1])

    def calculate_ops(self,write=True,all_k=True,post=True,bmin=False,bmax=False):
        '''Calls H_R Fouriertransform, diagonalizes H_k, calculates Expectation values.'''

        self.initialize_op_val()
        print("Calculating operators on the given k-space...")

        def expval(evecs,op):
            '''Calculates <Psi|Op|Psi> along all dimensions of the operator on all k-points.'''
            val = np.einsum('...df,cde,...ef->...cf',np.conj(evecs),op,evecs,optimize=True).real
            return val
        shape_eval = self.k_space.k_space_red.shape[:-1] + (self.ham.n_bands,)
        shape_evec = self.k_space.k_space_red.shape[:-1] + (self.ham.n_bands,self.ham.n_bands)
        self.evals = np.zeros(shape_eval)
        self.evecs = np.zeros(shape_evec,dtype=np.csingle)

        def calc_k(i_k):
            '''Calls expval() for each operator on a single k-point.
               Transforms k_i.dim=1 to k_i.dim=2 for using the all k-point routines.'''
            hk = self.ham.hk(self.k_space.k_space_red[i_k])
            evals,evecs = np.linalg.eigh(hk,UPLO="U")
            self.evals[i_k], self.evecs[i_k] = evals, evecs
    
            for op_type in self.op_types:
                self.ops[op_type].val[i_k] = expval(evecs,self.ops[op_type].op)#[0]
    
            for op_type_k in self.op_types_k:
                    self.ops[op_type_k].val[i_k] = self.ops[op_type_k].expval(self.k_space.k_space_red[i_k],evals,evecs)#[0]

    
        def calc_serial():
            str_prog = "          "
            n_prog   = 1
            print("Progress: ["+str_prog+"]", end="\r", flush=True)
            nk = np.shape(self.k_space.k_space_red)[0]
            for i_k,k_i in enumerate(self.k_space.k_space_red):
               #k_i = np.array([k_i])
                calc_k(i_k)
                if i_k >= n_prog *nk/10.0:
                    str_prog = ""
                    for i_prog in range(10):
                        if i_prog < n_prog:
                            str_prog = str_prog+"="
                        else:
                            str_prog = str_prog+" "
                    print("Progress: ["+str_prog+"]", end="\r", flush=True)
                    n_prog += 1
            print("Progress: [==========]")
                           


        def calc_parallel():
            Parallel(num_cores,prefer="threads",require='sharedmem')(delayed(calc_k)(i_k) for i_k in range(np.shape(self.k_space.k_space_red)[0]))

        def calc_parallel_new():
            pool = multiprocessing.Pool(processes=num_cores)
            pool.map(calc_k,range(10))


        def calc_all_k():
            '''
            Diagonalize H(k) at all k-points in one step.
            '''
            time_hk = time.time()
            hk = self.ham.hk(self.k_space.k_space_red)
            print("Time for running H(k) FT:",time.time()-time_hk)
            time_eigh = time.time()
            self.evals,self.evecs = np.linalg.eigh(hk,UPLO="U")
            del hk
            print("Time for diagonalizing H(k):",time.time()-time_eigh)

            for op_type in self.op_types:
                time_op0 = time.time()
                self.ops[op_type].val = expval(self.evecs,self.ops[op_type].op)
                print("Time for calculating expectation value of operator "+op_type+":",time.time()-time_op0)
            for op_type_k in self.op_types_k:
                 time_op0 = time.time()
                 self.ops[op_type_k].val = self.ops[op_type_k].expval(self.k_space.k_space_red,self.evals,self.evecs)
                 print("Time for calculating expectation value of operator "+op_type_k+":",time.time()-time_op0)

        if all_k == True:
            print("Diagonalizing all k-points in parallel.")
            calc_all_k()
        else:
            if 'num_cores' in globals():
                print("Running k-parralelized on "+str(num_cores)+" cores.")
                calc_parallel()
            else:
                print("Running in serial mode.")
                calc_serial()
            

        if self.ham.ef != None:
            print("Shifting eigenvalues w.r.t. Fermi level...")
            self.evals -= self.ham.ef

        # Run post-processing
        if post == True:
            self.post_ops()
        # Write observables
        if write == True:
            self.write_ops(bmin,bmax)

    def post_ops(self):
        '''If defined, run the post-processing for the operators.'''
        for op_type in self.op_types+self.op_types_k:
            if type(self.ops[op_type].post) != None:
                print("Running post-processing for operator "+op_type+".")
                self.ops[op_type].post(self.evals)

    def k_int(self,sigma=0.05,wstep=0.001,write=True):
        '''Calculates k-integrated expecation values.'''
        for op_type in self.op_types+self.op_types_k:
            print("Calculating k-integrated values of "+op_type+".")
            self.ops[op_type].k_int(self.evals,sigma,wstep)

        if write==True:
           self.write_k_int()

    def sphere_winding(self):
        '''Calculates the Pontryagin-index for the observables, calculated on a sphere.
           Requires 'k_type="sphere_ster_proj"' !!!
        '''
        if self.k_space.k_type == "sphere_ster_proj":
            output = open(self.prefix+"pontryagin.dat","w")
            for op_type in self.op_types+self.op_types_k:
                self.ops[op_type].sphere_winding(self.k_space.k_space_proj,self.k_space.n_points)
                output.write("Calculating Pontryagin-index for operator "+op_type+".\n")
                for band in range(self.ham.n_bands):
                    output.write("Band {b:3d}: S={i:7.4f}\n".format(b=(band+1),i=self.ops[op_type].pont[band]))
            output.close()
        else:
            print('''Pontryagin index cannot be calculated, set k_type="sphere_ster_proj" !!!''')

    def write_ops(self,bmin=False,bmax=False):
        '''Writes calculated observables to output files.'''

        if not bmin:
            bmin = 1
        if not bmax:
            bmax = self.ham.n_bands + 1
        # Write only k-coordinates and eigenvalues
        print("Writing eigenvalues output.")
        if self.k_space.k_kind == "path":
            f_path = '{:4d}{:13.8f}{:13.8f}'
            f_spec = f_path+' \n'
            output = open(self.prefix+"evals_path.dat", "w")
            for band in range(bmin,bmax):
                for i_k in range(np.shape(self.k_space.k_space_red)[0]):
                    output.write(f_spec.format(band,self.k_space.k_dist[i_k],self.evals[i_k,band-1]))
                output.write("\n")
            output.close()

        if self.k_space.k_kind == "mesh":
            f_map = '{:4d}{k[0]:13.8f}{k[1]:13.8f}{k[2]:13.8f}{e:13.8f}'
            f_spec = f_map+' \n'
            output = open(self.prefix+"evals_map.dat", "w")
            for band in range(bmin,bmax):
                for i_k in range(np.shape(self.k_space.k_space_red)[0]):
                    output.write(f_spec.format(band,k=self.k_space.k_space_car[i_k],e=self.evals[i_k,band-1]))
                    if (i_k+1)%np.sqrt(np.shape(self.k_space.k_space_red)[0]) ==0:
                       output.write("\n")
                output.write("\n")
            output.close()

        #val

        for op_type in self.op_types+self.op_types_k:
            print("Writing output for operator "+op_type+".") 
            if self.k_space.k_kind == "path":
                f_path = '{:4d}{:13.8f}{:13.8f}'
                f_spec = f_path+self.ops[op_type].f_spec+' \n'
                output = open(self.prefix+op_type+"_path.dat", "w")
                for band in range(bmin,bmax):
                    for i_k in range(np.shape(self.k_space.k_space_red)[0]):
                        output.write(f_spec.format(band,self.k_space.k_dist[i_k],self.evals[i_k,band],d=self.ops[op_type].val[i_k,:,band+1]))
                    output.write("\n")
                output.close()


            if self.k_space.k_kind == "mesh":
                f_map = '{:4d}{k[0]:13.8f}{k[1]:13.8f}{k[2]:13.8f}{e:13.8f}'
                f_spec = f_map+self.ops[op_type].f_spec+' \n'
                output = open(self.prefix+op_type+"_map.dat", "w")
                for band in range(self.ham.n_bands):
                    for i_k in range(np.shape(self.k_space.k_space_red)[0]):
                        output.write(f_spec.format(band+1,k=self.k_space.k_space_car[i_k],e=self.evals[i_k,band],d=self.ops[op_type].val[i_k,:,band]))
                        if (i_k+1)%np.sqrt(np.shape(self.k_space.k_space_red)[0]) ==0:
                           output.write("\n")
                    output.write("\n")
                output.close()

        #val_b_int

        for op_type in self.op_types+self.op_types_k:
            if type(self.ops[op_type].val_b_int) ==  np.ndarray:
                print("Writing band-integrated output for operator "+op_type+".")
                if self.k_space.k_kind == "path":
                    f_path = '{:13.8f}'
                    f_spec = f_path+self.ops[op_type].f_spec+' \n'
                    output = open(self.prefix+op_type+"_b_int_path.dat", "w")
                    for i_k in range(np.shape(self.k_space.k_space_red)[0]):
                        output.write(f_spec.format(self.k_space.k_dist[i_k],d=self.ops[op_type].val_b_int[i_k,:]))
                    output.write("\n")
                    output.close()


                if self.k_space.k_kind == "mesh":
                    f_map = '{k[0]:13.8f}{k[1]:13.8f}{k[2]:13.8f}'
                    f_spec = f_map+self.ops[op_type].f_spec+' \n'
                    output = open(self.prefix+op_type+"_b_int_map.dat", "w")
                    for i_k in range(np.shape(self.k_space.k_space_red)[0]):
                        output.write(f_spec.format(k=self.k_space.k_space_car[i_k],d=self.ops[op_type].val_b_int[i_k,:]))
                        if (i_k+1)%np.sqrt(np.shape(self.k_space.k_space_red)[0]) ==0:
                           output.write("\n")
                    output.close()

        #val_k_int
        # to be written, not used yet...

    def write_k_int(self):
        for op_type in self.op_types+self.op_types_k:
            if type(self.ops[op_type].val_k_int) ==  np.ndarray:
                print("Writing k-integrated output for operator "+op_type+".")
                f_ener = '{e:13.8f}{dos:13.8f}'
                f_spec = f_ener+self.ops[op_type].f_spec+' \n'
                np.savetxt(self.prefix+op_type+"_DOS.dat",self.ops[op_type].val_k_int.transpose((1,0)),fmt='%16.6e')
                np.savetxt(self.prefix+op_type+"_DOS_E_int.dat",self.ops[op_type].val_kE_int.transpose((1,0)),fmt='%16.6e')

class z2_wcc:
    '''Class for calculation Z_2 invariants following the approach of A. Soluyanov and D. Vanderbilt.
    '''
    def __init__(self,HAMILTONIAN,N_PUMP,NORMAL):
        '''Initializes the z2-calculation object, calls the calculation function'''
        self.ham     = HAMILTONIAN
        self.n_pump  = N_PUMP
        self.normal  = NORMAL
        if self.ham.spin == False:
            print("Hamiltonian is spin-less, cannot calculate Z_2 invariant")
        self.z2_calc()

    def z2_calc(self):
        '''Calculates the Wannier Charge Centers (WCC) movement.
        '''
        self.wcc = np.zeros((2,self.n_pump[0],self.ham.n_elec))
        self.gap = np.zeros((2,self.n_pump[0]))
        #pumping index
        p = self.normal%3
        #FT index
        f = (self.normal+1)%3
        #normal index
        n = (self.normal+2)%3
        for k3 in range(2):
            if k3 == 0:
                print("Calculating Z_2 invariant for Gamma-plane...")
            else:
                print("Calculating Z_2 invariant for BZ-surface plane...")
            k_normal = k3*0.5
            m_tot = 0 
            for k1 in range(self.n_pump[0]):
                k = np.zeros((self.n_pump[1],3))
                k[:,p] = 0.5/(self.n_pump[0]-1)*k1
                k[:,f] = np.linspace(0,1,self.n_pump[1],endpoint=False)
                k[:,n] = k_normal
                hk = np.zeros((self.n_pump[1],self.ham.n_bands,self.ham.n_bands),dtype=np.csingle)
                #for k2 in range(self.n_pump[1]):
                #    hk[k2] = self.ham.hk(k[k2])
                
                evals,evecs = np.linalg.eigh(self.ham.hk(k),UPLO="U")
                M = np.zeros((self.ham.n_elec,self.ham.n_elec),dtype=np.csingle)+np.identity(self.ham.n_elec)
                for k2 in range(self.n_pump[1]-1):
                    M = np.einsum("ij,jk",M,np.einsum("ji,jk->ik",np.conj(evecs[k2,:,:self.ham.n_elec]),evecs[k2+1,:,:self.ham.n_elec]))
                M = np.einsum("ij,jk",M,np.einsum("ji,jk->ik",np.conj(evecs[self.n_pump[1]-1,:,:self.ham.n_elec]),evecs[0,:,:self.ham.n_elec]))
                #Singular value decomposition of the overlap Matrix M
                u,s,vh = np.linalg.svd(M)
                U = np.einsum("ij,jk",u,vh)
                y=np.linalg.eigvals(U)
                #Localization of the Wannier centers: log of the complex valued U eigenvalues
                y = np.log(y)
                self.wcc[k3,k1]=np.sort(np.imag(y)%(2*np.pi)/(2*np.pi))
            #sort for calculating largest gap
            #likely not needed
            self.wcc[k3] = np.sort(self.wcc[k3],axis=1)
            gap = (np.roll(self.wcc[k3],-1,axis=1)+0.999999-self.wcc[k3])%1
            gap_i = np.argmax(gap,axis=1)
            for k1 in range(self.n_pump[0]):
                self.gap[k3,k1] = (self.wcc[k3,k1,gap_i[k1]]+gap[k1,gap_i[k1]]/2.0)%1

            # Computing the directed area of the triangle spanned by the wannier center x_i+1 and the lagest gaps z_i and z_i+1
            for k1 in range(self.n_pump[0]-1):
                del_m = 1
                k = (k1+1)%self.n_pump[0]
                g = np.sin(2*np.pi*(self.gap[k3,k,None]-self.gap[k3,k1,None])) + np.sin(2*np.pi*(self.wcc[k3,k]-self.gap[k3,k,None])) + np.sin(2*np.pi*(self.gap[k3,k1,None]-self.wcc[k3,k,None]))
                if 0 in g:
                   print("O in directed area encountered, be cautious, check wccs and gap!!!")
                g[g>=0] = 1
                g[g<0]  = -1
                del_m = np.prod(g)
#               for oc in range(self.ham.n_elec):
#                    g = np.sin(2*np.pi*(self.gap[k3,k]-self.gap[k3,k1])) + np.sin(2*np.pi*(self.wcc[k3,k1,oc]-self.gap[k3,k])) + np.sin(2*np.pi*(self.gap[k3,k1]-self.wcc[k3,k1,oc]))
#                    if -gap_tol <= self.wcc[k1,2]-self.wcc[k1+1,n+3] <= gap_tol:
#                         print "WARNING: z_m is very close to x_m+1!!! Del = "+str(self.wcc[k1,2]-self.wcc[k1+1,n+3])
#                    if g >= 0:
#                         del_m = del_m * 1
#                    else:
#                         del_m = del_m *(-1)
                if del_m < 0:
                    m_tot = m_tot + 1
                    print("Jump at pump:"+str(k1+1))
            z2 = m_tot % 2
            print("Z_2="+str(z2)) 

    def plot_wcc(self):
        '''Plots the wccs and the largest gap.'''
        fig_wcc = plt.figure(num=None, figsize=(20, 5))
        plt.subplot(121)
        plt.plot(self.wcc[0],'bs')
        plt.plot(self.gap[0],'r-')
        plt.subplot(122)
        plt.plot(self.wcc[1],'bs')
        plt.plot(self.gap[1],'r-')
        plt.show

    def write_wcc(self):
        '''Writes the wannier charge centers and the largest gap to an output file'''
        print("Writing WCC output...")
        fss = "{k:8.4f}{g:8.4f}"
        for n in range(self.ham.n_elec):
            fss +="{wcc["+str(n)+"]:8.4f}"
        fss += " \n"
        for k3 in range(2):
            if k3 == 0:
                output = open("WCC_gam.dat","w")
            else:
                output = open("WCC_sur.dat","w")
            for k in range(self.n_pump[0]):
                output.write(fss.format(k=0.5*k/(self.n_pump[0]-1),g=self.gap[k3,k],wcc=self.wcc[k3,k]))
            output.close()
        print("Finished writing WCC output.")


if __name__== "__main__":
    print("Testing class observables...")
    print("Creating hamiltonian...")
    real_vec = np.array([[3.0730000,    0.0000000,    0.0000000],[-1.5365000,    2.6612960,    0.0000000],[0.0000000,    0.0000000,   20.0000000]])
    basis = np.array([0,1])
    ef = 1
    prefix = "test_data/"
    my_ham = hamiltonian("test_ham/hr_In_soc.dat",real_vec,True,basis,ef)
    print("Creating k_space...")
    vecs=np.array([[0,0,0],[1/3,1/3,0],[2/3,-1/3,0]])
    points = 3 
    path = k_space("path","red",vecs,my_ham.bra_vec,points)
    print("Testing operator 'E_triang' and k-integration...")
    nbins = 100
    sigma = 0.6
    op_types_k = ["E_triang"]
    op_types   = ["L"] 
    o_triang = observables(my_ham,path,op_types,OP_TYPES_K=op_types_k,PREFIX=prefix) 
    o_triang.calculate_ops()
    o_triang.k_int(nbins,sigma)
    print("Creating Spin-observable...")
    op_types = ["S"]
    o_spin = observables(my_ham,path,op_types,PREFIX=prefix)
    print("Instance attributes of the generated spin-operator")
    print(o_spin.__dict__)
    print('Testing function "calculate_ops"...')
    o_spin.calculate_ops()
    print("Testing calculate_ops on all k-points.")
    o_spin.calculate_ops(all_k=True)
    print('Testing function "post_ops"...')
    o_spin.post_ops()
   #print(o_spin.ops["S"].val)
    print('Testing function "write_ops"...')
    o_spin.write_ops()
    print("Testing calculation on a mesh...")
    vecs = np.array([[-1,0,0],[1,0,0],[0,1,0]])
    plane = k_space("plane","car",vecs,real_vec,points)
    op_types   = ["S","L"]
    op_types_k = ["BC"]
    o_plane = observables(my_ham,plane,op_types,op_types_k,prefix)
    o_plane.calculate_ops(all_k=False)
    o_plane.post_ops()
    o_plane.write_ops()
    print("Testing calculation with all-k routines")
    o_plane_all_k = observables(my_ham,plane,op_types,op_types_k,prefix)
    o_plane_all_k.calculate_ops()
    o_plane_all_k.post_ops()
    o_plane_all_k.write_ops()
    for myops in op_types+op_types_k:
        if np.allclose(o_plane.ops[myops].val,o_plane_all_k.ops[myops].val,atol=1e-05) == True:
            print("Expectation value of operator "+myops+" is identical.")
        else:
            print("Expectation value of operator "+myops+" is not identical!!!")
        print(np.amax(np.abs((o_plane.ops[myops].val-o_plane_all_k.ops[myops].val))))

    print("Testing calculation for higher dimensional k-arrays")
    k1=np.random.rand(9,7,4,3)
    k2=k1.reshape((9*7*4,3))
    k_flat = k_space('self-defined','red',k2,real_vec)
    k_high = k_space('self-defined','red',k1,real_vec)
    o_flat =  observables(my_ham,k_flat,op_types,op_types_k)
    o_flat.k_space.k_space_red = k2
    o_high =  observables(my_ham,k_high,op_types,op_types_k)
    ALL_K = False
    o_flat.calculate_ops(write=False,all_k=ALL_K,post=True)
    o_high.calculate_ops(write=False,all_k=ALL_K,post=True)
  
    print("Eigenvalues are identical?:",
          np.allclose(o_flat.evals.flatten(),o_high.evals.flatten()))
    print("Eigenvectors are identical?:",
          np.allclose(o_flat.evecs.flatten(),o_high.evecs.flatten()))
    for myops in op_types+op_types_k:
        if np.allclose(o_flat.ops[myops].val.flatten(),
                       o_high.ops[myops].val.flatten(),
                       atol=1e-02,equal_nan=True) == True:
            print("Expectation value of operator "+myops+" is identical.")
        else:
            print("Expectation value of operator "+myops+" is not identical!!!")
        print("Max deviation:",np.amax(np.abs((o_flat.ops[myops].val.flatten()-o_high.ops[myops].val.flatten()))))    
   
