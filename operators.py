#! /bin/python
### operator class, sub-class of observables. 
import numpy as np
import scipy.ndimage
from hamiltonian import hamiltonian 
class operator:
    ''' operator class, sub-class of observables.
        When called, the class generates an operator with all required attributes.
        The object operator is meant to be passed to create the dict observables.
        Instance attributes:
        op_type       # Type of operator: {S,L,J,BC,BC_mag,Orb_SOC_inv,E_triang}
        op            # Generated operator
        val           # Array containing the output [kpoint,dim,n_bands]
        val_b_int     # Band integrated output
        val_k_int     # k integrated output
        f_spec        # format specifier string, for writing the output
        prec          # precision used in the format specifier
        post          # Function for postprocessing
        ham           # Hamiltonian, required for Hamiltonian dependent operators
        expval        # Function, wich is called for k-dependent operators for calculating the expectation value
    '''
       

    def __init__(self,OP_TYPE,HAMILTONIAN=None):
        '''Initializes the operator object'''

        self.op_type = OP_TYPE
        self.op      = None
        self.val     = None
        self.val_b_int = None
        self.val_k_int = None
        self.f_spec  = ''
        self.prec    = "9.4f"
        self.post    = self.no_post
        self.ham     = HAMILTONIAN
        self.expval  = None
        if self.op_type == "S":
            self.set_S_op()
        elif self.op_type == "L":
            self.set_L_op()
        elif self.op_type == "J":
            self.set_J_op()
        elif self.op_type == "BC":
            self.set_BC_op()
        elif self.op_type == "BC_mag":
            self.set_BC_mag_op()
        elif self.op_type == "Orb_SOC_inv":
            self.set_Orb_SOC_inv()
        elif self.op_type == "E_triang":
            self.set_E_triang()
        else:
            ### Create empty k-independent operator
            print('Creating empty operator "'+self.op_type+'".')
        #   self.set_empty_op()

    def set_empty_op(self):
        '''Creates an empty k-independent operator'''
        self.op = np.zeros((self.dim,self.ham.n_bands,self.ham.n_bands),dtype=complex)
    
    def set_S_op(self):
        '''Sets the spin-operators in the wannier90 (VASP) basis.
           Spin is the slowest index!!!
        '''
        if self.ham.spin == False:
            print("Spin-operators cannot be defined for spin-less System...")
        else:
            self.op  = self.S_op()
            self.post = self.L_2


    def set_L_op(self):
        '''Sets L-operator in wannier90 (VASP) basis.
           Spin is the slowest index!!!'''
        self.op  = self.L_op()
        self.post = self.L_2

    def set_J_op(self):
        '''Sets J-operator in wannier90 (VASP) basis.
           Spin is the slowest index!!!'''
        ### J^2 and J_i in tesseral harmonics basis
        self.op = self.J_op()
        self.post = self.L_2


    def set_BC_op(self):
        '''Sets BC-operator, requires nabla_k H(k).'''
        self.prec    = "12.3e"
        self.op = self.BC_op
        self.expval = self.BC_expval
        self.expval_all_k = self.BC_expval 
        self.post = self.b_int_ef

    def set_BC_mag_op(self):
        '''Sets BC_mag-operator, requires nabla_k H(k).'''
        self.prec    = "12.3e"
        self.op = self.BC_mag_op
        self.expval = self.BC_mag_expval
        self.expval_all_k = self.BC_mag_expval
        self.post = self.b_int_ef

    def set_Orb_SOC_inv(self):
        '''Sets Orb-SOC-inversion operator.'''
        self.op = self.Orb_SOC_inv_op
        self.expval = self.Orb_SOC_inv_expval
        self.expval_all_k = self.Orb_SOC_inv_expval

    def set_E_triang(self):
        '''Sets the operator for calculating the Euler localization in triangular lattices.'''
        self.op = self.E_triang_op
        self.expval = self.E_triang_expval
        self.expval_all_k = self.E_triang_expval
    ####Operators

    def S_op(self,k=None,evals=None,evecs=None):
        '''Returns spin-operators in the wannier90 (VASP) basis.
           Spin is the slowest index!!!
        '''
        one = np.diag(np.ones(self.ham.n_orb))
        sx = np.array([[0,1],[1,0]])/2.0
        sy = np.array([[0,-1],[1,0]])*1j/2.0
        sz = np.array([[1,0],[0,-1]])/2.0
        S = np.zeros((4,self.ham.n_orb*2,self.ham.n_orb*2),dtype=complex)
        S[:3] = np.array((np.kron(sx,one),np.kron(sy,one),np.kron(sz,one)))
        ### S^2
        for i in range(3):
            S[3] += np.einsum('ij,jk->ik',S[i],S[i])
        return S

    def L_op(self,k=None,evals=None,evecs=None):
        '''Returns L-operator in wannier90 (VASP) basis.
           Spin is the slowest index!!!'''

        #### Here we up build the l operator ####
        # note that the ordering of the l basis will be {0,+1,-1,...,+l,-l}
        # L_x =  1/2(L_+ + L_-)
        # L_y = -i/2(L_+ - L_-)
        # L_+|j,m> = hbar sqrt(j(j+1)-m(m+1))|j,m+1>
        # L_-|j,m> = hbar sqrt(j(j+1)-m(m-1))|j,m-1>
        L = np.zeros((4,self.ham.n_orb,self.ham.n_orb),dtype=complex)
        # we rotate the L-operator in the tesseral harmonics eigenbasis
        ### build up rotation matrix
        u = np.zeros((self.ham.n_orb,self.ham.n_orb), dtype=complex)
        ind = 0
        for j in self.ham.basis:
            u[ind,ind] = 1
            ind += 1
            for m in range(1,j+1,1):
               u[ind  ,ind  ] = (-1)**m /np.sqrt(2)
               u[ind  ,ind+1] =-(-1)**m *1j /np.sqrt(2) ### follows from backtransform
               u[ind+1,ind  ] =   1/np.sqrt(2)
               u[ind+1,ind+1] =  1j/np.sqrt(2)
               ind += 2

        u = np.asmatrix(u)

        #### L_x ####
        ### build up L_x operator
        op = np.zeros((self.ham.n_orb,self.ham.n_orb))
        i0 = 0                                    # index of the m=0 element
        for j in self.ham.basis:
            if j > 0:
               op[i0,i0+1] = np.sqrt(j*(j+1))     # 0->1
               op[i0+1,i0] = np.sqrt(j*(j+1))
               op[i0,i0+2] = np.sqrt(j*(j+1))     # 0->-1
               op[i0+2,i0] = np.sqrt(j*(j+1))
               for m in range(1,j,1):
                   op[i0+2*m-1,i0+2*m+1] = np.sqrt(j*(j+1)-m*(m+1))
                   op[i0+2*m+1,i0+2*m-1] = np.sqrt(j*(j+1)-m*(m+1))
                   op[i0+2*m  ,i0+2*m+2] = np.sqrt(j*(j+1)-m*(m+1))
                   op[i0+2*m+2,i0+2*m  ] = np.sqrt(j*(j+1)-m*(m+1))
            i0 += 2*j + 1
        L[0] = np.matmul(u.getH(),np.matmul(op/2.0,u))
        Lx = op

        #### L_y ####
        ### build up L_y operator
        op = np.zeros((self.ham.n_orb,self.ham.n_orb),dtype=complex)
        i0 = 0                                    # index of the m=0 element
        for j in self.ham.basis:
            if j > 0:
               op[i0,i0+1] = np.sqrt(j*(j+1))*1j     # 0->1
               op[i0+1,i0] =-np.sqrt(j*(j+1))*1j
               op[i0,i0+2] =-np.sqrt(j*(j+1))*1j     # 0->-1
               op[i0+2,i0] = np.sqrt(j*(j+1))*1j
               for m in range(1,j,1):
                   op[i0+2*m-1,i0+2*m+1] = np.sqrt(j*(j+1)-m*(m+1))*1j
                   op[i0+2*m+1,i0+2*m-1] =-np.sqrt(j*(j+1)-m*(m+1))*1j
                   op[i0+2*m  ,i0+2*m+2] =-np.sqrt(j*(j+1)-m*(m+1))*1j
                   op[i0+2*m+2,i0+2*m  ] = np.sqrt(j*(j+1)-m*(m+1))*1j
            i0 += 2*j + 1
        L[1] = np.matmul(u.getH(),np.matmul(op/2.0,u))

        #### L_z ####
        ### build up L_z operator ###
        l_ar = []
        for j in self.ham.basis:
            l_ar.append(0)
            if j > 0:
               for m in range(1,j+1,1):
                   l_ar.append(m)
                   l_ar.append(-m)

        l_ar = np.array(l_ar)
        l_z  = np.diag(l_ar)
        ### compute rotated lz operator and lz expectation values
        L[2] = np.matmul(u.getH(),np.matmul(l_z,u))

        for i in range(3):
            L[3] += np.einsum('ij,jk->ik',L[i],L[i])

        if self.ham.spin == True:
            one = np.diag(np.ones(2))
            L = np.kron(one,L)
        return L

    def J_op(self,k=None,evals=None,evecs=None):
        '''Returns J-operator in wannier90 (VASP) basis.
           Spin is the slowest index!!!'''
        #### Building the J-Operators ####
        # Strategy: Since we have already L and S, we just add them vectorially in the tesseral harmonics basis
        # Note: We compute also J^2 -> J=(J_x,J_y,_z,J^2)

        ### J^2 and J_i in tesseral harmonics basis
        J = np.zeros((4,self.ham.n_bands,self.ham.n_bands),dtype=complex)
        S = self.S_op()
        L = self.L_op()
        for i in range(3):
            J[i] = L[i]+S[i]
            J[3] += np.einsum('ij,jk->ik',J[i],J[i])
        return J


    def BC_op(self,k=None,evals=None,evecs=None):
        '''Generates an empty array.
           BC expectation value is calculated with a more efficient function.
           Components 1-3: Berry curvature
           !!!Currently not used!!!: Components 4-6: Orbital moment of Bloch state
        '''
        BC_op = np.zeros((3,self.ham.n_bands))
        return BC_op


    def BC_mag_op(self,k=None,evals=None,evecs=None):
        '''Generates an empty array.
           Calculates the orbital moment of the Bloch state
        '''
        BC_mag_op = np.zeros((3,self.ham.n_bands))
        return BC_mag_op

    def BC_expval(self,k=None,evals=None,evecs=None):

        if k.ndim == 1:
            #Nominator <n|nabla_k H_k|m>
            #To get the BC-operator
           #n_del_m_old = np.einsum("...ji,...jk",evecs.conj()[None],np.einsum("...ij,...jk",self.ham.del_hk(k),evecs[None]))
            n_del_m = np.einsum('db,cde,ef->cbf',evecs.conj(),self.ham.del_hk(k),evecs,optimize=True)

            #### Cheaty way to set diag of n_del_m to 0
            non_diag = np.ones_like(n_del_m)-np.diag(np.ones(self.ham.n_bands))[None]
            n_del_m *=non_diag
     
     
            #### Compute the denominator (E_m-E_n) and set diagonal to 1
            Em_En  = np.reshape(np.kron(np.ones(self.ham.n_bands),evals),(self.ham.n_bands,self.ham.n_bands))
            Em_En -= np.transpose(Em_En)
            np.fill_diagonal(Em_En,1) 

            n_del_m /= Em_En
    
            #### Compute BC, use np.roll to generate the permutation
           #BC_old = 2*np.imag(np.einsum('...ii->...i',np.einsum('...ji,...jk->...ik',np.roll(np.conj(n_del_m),-1,axis=0),np.roll(n_del_m,-2,axis=0))))
            BC =-2*np.imag(np.einsum('dji,dji->dj',np.roll(np.conj(n_del_m),-1,axis=0),np.roll(n_del_m,-2,axis=0),optimize=True))
            #### Calculate orbital moment of the Bloch state
            #         <n|nabla_k H_k|m>x<m|nabla_k H_k|n> 
            # m = -Im -----------------------------------
            #                     E_m - E_n
           #orb_mom = 2*np.imag(np.einsum('dji,dji->di',np.roll(np.conj(n_del_m),-1,axis=0),np.roll(n_del_m*Em_En,-2,axis=0),optimize=True))
           #out = np.concatenate((BC,orb_mom))
        if k.ndim == 2:
            #Nominator <n|nabla_k H_k|m>
            #To get the BC-operator
           #path_ndelm = np.einsum_path('kdb,kcde,kef->kcbf',np.array([evecs[0].conj()]),np.array([self.ham.del_hk(k)[0]]),np.array([evecs[0]]),optimize='optimal')[0]
            n_del_m = np.einsum('kdb,kcde,kef->kcbf',evecs.conj(),self.ham.del_hk(k),evecs,optimize=True)
            #### Cheaty way to set diag of n_del_m to 0
            non_diag = np.ones_like(n_del_m)-np.diag(np.ones(self.ham.n_bands))[None,None]
            n_del_m *=non_diag


            #### Compute the denominator (E_m-E_n) and set diagonal to 1
            Em_En  = np.reshape(np.kron(np.ones(self.ham.n_bands),evals),(np.shape(k)[0],self.ham.n_bands,self.ham.n_bands))
            Em_En -= np.transpose(Em_En,(0,2,1))
            [np.fill_diagonal(Em_En[i],1) for i in range(k.shape[0])]

            n_del_m /= Em_En[:,None]
            #### Calculate Berry curvature                  
            #         <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
            # m = -Im -----------------------------------
            #                    (E_m - E_n)^2
            # Note: the expectation value <m|nabla_k H_k|n>/(E_m - E_n) are written as line vectors in n_del_m
            # use np.roll to generate the permutation
           #path_BC = np.einsum_path('kdji,kdji->kdi',np.array([np.roll(np.conj(n_del_m),-1,axis=1)[0]]),np.array([np.roll(n_del_m,-2,axis=1)[0]]),optimize='optimal')
            BC =-2*np.imag(np.einsum('kdji,kdji->kdj',np.roll(np.conj(n_del_m),-1,axis=1),np.roll(n_del_m,-2,axis=1),optimize=True))

        return BC
    def BC_mag_expval(self,k=None,evals=None,evecs=None):

        if k.ndim == 1:
            #Nominator <n|nabla_k H_k|m>
            #To get the BC-operator
            n_del_m = np.einsum('db,cde,ef->cbf',evecs.conj(),self.ham.del_hk(k),evecs,optimize=True)

            #### Cheaty way to set diag of n_del_m to 0
            non_diag = np.ones_like(n_del_m)-np.diag(np.ones(self.ham.n_bands))[None]
            n_del_m *=non_diag


            #### Compute the denominator (E_m-E_n) and set diagonal to 1
            Em_En  = np.reshape(np.kron(np.ones(self.ham.n_bands),evals),(self.ham.n_bands,self.ham.n_bands))
            Em_En -= np.transpose(Em_En)
            np.fill_diagonal(Em_En,1)

            n_del_m /= Em_En

            #### Calculate orbital moment of the Bloch state
            #         <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
            # m = -Im -----------------------------------
            #                     E_m - E_n
            orb_mom =+2*np.imag(np.einsum('dji,dji->dj',np.roll(np.conj(n_del_m),-1,axis=0),np.roll(n_del_m*Em_En,-2,axis=0),optimize=True))
        if k.ndim == 2:
            #Nominator <n|nabla_k H_k|m>
            #To get the BC-operator
            n_del_m = np.einsum('kdb,kcde,kef->kcbf',evecs.conj(),self.ham.del_hk(k),evecs,optimize=True)
            #### Cheaty way to set diag of n_del_m to 0
            non_diag = np.ones_like(n_del_m)-np.diag(np.ones(self.ham.n_bands))[None,None]
            n_del_m *=non_diag


            #### Compute the denominator (E_m-E_n) and set diagonal to 1
            Em_En  = np.reshape(np.kron(np.ones(self.ham.n_bands),evals),(np.shape(k)[0],self.ham.n_bands,self.ham.n_bands))
            Em_En -= np.transpose(Em_En,(0,2,1))
            [np.fill_diagonal(Em_En[i],1) for i in range(k.shape[0])]

            n_del_m /= Em_En[:,None]
            #### Calculate orbital moment of the Bloch state
            #         <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
            #l_z= +Im -----------------------------------
            #                     E_m - E_n
            # Note: global minus sign arising from magnetic moment of the electron
            # BC = 2*np.imag(np.einsum('kdji,kdji->kdi',np.roll(np.conj(n_del_m),-1,axis=1),np.roll(n_del_m,-2,axis=1),optimize=True))
            orb_mom =+2*np.imag(np.einsum('kdji,kdji->kdj',np.roll(np.conj(n_del_m),-1,axis=1),np.roll(n_del_m*Em_En[:,None],-2,axis=1),optimize=True))
        return orb_mom
 
    def Orb_SOC_inv_op(self,k=None,evals=None,evecs=None):
        '''Generates an empty array.
           orb_soc_inv expectation value is calculated with a more efficient function.
           When called, initialized the creation of a spin-less Hamiltonian'''
        orb_soc_inv_op = np.zeros((3,self.ham.n_bands))
        self.ham.hr_spinless()
        return orb_soc_inv_op

    def Orb_SOC_inv_expval(self,k=None,evals=None,evecs=None):
        '''Here we project eigenstates computed with H^soc onto the occupied eigenstates obtained without SOC
           n_occ: Total number of electrons
           We compute with Psi (soc) and psi (wo soc) the expectation value:
           <O_nk> = |<Psi_nk|Sum_i^n_occ |psi_ik><psi_ik|Psi_nk>
           We repeat this by projecting onto the unoccupied states
           Summation over both projectors is a complete projection --> Has to yield 1!!!!
        '''
        if k.ndim == 1:

            out = np.zeros((3,self.ham.n_bands))
            n_occ = self.ham.n_elec//2
            one2 = np.eye(2)
            hk_wo_soc = self.ham.hk_spinless(k)
            evals_wo_soc,evecs_wo_soc = np.linalg.eigh(hk_wo_soc,UPLO="U")
            # evals_wo_soc: 1. dimension of output
            if self.ham.ef != None:
               evals_wo_soc -= self.ham.ef
            out[0] = np.kron(evals_wo_soc,np.array([1,1]))
            
            #projection onto occupied states
            proj_occ = np.kron(one2,evecs_wo_soc[:,0:n_occ])
            val_occ = np.einsum('ji,jk->ik',np.conj(proj_occ),evecs)
            out[1]  = np.sum(np.abs(val_occ)**2,axis=0).real
        
    
            #projection onto unoccupied states
            proj_un_occ = np.kron(one2,evecs_wo_soc[:,n_occ:])
            val_un_occ = np.einsum('ji,jk->ik',np.conj(proj_un_occ),evecs)
            out[2]     = np.sum(np.abs(val_un_occ)**2,axis=0).real    

        if k.ndim == 2:

            out = np.zeros((np.shape(k)[0],3,self.ham.n_bands))
            n_occ = self.ham.n_elec//2
            one2 = np.eye(2)
            hk_wo_soc = self.ham.hk_spinless(k)
            evals_wo_soc,evecs_wo_soc = np.linalg.eigh(hk_wo_soc,UPLO="U")
            # evals_wo_soc: 1. dimension of output
            if self.ham.ef != None:
               evals_wo_soc -= self.ham.ef
            out[:,0] = [np.kron(evals_wo_soc[i_k],np.array([1,1])) for i_k in range(np.shape(k)[0])]

            #projection onto occupied states
            proj_occ = [np.kron(one2,evecs_wo_soc[i_k,:,0:n_occ]) for i_k in range(np.shape(k)[0])]
            val_occ = np.einsum('aji,ajk->aik',np.conj(proj_occ),evecs)
            out[:,1]  = np.sum(np.abs(val_occ)**2,axis=1).real


            #projection onto unoccupied states
            proj_un_occ = [np.kron(one2,evecs_wo_soc[i_k,:,n_occ:]) for i_k in range(np.shape(k)[0])]
            val_un_occ = np.einsum('aji,ajk->aik',np.conj(proj_un_occ),evecs)
            out[:,2]     = np.sum(np.abs(val_un_occ)**2,axis=1).real
        
        return out

    def E_triang_op(self,k=None,evals=None,evecs=None):
        '''Generates an empty array.
           Expectation value is calculated with E_triang_expval.
        '''
        e_triang_op = np.zeros((2,self.ham.n_bands))
        return e_triang_op

    def E_triang_expval(self,k=None,evals=None,evecs=None):
        '''Calculates the Euler localization by using the projector:

           P = Sum Sum exp[i(l_z*phi_R+k*R)]  Y^l_z><Y^l_z|
               l_z   R

           where phi_R is the polar angle defining the orientation of the Euler point w.r.t. site located at R.
           Y^l_z are spherical harmonics, here we use the eigenbasis transformation used also for the L-operator.
           i*k*R denotes the bloch phase.
           !!!ATTENTION: Assumes PBCs in the xy-plane!!! 
        '''

        # Define atom site orientation by calculating the Bravais vec orientation.
        # uc: __       __
        #     \__\ or /__/  (Bravais vectors starting in the lower-left corner.

        if np.dot(self.ham.bra_vec[0],self.ham.bra_vec[1]) < 0:
           R = np.array([[[0,0,0],[1,0,0],[1,1,0]],   # 1.Euler point
                         [[0,0,0],[1,1,0],[0,1,0]]])   # 2.Euler point
        elif np.dot(self.ham.bra_vec[0],self.ham.bra_vec[1]) < 0:
           R = np.array([[[0,0,0],[1,0,0],[0,1,0]],   # 1.Euler point
                         [[1,0,0],[1,1,0],[0,1,0]]])   # 2.Euler point
           print("Not thoroughly tested for sharp angle Bravais lattice definition!!!")
        else:
           print("Bravais vectors are orthogonal, is this a triangular lattice?!!!")

        phi = np.arange(3)/3
        # we rotate the L-operator in the tesseral harmonics eigenbasis
        ### build up rotation matrix
        if k.ndim == 2:
            P = np.zeros((2,self.ham.n_bands,self.ham.n_bands,np.shape(k)[0]), dtype=complex)
        else:
            P = np.zeros((2,self.ham.n_bands,self.ham.n_bands), dtype=complex)
        def phase(lz,R,k):
            if k.ndim == 2:
                out = np.exp(1j*2*np.pi*(lz*phi[None,None,:]+np.einsum("Kj,Eij->EKi",k,R))).sum(axis=2)/3.0
            else:
                out = np.exp(1j*2*np.pi*(lz*phi[None,:]+np.einsum("j,Eij->Ei",k,R))).sum(axis=1)/3.0
            return out

        if self.ham.spin == False:
            basis = self.ham.basis
        else:
            basis = np.append(self.ham.basis,self.ham.basis)

        ind = 0
        for j in basis:
            P[:,ind,ind] = 0
            ind += 1
            for lz in range(1,j+1,1):
               p1 = phase(lz,R,k)
               P[:,ind  ,ind  ] = p1*(-1)**lz /np.sqrt(2)
               P[:,ind  ,ind+1] = p1*(-(-1)**lz) *1j /np.sqrt(2) ### follows from backtransform
               p2 = phase(-lz,R,k)
               P[:,ind+1,ind  ] = p2*  1/np.sqrt(2)
               P[:,ind+1,ind+1] = p2* 1j/np.sqrt(2)
               ind += 2

        #Expectation value. Note: <P|Psi> is calculated
        if k.ndim == 2:
            P_Psi = np.einsum("EijK,Kjk->KEik",P,evecs)
            exp_val = np.einsum("KEii->KEi",np.einsum("KEji,KEjk->KEik",P_Psi.conj(),P_Psi,optimize=True),optimize=True)
        else:
            P_Psi = np.einsum("Eij,jk->Eik",P,evecs)
            exp_val = np.einsum("Eii->Ei",np.einsum("Eji,Kjk->Eik",P_Psi.conj(),P_Psi,optimize=True),optimize=True)
        return np.abs(exp_val)
    def initialize_val(self,nk):
        '''Initializes the val array in which the calculated expecation values are saved.
           Creates format specifier string.
           Checks the dimension of the Matrix, if NxN matrix, expand to 1xNxN.    
        '''
        if self.op.ndim == 2:
           self.op = np.array([self.op])
        self.val = np.zeros((nk,np.shape(self.op)[0],self.ham.n_bands))
        for dim in range(np.shape(self.val)[1]):
            self.f_spec += '{d['+str(dim)+']:'+self.prec+'}'

    def initialize_val_k(self,nk):
        '''Initializes the val array for k-dependent operators in which the calculated expecation values are saved.
           Creates format specifier string.'''
        k = np.array([0,0,0])
        evals = np.zeros(self.ham.n_bands)
        evecs = np.zeros((self.ham.n_bands,self.ham.n_bands))
        self.val = np.zeros((nk,np.shape(self.op(k,evals,evecs))[0],self.ham.n_bands))
        for dim in range(np.shape(self.val)[1]):
            self.f_spec += '{d['+str(dim)+']:'+self.prec+'}'

    ####General post-processing functions

    def no_post(self,evals):
        '''No post-processing...'''
        print("No post-processing.")


    def L_2(self,evals):
        '''Calculates L^2.'''
        self.val[:,3] = (-1+np.sqrt(1+4*self.val[:,3]))/2


    def b_int_ef(self,evals):
        '''Performs band integration above all occupied bands'''
        if self.ham.ef == None:
            print("Fermi-energy is not defined for the Hamiltonian, k-integration over occupied bands is not possible...")
        else:
            occ = np.zeros_like(evals)
            #occ[evals <= self.ham.ef] = 1
            occ[evals <= 0] = 1
            val_int = np.multiply(self.val,occ[:,None])
            self.val_b_int = np.sum(val_int, axis = 2)


    def k_int(self,evals,nbins=1000,sigma=0.1):
        '''Performs k-integration by using a histogram.'''
        self.val_k_int = np.zeros((np.shape(self.val)[1]+2,nbins))
        dos = np.histogram(evals,nbins)
        self.val_k_int[0] = (dos[1][1:]+dos[1][:-1])/2
        self.val_k_int[1] = scipy.ndimage.filters.gaussian_filter1d(dos[0],sigma)/evals.shape[0]
        for dim in range(np.shape(self.val)[1]):
            self.val_k_int[dim+2] = scipy.ndimage.filters.gaussian_filter1d(np.histogram(evals,nbins,weights=self.val[:,dim])[0],sigma)/evals.shape[0]
           


    #### Further post-processing
    def sphere_winding(self,ste,steps):
        '''Calculates the winding number/Pontryagin index for a given vector field calculated on the unit sphere.
           Vector field is expected to be projected onto {R,phi}-plane.
           Vector field is required to be stored in the following order vec[R,phi,3].
           ste (stereographic map) is also required to have the form ste[R,phi,2]
        '''
        print("Calculating Pontryagin-index for operator "+self.op_type+".")
        self.pont = np.zeros(self.ham.n_bands)
        for band in range(self.ham.n_bands):
            vec = self.val[:,:3,band].transpose((1,0))
            ### Normalize the initial vector
            vec = vec/np.linalg.norm(vec,axis=0,keepdims=True)
            ### Reshape the initial vector
            vec = np.reshape(vec,(3,steps,2*steps))
            vec = np.transpose(vec,(1,2,0))


            ### Derivatives
            delta_vec_R   = vec[1:]-vec[:-1]
            delta_vec_phi = vec[:,1:] - vec [:,:-1]

            delta_ste_R   = ste[1:,:,0] - ste[:-1,:,0]
            delta_ste_phi = ste[:,1:,1] - ste[:,:-1,1]


            del_vec_R = (delta_vec_R/delta_ste_R[:,:,None])[:,:-1] # We sample on phi=0 and 2pi, discard 2pi sampling

            del_vec_phi = (delta_vec_phi[1:]/delta_ste_phi[1:,:,None]+delta_vec_phi[0:-1]/delta_ste_phi[:-1,:,None])/2


            #Cross product
            cross = np.zeros_like(del_vec_phi)
            for i in range(3):
                j = (i+1)%3
                k = (i+2)%3
                cross[:,:,i] = del_vec_R[:,:,j]*del_vec_phi[:,:,k]-del_vec_phi[:,:,j]*del_vec_R[:,:,k]

            ### Bring the vector in the new shape
            vec_new = (vec[1:,:-1]+ vec[:-1,:-1])/2


            #Calculate the scalarproduct of the new_vec and the crossproduct
            scal = np.einsum('...i,...i',cross,vec_new)


            #Calculate the infinitesimal area element, and new stereographic map

            dR = ste[1:,:-1,0]-ste[:-1,:-1,0]
            dA =dR

            #Integral
            self.pont[band] = np.mean(np.sum(scal*dA,axis=0))/2
            print("Band {b:3d}: S={i:7.4f}".format(b=(band+1),i=self.pont[band]))

####Testing section
def commutator_L(op):
    '''Calculates the commutation relation for angular momentum operators.'''
    print("Calculating commutator...")
    com = np.matmul(op[0],op[1])-np.matmul(op[1],op[0])-1j*op[2]
    print("Maximum value of cummutator:",np.amax(com))
    if abs(np.amax(com))<=1e-10:
        print("Commutator check passed!!!")
    else:
        print("Commutator check failed!!!")

if __name__== "__main__":
    print("Testing class operator...")
    print("Testing spin-operator for a spin-less system...")
    real_vec = np.array([[3.0730000,    0.0000000,    0.0000000],[-1.5365000,    2.6612960,    0.0000000],[0.0000000,    0.0000000,   20.0000000]])
    basis = np.array([0,1])
    my_ham_spinless = hamiltonian("test_ham/hr_In_soc.dat",real_vec,False,basis)
    s_spinless = operator("S",my_ham_spinless)
    print("Testing spin-operator for a spin-full system...")
    my_ham = hamiltonian("test_ham/hr_In_soc.dat",real_vec,True,basis)
    s_spinfull = operator("S",my_ham)
    print("Instance attributes of the generated spin-operator")
    print(s_spinfull.__dict__)
    print("Spin-operator: commutator test...")
    commutator_L(s_spinfull.op)
    print('Testing function "initialize_val"...')
    s_spinfull.initialize_val(3)
    print("Shape val",np.shape(s_spinfull.val))
    print("Testing L-Operator...")
    L_op = operator("L",my_ham)
    commutator_L(L_op.op)
    print("Testing J-Operator...")
    J_op = operator("J",my_ham)
    commutator_L(J_op.op)
    print("Testing BC-Operator...")
    BC_op = operator("BC",my_ham)
    print('Testing function "initialize_val_k" for k-dependent operators...')
    BC_op.initialize_val_k(3)
    print(np.shape(BC_op.val))
    print('Testing a self-defined operator with one component input...')
    self_def_op = operator("self_def",my_ham)
    print('Define operator matrix by hand.')
    self_def_op.op = np.eye(my_ham.n_bands)
    self_def_op.initialize_val(5)
    print(np.shape(self_def_op.op))

    print('Testing a self-defined operator with multidimensional input...')
    self_def_op = operator("self_def",my_ham)
    print('Define operator matrix by hand.')
    self_def_op.op = np.array([np.eye(my_ham.n_bands),np.eye(my_ham.n_bands)])
    self_def_op.initialize_val(5)
    print(np.shape(self_def_op.op))

    print('Testing operator "E_triang"...')
    E_triang_op =  operator("E_triang",my_ham) 
