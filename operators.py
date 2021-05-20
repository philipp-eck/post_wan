#! /bin/python
### operator class, sub-class of observables. 
import numpy as np
import scipy.stats
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
        V             # Volume element for k-integration
    '''
       

    def __init__(self,OP_TYPE,HAMILTONIAN=None):
        '''Initializes the operator object'''

        self.op_type = OP_TYPE
        self.op      = None
        self.val     = None
        self.val_b_int = None
        self.val_k_int = None
        self.val_kE_int= None
        self.f_spec  = ''
        self.prec    = "9.4f"
        self.post    = self.no_post
        self.ham     = HAMILTONIAN
        self.expval  = None
        self.V      = 1
        if self.op_type == "S":
            self.set_S_op()
        elif self.op_type == "L":
            self.set_L_op()
        elif self.op_type == "J":
            self.set_J_op()
        elif self.op_type == "BC":
            self.set_BC_op()
        elif self.op_type == "BC_S":
            self.set_BC_spin_op()
            self.BC_type = "S"
        elif self.op_type == "BC_L":
            self.set_BC_oam_op()
            self.BC_type = "L"
        elif self.op_type == "BC_J":
            self.set_BC_J_op()
            self.BC_type = "J"
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
       #self.post = self.b_int_ef
        self.post = self.b_int_n_elec
        self.V   = None
    def set_BC_spin_op(self):
        '''Sets spin-BC-operator, requires nabla_k H(k).'''
        self.prec    = "12.3e"
        self.op = self.BC_op_op
        self.expval = self.BC_op_expval
        self.expval_all_k = self.BC_op_expval
       #self.post = self.b_int_ef
        self.post = self.b_int_n_elec
        self.V   = None

    def set_BC_oam_op(self):
        '''Sets OAM-BC-operator, requires nabla_k H(k) and a defined basis.'''
        self.prec    = "12.3e"
        self.op = self.BC_op_op
        self.expval = self.BC_op_expval
        self.expval_all_k = self.BC_op_expval
       #self.post = self.b_int_ef
        self.post = self.b_int_n_elec
        self.V   = None

    def set_BC_J_op(self):
        '''Sets J-BC-operator, requires nabla_k H(k) and a defined basis.'''
        self.prec    = "12.3e"
        self.op = self.BC_op_op
        self.expval = self.BC_op_expval
        self.expval_all_k = self.BC_op_expval
       #self.post = self.b_int_ef
        self.post = self.b_int_n_elec
        self.V   = None

    def set_BC_mag_op(self):
        '''Sets BC_mag-operator, requires nabla_k H(k).'''
        self.prec    = "12.3e"
        self.op = self.BC_mag_op
        self.expval = self.BC_mag_expval
        self.expval_all_k = self.BC_mag_expval
        self.post = self.b_int_ef
        self.V   = None

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
           #L = np.kron(one,L)
            L = np.kron(np.eye(2),L)
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
        '''
        BC_op = np.zeros((3,self.ham.n_bands))
        return BC_op

###    Not needed anymore, replaced by BC_op_op
###    def BC_spin_op(self,k=None,evals=None,evecs=None):
###        '''Generates an empty array.
###           Spin-BC expectation value is calculated with a more efficient function.
###           Components  1- 4: Bx*s
###           Components  5- 8: By*s
###           Components  9-12: Bz*s
###        '''
###        BC_spin_op = np.zeros((12,self.ham.n_bands))
###        return BC_spin_op

    def BC_op_op(self,k=None,evals=None,evecs=None):
        '''Generates an empty array.
           OP-BC expectation value is calculated with a more efficient function.
           Components  1- 4: Bx*OP
           Components  5- 8: By*OP
           Components  9-12: Bz*OP
        '''
        BC_op_op = np.zeros((12,self.ham.n_bands))
        return BC_op_op

    def BC_mag_op(self,k=None,evals=None,evecs=None):
        '''Generates an empty array.
           Calculates the orbital moment of the Bloch state
        '''
        BC_mag_op = np.zeros((3,self.ham.n_bands))
        return BC_mag_op


##### Modern theory of polarization properties/functions #####

    def calc_Em_En(self,k,evals):
        '''Computes the denominator matrix (E_m-E_n) and sets diagonal to 1.
        '''
        Em_En  = np.reshape(np.kron(np.ones(self.ham.n_bands),evals),(np.shape(k)[0],self.ham.n_bands,self.ham.n_bands))
        Em_En -= np.transpose(Em_En,(0,2,1))
        #[np.fill_diagonal(Em_En[i],np.Inf) for i in range(k.shape[0])]
        [np.fill_diagonal(Em_En[i],0.0001) for i in range(k.shape[0])]
        return Em_En

    def calc_vel(self,k,evecs,kind="standard"):
        '''Calculates the velocity operator matrixm <m|nabla_k H_k|n>.
           If spin=True, calculates the anti-commutator v=1/2{sigma,v}.
           See also: https://arxiv.org/pdf/1901.05651.pdf
        '''
        if kind == "standard":
            m_del_n = np.einsum('kdb,kcde,kef->kcbf',evecs.conj(),self.ham.del_hk(k),evecs,optimize=True)
#           diag = np.eye(self.ham.n_bands)[None,None]

        else:
            if kind == "S":
                # Multiply by two, since we want the commutator {sigma,v}
                op = self.S_op()*2
            elif kind == "L":
                op = self.L_op()
            elif kind == "J":
                op = self.J_op()
            op = np.roll(op,1,axis=0)
            op[0] = np.eye(op[0].shape[0]) # op[0]=1/2 identity matrix
            #### Calculate the anti-commutator
            del_hk = self.ham.del_hk(k)
            v = 0.5*(np.einsum('slm,kcmn->kcsln',op,del_hk) + np.einsum('kclm,smn->kcsln',del_hk,op))
            m_del_n = np.einsum('kdb,kcsde,kef->kcsbf',evecs.conj(),v,evecs,optimize=True)
#           diag = np.eye(self.ham.n_bands)[None,None,None]


        #### Cheaty way to set diag of m_del_n to 0
        #don't use this!!!
#       non_diag = np.ones_like(m_del_n)-diag
#       m_del_n *=non_diag
        return m_del_n

           

    def BC_expval(self,k=None,evals=None,evecs=None):
        ''' Calculates Berry curvature:
                <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
        m = -Im -----------------------------------
                            (E_m - E_n)^2
        Note: the expectation value <m|nabla_k H_k|n>/(E_m - E_n) are written as line vectors in m_del_n.
        use np.roll to generate the permutation.
        '''

        Em_En    = self.calc_Em_En(k,evals)
        m_del_n  = self.calc_vel(k,evecs)
        m_del_n /= Em_En[:,None]
        BC =-2*np.imag(np.einsum('kdji,kdji->kdi',np.roll(np.conj(m_del_n),-1,axis=1),np.roll(m_del_n,-2,axis=1),optimize=True))
        return BC

### Not needed anymore, can be deleted if function BC_op_expval is sufficiently tested
###    def BC_spin_expval(self,k=None,evals=None,evecs=None):
###        ''' Calculates Berry curvature:
###                <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
###        m = -Im -----------------------------------
###                            (E_m - E_n)^2
###        Note: the expectation value <m|nabla_k H_k|n>/(E_m - E_n) are written as line vectors in m_del_n.
###        use np.roll to generate the permutation.
###        '''
###
###        Em_En    = self.calc_Em_En(k,evals)
###        m_del_n  = self.calc_vel(k,evecs,"spin")
###        m_del_n /= Em_En[:,None,None]
###        m_del_n *= (1-np.eye(self.ham.n_bands))[None,None,None]
####       BC_spin =-2*np.imag(np.einsum('kcsji,kcji->kcsi',np.roll(np.conj(m_del_n),-1,axis=1),np.roll(m_del_n[:,:,0],-2,axis=1),optimize=True))
###        BC_spin =-1*np.imag(np.einsum('kcsmn,kcmn->kcsn',np.roll(np.conj(m_del_n),-1,axis=1),np.roll(m_del_n[:,:,0],-2,axis=1),optimize=True)
###                           -np.einsum('kcmn,kcsmn->kcsn',np.roll(m_del_n[:,:,0],-1,axis=1),np.roll(np.conj(m_del_n),-2,axis=1),optimize=True))
###        BC_spin = BC_spin.reshape((k.shape[0],12,self.ham.n_bands))
###        return BC_spin
###
###    def BC_oam_expval(self,k=None,evals=None,evecs=None):
###        ''' Calculates Berry curvature:
###                <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
###        m = -Im -----------------------------------
###                            (E_m - E_n)^2
###        Note: the expectation value <m|nabla_k H_k|n>/(E_m - E_n) are written as line vectors in m_del_n.
###        use np.roll to generate the permutation.
###        '''
###
###        Em_En    = self.calc_Em_En(k,evals)
###        m_del_n  = self.calc_vel(k,evecs,"spin")
###        m_del_n /= Em_En[:,None,None]
###        m_del_n *= (1-np.eye(self.ham.n_bands))[None,None,None]
###        BC_oam =-1*np.imag(np.einsum('kcsmn,kcmn->kcsn',np.roll(np.conj(m_del_n),-1,axis=1),np.roll(m_del_n[:,:,0],-2,axis=1),optimize=True)
###                           -np.einsum('kcmn,kcsmn->kcsn',np.roll(m_del_n[:,:,0],-1,axis=1),np.roll(np.conj(m_del_n),-2,axis=1),optimize=True))
###        BC_oam = BC_oam.reshape((k.shape[0],12,self.ham.n_bands))
###        return BC_oam

    def BC_op_expval(self,k=None,evals=None,evecs=None):
        ''' General function for calculating the projected BC with velocity operator V = {v,op}.
        Calculates Berry curvature:
                <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
        m = -Im -----------------------------------
                            (E_m - E_n)^2
        Note: the expectation value <m|nabla_k H_k|n>/(E_m - E_n) are written as line vectors in m_del_n.
        use np.roll to generate the permutation.
        '''

        Em_En    = self.calc_Em_En(k,evals)
        m_del_n  = self.calc_vel(k,evecs,self.BC_type)
        m_del_n /= Em_En[:,None,None]
        m_del_n *= (1-np.eye(self.ham.n_bands))[None,None,None]
        BC_op =-1*np.imag(np.einsum('kcsmn,kcmn->kcsn',np.roll(np.conj(m_del_n),-1,axis=1),np.roll(m_del_n[:,:,0],-2,axis=1),optimize=True)
                           -np.einsum('kcmn,kcsmn->kcsn',np.roll(m_del_n[:,:,0],-1,axis=1),np.roll(np.conj(m_del_n),-2,axis=1),optimize=True))
        BC_op = BC_op.reshape((k.shape[0],12,self.ham.n_bands))
        return BC_op

    def BC_mag_expval(self,k=None,evals=None,evecs=None):
        ''' Calculates orbital moment of the Bloch state:
                 <n|nabla_k H_k|m>x<m|nabla_k H_k|n>
        l_z= +Im -----------------------------------
                             E_m - E_n
         Note: global minus sign arising from magnetic moment of the electron.
        '''

        Em_En    = self.calc_Em_En(k,evals)
        m_del_n  = self.calc_vel(k,evecs)
        m_del_n /= Em_En[:,None]
        orb_mom =+2*np.imag(np.einsum('kdji,kdji->kdi',np.roll(np.conj(m_del_n),-1,axis=1),np.roll(m_del_n*Em_En[:,None],-2,axis=1),optimize=True))
        return orb_mom
 
    def Orb_SOC_inv_op(self,k=None,evals=None,evecs=None):
        '''Generates an empty array.
           orb_soc_inv expectation value is calculated with a more efficient function.
           When called, initialized the creation of a spin-less Hamiltonian'''
        orb_soc_inv_op = np.zeros((3,self.ham.n_bands))
        self.ham.set_hr_spinless()
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

    def b_int_n_elec(self,evals):
        '''Performs band integration above the lowest n-elec bands'''
        if self.ham.n_elec == None:
            print("Number of electrons is not defined for the Hamiltonian, k-integration over the lowest n-elec bands is not possible...")
        else:
            occ = np.zeros_like(evals)
            occ[:,:self.ham.n_elec] = 1
            val_int = np.multiply(self.val,occ[:,None])
            self.val_b_int = np.sum(val_int, axis = 2)


    def k_int(self,evals,sigma=0.05,wstep=0.001):
        '''Performs k-integration by using a histogram.
           Integrates over the energy axis in the second step.
        '''
        ### BZ Volume element by calculating the triple product
        if self.V == None:
            ### Actually 2pi³, however the factor 2pi is not considered in del_hk
            self.V = (2*np.pi)**1/np.abs(np.dot(np.cross(self.ham.bra_vec[0],self.ham.bra_vec[1]),self.ham.bra_vec[2]))

        ### taken from w2dyn
       # sigma = .05 #* discr  # on average, take smoothen over 2 energy levels
        wborder = 3*sigma
       # wstep = .001 # put into small bins, and let the filter work its magic
        wmin = np.around(np.amin(evals) - wborder, 3)
        wmax = np.around(np.amax(evals) + wborder, 3) + 1e-4
        w = np.arange(wmin, wmax, wstep)
        self.val_k_int = np.zeros((np.shape(self.val)[1]+2,len(w)-1))
        dos = np.histogram(evals,w)
        self.val_k_int[0] = (dos[1][1:]+dos[1][:-1])/2
        self.val_k_int[1] = scipy.ndimage.filters.gaussian_filter1d(dos[0],sigma/wstep,truncate=40)/evals.shape[0]/wstep
#       print("Integrated total DOS:",scipy.integrate.simps(self.val_k_int[1],self.val_k_int[0]))
#       print("Summed total DOS:", np.sum(dos[0])/evals.shape[0])
        self.val_kE_int = np.zeros_like(self.val_k_int)
        self.val_kE_int[0] = self.val_k_int[0]
       #self.val_kE_int[1] = scipy.stats.rv_histogram(dos).cdf(self.val_kE_int[0])          
       #self.val_kE_int[1,:-1] = scipy.integrate.cumtrapz(self.val_k_int[1],self.val_k_int[0])
        self.val_kE_int[1] = np.cumsum(dos[0])/evals.shape[0]
        for dim in range(np.shape(self.val)[1]):
            dos_i = np.histogram(evals,w,weights=self.val[:,dim])
            self.val_k_int[dim+2]  = scipy.ndimage.filters.gaussian_filter1d(dos_i[0],sigma/wstep)/evals.shape[0]/wstep*self.V
           #self.val_kE_int[dim+2] = scipy.stats.rv_histogram(dos_i).cdf(self.val_kE_int[0])
           #self.val_kE_int[dim+2,:-1] = scipy.integrate.cumtrapz(self.val_k_int[dim+2],self.val_kE_int[0])
            self.val_kE_int[dim+2] = np.cumsum(dos_i[0])/evals.shape[0]*self.V
       #self.val_kE_int[:,-1] = self.val_kE_int[:,-2]
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
