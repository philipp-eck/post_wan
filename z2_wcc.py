#! /bin/python
### WCC pumping class, to calculate the Z_2 index 
import numpy as np
import time
import matplotlib.pyplot as plt


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
            gap = (np.roll(self.wcc[k3],-1,axis=1)+1E-5*0-self.wcc[k3])
            gap[:,-1] +=1 #Safer choice then using the modulo operator
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

    def write_wcc(self,PREFIX=""):
        '''Writes the wannier charge centers and the largest gap to an output file'''
        print("Writing WCC output...")
        fss = "{k:8.4f}{g:8.4f}"
        for n in range(self.ham.n_elec):
            fss +="{wcc["+str(n)+"]:8.4f}"
        fss += " \n"
        for k3 in range(2):
            if k3 == 0:
                output = open(PREFIX+"WCC_gam.dat","w")
            else:
                output = open(PREFIX+"WCC_sur.dat","w")
            for k in range(self.n_pump[0]):
                output.write(fss.format(k=0.5*k/(self.n_pump[0]-1),g=self.gap[k3,k],wcc=self.wcc[k3,k]))
            output.close()
        print("Finished writing WCC output.")

