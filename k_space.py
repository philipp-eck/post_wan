#! /bin/python
### k_space class, generates k-paths/grids for the calculations.
import numpy as np

class k_space:
    '''k_space class, generates k-paths/grids for the calculations.
       We use reduced coordinates for the Fourier-transform of H(R)
       and cartesian vectors for the output files.
       Instance attributes:
       k_type        # Defines the called function for generating the k-space {path,plane,self-defined,monkhorst,sphere,sphere_ster_proj}
       k_basis       # Defines the basis, cartesian or reduced coordinates
       vecs          # Vectors spanning the path/mesh, has to be a 2-dim matrix
       bra_vec       # Bravais vectors of the corresponding crystal structure/Hamiltonian
       n_points      # Number of points used for the segments
       radius        # Radius for spheres/circles
       vecs_red      # vecs in reduced coordinates
       vecs_car      # vecs in cartesian coordinates
       k_space_red   # Generated k_space in reduced coordinates
       k_space_car   # Generated k_space in cartesian coordinates
       k_space_proj  # Projected k_space in cartesian coordinates e.g. for stereographic projection
       k_dist        # Distance in k-space
       k_kind        # Defines the output {mesh/path}
    '''


    def __init__(self,K_TYPE,K_BASIS,VECS,BRA_VEC=None,N_POINTS=None,RADIUS=None):
        '''Initializes the k_space class'''

        self.k_type   = K_TYPE        # Defines the kind of mesh/path
        self.k_basis  = K_BASIS       # Defines the basis, cartesian or reduced coordinates
        self.vecs    = VECS          # Vectors spanning the path/mesh, has to be a 2-dim matrix
        self.bra_vec = BRA_VEC       # Bravais vectors of the corresponding crystal structure/Hamiltonian
        self.n_points = N_POINTS      # Number of points used for the segments
        self.radius  = RADIUS

        if np.shape(self.bra_vec)!=(3,3):
            print("bra_vec has not the shape=(3,3)!!!")
        else:
            if self.k_basis == "red":
                self.vecs_red = VECS
                self.vecs_car = self.red_to_car(self.vecs)
            elif self.k_basis == "car":
                self.vecs_red = self.car_to_red(self.vecs)
                self.vecs_car = VECS
            else:
                print('Unknown basis representation, set kbasis ="car"/"red".')

        if self.k_type == "path":
            self.k_space_red = self.Path(self.vecs_red)
            self.k_space_car = self.Path(self.vecs_car)
            self.set_k_dist()
            self.k_kind = "path"

        if self.k_type == "plane":
            self.k_space_red = self.Plane(self.vecs_red)
            self.k_space_car = self.Plane(self.vecs_car)
            self.k_kind = "mesh"

        if self.k_type == "sphere_ster_proj":
            self.k_space_car, self.k_space_proj = self.sphere_ster_proj(self.vecs_car)
            self.k_space_red = self.car_to_red(self.k_space_car)
            self.k_kind = "mesh"

        if self.k_type == "sphere_ster_proj_plot":
            self.k_space_red, self.k_space_car = self.sphere_ster_proj_plot(self.vecs_car)
            self.k_space_red = self.car_to_red(self.k_space_red)
            self.k_kind = "mesh"

        if self.k_type == "sphere":
            self.k_space_red, self.k_space_car = self.Sphere(self.vecs_car)
            self.k_space_red = self.car_to_red(self.k_space_red)
            self.k_kind = "mesh"

        if self.k_type == "monkhorst":
            print('Setting k_basis="red".')
            self.k_basis = "red"
            self.k_space_red = self.Monkhorst(self.vecs)
            if self.bra_vec is not None:
               self.k_space_car = self.red_to_car(self.k_space_red)
            self.k_kind = "mesh"
            if type(self.bra_vec) == np.ndarray:
               self.k_space_car = self.red_to_car(self.k_space_red)

        if self.k_type == "self-defined":
            self.k_space_red = self.vecs_red
            self.k_space_car = self.vecs_car
            self.set_k_dist()
            self.k_kind = "path"


    def car_to_red(self,car):
        '''Transforms from cartesian into reduced coordinates'''
        red = np.einsum("ij,kj->ki",self.bra_vec,car)/(2*np.pi)
        return red

    def red_to_car(self,red):
        '''Transforms from reduced into cartesian coordinates'''
        car = np.einsum("ij,kj->ki",np.linalg.inv(self.bra_vec),red)*2*np.pi
        return car


    def set_k_dist(self):
        '''Calculates the k-distance along a given connected path.
           P.E: Could be improved by doing this segment-wise => Non-connected paths...'''
        k_dist = np.zeros(np.shape(self.k_space_car)[0])
        k_prev  = self.k_space_car[0]
        for i_k in range(np.shape(self.k_space_car)[0]):
            k_dist[i_k] = k_dist[i_k-1] + np.linalg.norm((self.k_space_car[i_k]-k_prev))
            k_prev = self.k_space_car[i_k]
        self.k_dist = k_dist

    def Path(self,vecs):
        '''Creates a connected path, defined by the vectors in vec'''

        path = np.zeros((self.n_points*(len(vecs)-1),3))
        for vec in range(len(self.vecs)-1):
            for i in range(3):
                path[vec*self.n_points:(vec+1)*self.n_points,i] = np.linspace(vecs[vec,i],vecs[vec+1,i],self.n_points)
        return path   

    def Plane(self,vecs):
        '''Creates a plane CONTAINING THE MARGINS OF THE PLANE.
           DON'T USE FOR BZ SAMPLING!!!
           First vector: Center
           Second and third vector: Vectors spanning the plane
           The first two vectors define a path, which is shifted by the third one, to generate the grid.'''
        if np.shape(self.vecs)!=(3,3):
            print("Input vectors have the wrong shape!!!")
        plane = np.zeros((self.n_points**2,3))
        for i in range(self.n_points):
            vec_0 = vecs[0] + vecs[1]*(i/(self.n_points-1)-0.5)
            for j in range(3):
                plane[i*self.n_points:(i+1)*self.n_points,j] = np.linspace(vec_0[j]-0.5*vecs[2,j],vec_0[j]+0.5*vecs[2,j],self.n_points)
        return plane


    def Monkhorst(self,vecs):
        '''Creates a gamma centered Monkhorst grid.
           The input vector specifies the number of sampling points in each direction.'''
        print("Creating a Gamma-centered Monkhorst grid with {v[0]:d}x{v[1]:d}x{v[2]:d} samples...".format(v=vecs[0]))
        out = np.zeros((vecs[0,0],vecs[0,1],vecs[0,2],3))
        out[:,:,:,0] = np.linspace(0,1,vecs[0,0],endpoint=False)[:,None,None]
        out[:,:,:,1] = np.linspace(0,1,vecs[0,1],endpoint=False)[None,:,None]
        out[:,:,:,2] = np.linspace(0,1,vecs[0,2],endpoint=False)[None,None]
        out = out.flatten().reshape((-1,3))
        return out

    def sphere_ster_proj(self,k):
        '''Creates a grid for a stereographic projection.
           First vector, center of the sphere.
           Second vector defines north pole.
           Third vector defines starting point of azimuthal angle. 
        '''
        if np.shape(k)==(1,3):
            v3 = np.array([0,0,1])
            v1 = np.array([1,0,0])
            print("North pole of the sphere points towards k_z-direction.")
        elif np.shape(k)==(3,3) and np.linalg.norm(k[1])==1.0 and np.linalg.norm(k[2])==1.0:
            v3 = k[1]
            v1 = k[2]
            print("North pole oriented along "+str(v3))
            print("Azimuthal vector starts in direction "+str(v1))
        else:
            print("Input vector has the wrong shape. Please specify the center of the sphere or give additionally north pole and starting point of azimuthal vector")

        if self.radius == None:
            print("Radius is not defined...")
    
        def ster_proj(theta,phi):
            '''Stereographic projection of the unit sphere.
               Transforms {theta,phi}->{R,phi}'''
            if theta==0:
                theta = 0.000001 # small epsilon, to map north-pole not to ifty
            R_phi = np.array([np.sin(theta)/(1-np.cos(theta)),phi])
            return R_phi
    
        ste  = np.zeros((self.n_points,2*self.n_points,2))
        mesh = np.zeros((2*self.n_points**2,3))
        v3 = np.array([0,0,1])
        v1 = np.array([1,0,0])
        v2 = np.cross(v3,v1)
        for i in range(self.n_points):
            for j in range(2*self.n_points):
                phi = 2*np.pi/(2*self.n_points-1)*j     #sample on 0 and 2pi
                theta = 1*np.pi/(self.n_points-1)*(i) #avoid north-pole which is projected to R->infty
               #mesh_old[i*self.n_points+j,0] = center[0,0]+self.radius*np.cos(phi)*np.sin(theta)
               #mesh_old[i*self.n_points+j,1] = center[0,1]+self.radius*np.sin(phi)*np.sin(theta)
               #mesh_old[i*self.n_points+j,2] = center[0,2]+self.radius*np.cos(theta)
                mesh[2*i*self.n_points+j] = k[0] + self.radius*(np.cos(theta)*v3+ np.sin(theta)*(np.cos(phi)*v1 + np.sin(phi)*v2))
                ste[i,j] = ster_proj(theta,phi)
        return mesh,ste


    def sphere_ster_proj_plot(self,vec):
        ''' For plotting of the ster-proj. This function defines a grid, with constant point density
            in the stereographic projection plane.
            The number per circle scales with 8.
        '''
        path = []
        ste = []
        point = np.array([0,0,0])
        ste.extend([point])
        pp    = vec[0] -np.array([0,0,1])*self.radius
        path.extend([pp])
        for ri in range(1,self.n_points):
            R = ri/(self.n_points-1)
            #theta = ster_proj_back(R**2/((1+R**2)**2))
            #theta = ster_proj_back(R)
            theta = np.pi*(1-1/(self.n_points-1)*ri)
            for phii in range(ri*8):
                phi = 2*np.pi/(ri*8)*phii
                point = np.array([np.cos(phi),np.sin(phi),0])*R
                ste.extend([point])
                pp = np.zeros(3)
                pp[0] = vec[0,0]+self.radius*np.cos(phi)*np.sin(theta)
                pp[1] = vec[0,1]+self.radius*np.sin(phi)*np.sin(theta)
                pp[2] = vec[0,2]+self.radius*np.cos(theta)
                path.extend([pp])
        path = np.array(path)
        ste  = np.array(ste)
        return path,ste

    def Sphere(self,vec):
        ''' Creates path on the surface of the sphere
            and the vector normal to the surface of the sphere'''
        path = []
        norm = []
        for thetai in range(self.n_points):
            theta = 1.0 * np.pi/(self.n_points-1)*thetai
            stepsphi = 2*round(self.n_points*np.sin(theta)**2)+1
            stepsphi = int(stepsphi)
            for phii in range(stepsphi):
                point = np.zeros(3)
                phi = 2.0 * np.pi/stepsphi*phii
                point[0] = vec[0,0]+self.radius*np.cos(phi)*np.sin(theta)
                point[1] = vec[0,1]+self.radius*np.sin(phi)*np.sin(theta)
                point[2] = vec[0,2]+self.radius*np.cos(theta)
                path.extend([point])
                normpoint = np.zeros(3)
                normpoint[0] = np.cos(phi)*np.sin(theta)
                normpoint[1] = np.sin(phi)*np.sin(theta)
                normpoint[2] = np.cos(theta)
                norm.extend([normpoint])
        path = np.array(path)
        norm = np.array(norm)
        return path,norm



if __name__== "__main__":
    print("Testing class k_space...")

    print("Checking the transform from reduced coodinates to cartesian coordinates...")
    points = 5
    real_vec = np.array([[3.0730000,    0.0000000,    0.0000000],[-1.5365000,    2.6612960,    0.0000000],[0.0000000,    0.0000000,   20.0000000]])
    vecs=np.array([[2/3,-1/3,0],[-2/3,1/3,0]]) # vecs has to be a 2-dim matrix
    print("Initial reduced vector:")
    print(vecs)
    red_latt = k_space("foo","red",vecs,real_vec,points)
    print("Vector in cartesian coordinates:")
    print(red_latt.vecs_car)

    print("Checking the back transform from cartesian coodinates to reduced coordinates...")
    car_latt = k_space("foo","car",red_latt.vecs_car,real_vec,points)
    print("Back-transformed reduced vector:")
    print(car_latt.vecs_red)
    if np.array_equal(car_latt.vecs_red,vecs):
       print("Transformation between reduced and cartesian coordinates works correctly!!!")
    else:
       print("Transformation between reduced and cartesian coordinates FAILES!!!")

    print("Checking function create_kPath on "+str(points)+" points:")
    path = k_space("path","red",vecs,real_vec,points)
    print("Created path in reduced space:")
    print(path.k_space_red)
    print("Created path in cartesian space:")
    print(path.k_space_car)
    print("k-distance",path.k_dist)


    print("Checking function create_kPlane on "+str(points)+"x"+str(points)+" grid:")
    vecs = np.array([[0,0,0],[1,0,0],[0,1,0]])
    print(np.shape(vecs))


    print("Plane defining vectors in cartesian coordinates:")
    print(vecs)
    plane = k_space("plane","car",vecs,real_vec,points)
    print("Created path in reduced space:")
    print(plane.k_space_red)
    print("Created path in cartesian space:")
    print(plane.k_space_car)
    print('Testing function "sphere_ster_proj" ...')
    radius = 0.030
    center = np.array([[ 0.50930216,   +0.03717546,   -0.31628638],[0,0,1],[1,0,0]])
    spheresterproj = k_space("sphere_ster_proj","car",center,real_vec,points,radius)
    print(spheresterproj.__dict__.keys())
    print('Testing function "sphere_ster_proj_plot" ...')
    spheresterprojplot = k_space("sphere_ster_proj_plot","car",center,real_vec,points,radius) 
    print('Testing function "Sphere" ...')
    sphere = k_space("sphere","car",center,real_vec,points,radius)
    print('Testing function "Monkhorst" ...')
    vecs = np.array([[2,3,4]])
    monkhorst = k_space("monkhorst","red",vecs,real_vec,points)
