# post_wan
This python3 library provides classes for tight-binding calculations with Wannier Hamiltonians, but can be easily applied to general user-defined tight-binding Hamiltonians. The object oriented usage allows for maximal flexibility.

## Capabilities
* Calculation of the following expectation values:
    + Polarization calculation of operators _L,S,J_
    + Expectation value calculation of self-defined k-independent operators 
    + Berry curvature and orbital moment of Bloch state
    + Band inversion character
* Winding number/Pontryagin index of 3D expectation values
* Wannier charge center movement
* _H(k)_ Fouriertransformation for w2dynamics input
* ...


## Required python modules
* numpy
* scipy
* time
* optional: joblib, multiprocessing


## Usage
Please see the examples, a short overview is given below.
### hamiltonian class
Allows to import VASP/WANNIER90 created Hamiltonians, provides $H(K)$ and $\nabla H(K)$.
<!---
#### Reading-in/Creating Hamiltonian objects
#### Functions
#### Extension to general tight-binding Models
-->
### k_space class
Allows to define paths and grids in momentum space. Converts between reduced and cartesian coordinates.
<!---
#### Available k-parametrizations
#### Generating a k-space
-->

### observables class
Main class for calculating k-dependent (BC, BC_OAM) and k-independent (_L,S,J_) observables. Allows also the usage of user-defined operators.
<!---
#### Initializing a calculation
#### User-defined operators
#### Calculation of expectation values
#### Post-processing
#### Pontryagin index calculation
-->

### wcc class
Calculation of the $\mathbb{Z}_2$-invariant from the Wannier Charge Center (WCC) movement. Follows the approach of A. Soluyanov and D. Vanderbilt (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.83.235401).

## Contact
philipp.eck@physik.uni-wuerzburg.de
