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
* time

## Usage
### hamiltonian class
#### Reading-in/Creating Hamiltonian objects
#### Functions
#### Extension to general tight-binding Models

### k_space class
#### Available k-parametrizations
#### Generating a k-space

### observables class
#### Initializing a calculation
#### User-defined operators
#### Calculation of expectation values
#### Post-processing
#### Pontryagin index calculation

### wcc class
#### Calculation of the Wannier Charge Center movement
