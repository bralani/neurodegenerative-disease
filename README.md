## Fisher-Kolmogorov Equation for Neurodegenerative Diseases

The Fisher-Kolmogorov equation describes the spread of misfolded proteins in the brain, which is associated with neurodegenerative diseases. The equation is given by:

$$
\frac{\partial c}{\partial t} - \nabla \cdot (D \nabla c) - \alpha c (1 - c) = 0 \quad \text{in } \Omega,
$$

with the following boundary and initial conditions:

$$
D \nabla c \cdot \mathbf{n} = 0 \quad \text{on } \partial \Omega,
$$

$$
c(t=0) = c_0 \quad \text{in } \Omega.
$$

### Problem Description
The goal of this project is to implement a numerical solver for the Fisher-Kolmogorov equation and validate the results by comparing them with those found in the literature.

### Parameters
- **$c(t, \mathbf{x})$**: Concentration of misfolded proteins at time $t$ and position $\mathbf{x}$.
- **$D$**: Diffusion coefficient (can be constant or spatially varying).
- **$\alpha$**: Reaction rate controlling the growth term $\alpha c (1 - c)$.
- **$\Omega$**: Computational domain.
- **$\partial \Omega$**: Boundary of the domain.
- **$\mathbf{n}$**: Outward normal to the boundary.
- **$c_0$**: Initial condition for $c$ at $t = 0$.

### Diffusion Tensor
The diffusion tensor $D$ is computed as follows:

$$
D = d_{ext} \mathbf{I} + d_{axn} \mathbf{n} \otimes \mathbf{n},
$$
Where
- **$d_{ext}$** is the scalar value related to the anisotropic diffusion
- **$d_{axn}$** is the scalar value related to the isotropic diffusion
- **$I$** is the identity matrix
- **$n$** is the vector value assumed by the fiber field in a particular point of the domain 

### Domain
The computational domain $\Omega$ can be:
- **Idealized** (e.g., a simple 2D or 3D shape like a square or sphere).
- **Realistic** (based on actual brain structures, as suggested in reference [13]).

A sample realistic mesh can be downloaded at the following link:  
[Mesh file (STL format)](https://polimi365-my.sharepoint.com/:u:/g/personal/10461512_polimi_it/EY9ZQp27JArvbXLRPljhNCB-wJ5tPZLlCf0_409EYbtg?e=ClaIRH). The mesh requires preprocessing using **Gmsh**.

## Implementation details
### Meshes
For our experiments, we employed both a cubic domain and a more realistic brain domain. We downloaded the recommended brain SDL model and processed it using Blender and Gmsh to scale it to a more compact size, center its mass at the origin of the mesh, and convert it into a volumetric mesh.
The resulting brain mesh is available [here](https://drive.google.com/drive/folders/1TTNpxtxJowC4FB890qo1zXkJ7gQu1PMJ?usp=sharing).


### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
### Executing
The executable will be created into `build`, and can be executed through
```bash
$ ./ngd
```
The solver is parallelized with MPI, in order to launch a parallelized version, make sure to use the following command
```bash
$ mpirun [-n PROCESS_NUM] ./ngd
```
Where PROCESS_NUM  (if specified) denotes the number of processes that MPI will create. It is generally recommended to not exceed the number of cores of the machine. 

#### Configuration file
The solver will work without any configuration with standard parameters set in the code. If you want to set custom parameters for the solver, you need to create a config file placed in the same directory as the executable named
```
config.txt
```
The config file is a plain text file with the following rules:
- **Comments:**  
  Lines starting with `#` are considered comments and will be ignored.

- **Empty Lines:**  
  Empty lines are allowed and are skipped during parsing.

- **Parameters:**  
  Each parameter is set on a separate line in the format:
```scss
parameter_name value(s)
```
The parameter name and its corresponding values are separated by spaces.
If a parameter is left out of the configuration file, the default value will be used.
Below is an example configuration file:

```plaintext
# comments starts with '#'
dext 10.2

# empty lines are allowed

dt 1.2
mass-center 2 3 4

# diffusion CYLINDRICAL
diffusion RADIAL
```

The following are all the available parameters and their descriptions

- **dext:**  
  The `dext` parameter represents the anistropic diffusion coefficient.  
  Example: `dext 0.1`

- **daxn:**  
  The `daxn` parameter represents the isotropic diffusion coefficient.  
  Example: `daxn 0.2`

- **T:**  
  The `T` parameter represents the total time for the simulation.  
  Example: `T 11.1`

- **dt:**  
  The `dt` parameter is the time step size.  
  Example: `dt 1.2`

- **alpha:**  
  The `alpha` parameter is the growth factor.
  Example: `alpha 1.55`

- **r:**  
  The `r` parameter is an integer that represent the degree for the finite elements basis functions. 
  Example: `r 1`

- **mass-center:**  
  The `mass-center` parameter defines a point where the initial concentration of protein is centered.  
  The following numbers are the coordinates. 
  Example (3D): `mass-center 2 3 4`

- **origin:**  
  Used in the cylindrical diffusion tensor model. The `origin` parameter defines a 2D point on the YZ plane where the axis of the cylinder meets the plane. 
  Example: `origin 1 2`

- **radial-center:**  
  Used in the radial diffusion tensor model. The `radial-center` parameter defines a point that represents the center for radial diffusion.  
  Example: `radial-center 4 2 1`

- **diffusion:**  
  The `diffusion` parameter specifies the type of diffusion to be used. Currently the options are `RADIAL` and `CYLINDRICAL`  
  Example: `diffusion RADIAL`

Upon launching the application, a diagnostic message regarding the configuration file may be displayed. If no errors occur, all parameters are printed on the terminal for user verification.
