#include <iostream>

#include "DiffusionTensor.hpp"
#include "FisherKolmogorov.hpp"

#define DIM 3

int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    double dext{1.5}, daxn{3.0}, T{1}, deltaT{0.1}, alpha{0.45};
    int r = 1;

    Point<DIM> mass_center = {0.5, 0.5, 0.5};
    Point<DIM> radial_center = {0.2, 0.1, 0.1};
    
    RadialDiffusionTensor<3> diffusionTensor(dext, daxn, radial_center);
    FisherKolmogorov problem("../mesh/mesh-cube-40.msh",r,T,deltaT,diffusionTensor,alpha,mass_center);

    problem.setup();
    problem.solve();
    
    return 0;
}
