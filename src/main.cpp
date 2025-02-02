#include <iostream>

#include "DiffusionTensor.hpp"
#include "FisherKolmogorov.hpp"

#define DIM 3

int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    double dext{0.05}, daxn{0.30}, T{1}, deltaT{0.1}, alpha{2};
    int r = 1;

    Point<DIM> mass_center = {0.0, 0.0, 0.25};

    // Point<DIM> radial_center = {0, 5, 0};
    // RadialDiffusionTensor<DIM> diffusionTensor(dext, daxn, radial_center);
    // FisherKolmogorov problem("../mesh/brain-3D-scaled.msh",r,T,deltaT,diffusionTensor,alpha,mass_center);
    
    Point<2> origin = {0,-0.4};
    CylindricalDiffusionTensor diffusionTensor(dext, daxn, origin);
    FisherKolmogorov problem("../mesh/brain-3D-scaled.msh",r,T,deltaT,diffusionTensor,alpha,mass_center);


    problem.setup();
    problem.solve();
    
    return 0;
}
