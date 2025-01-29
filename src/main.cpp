#include <iostream>

#include "DiffusionTensor.hpp"
#include "FisherKolmogorov.hpp"


int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    double dext{1.5}, daxn{3.0}, T{1.0}, deltaT{0.01}, alpha{0.45};
    int r = 1;

    RadialDiffusionTensor<3> diffusionTensor(dext, daxn);
    FisherKolmogorov problem("../mesh/mesh-cube-5.msh",r,T,deltaT,diffusionTensor,alpha);

    problem.setup();
    problem.solve();
    
    return 0;
}