#include <iostream>
#include <fstream>

#include "DiffusionTensor.hpp"
#include "FisherKolmogorov.hpp"

#define DIM 3

int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    double dext{0.05}, daxn{0.30}, T{1}, deltaT{0.1}, alpha{2};
    int r = 1;
    Point<DIM> mass_center = {0.0, 0.0, 0.25};
    Point<2> origin = {0,-0.4};
    Point<DIM> radial_center = {0, 5, 0};

    std::string line,
                w,
                diffusion="CYLINDRICAL",
                mesh_file_name="../mesh/brain-3D-scaled.msh";
    std::ifstream f("./config.txt");
    if (f.good()) {
        while (std::getline(f, line)) {
            std::istringstream iss(line);
            iss >> w;
            if (w == "r") {
                iss >> r;
            }  else if (w == "mesh_file"){
                // char c;
                // iss >> c;
                std::string s(std::istreambuf_iterator<char>(iss), {});
                mesh_file_name=s;
                mesh_file_name.erase(0,1);
            }else if (w == "dext") {
                iss >> dext;
            } else if (w == "daxn") {
                iss >> daxn;
            } else if (w == "T") {
                iss >> T;
            } else if (w == "dt") {
                iss >> deltaT;
            } else if (w == "alpha") {
                iss >> alpha;
            } else if (w == "diffusion") {
                iss >> diffusion;
                if (diffusion != "RADIAL" && diffusion != "CYLINDRICAL") {
                    std::cerr << "Invalid diffusion tensor type\n";
                    exit(1);  
                }
            } else if (w == "origin") {
                iss  >> origin[0] >> origin[1];
            } else if (w == "mass-center") {
                for(int i = 0; i  < DIM;i++) iss >> mass_center[i];
            } else if (w == "radial-center") {
                for(int i = 0; i  < DIM;i++) iss >> mass_center[i];
            } else if (w.length()==0 || w[0]=='#') {
                // ignore empty lines and comments
            } 
            else {
                std::cerr << "Invalid parameter: "<<w<<"\n";
                exit(1);
            }
            
        }
    }

    std::cout << "dext          = " << dext << "\n";
    std::cout << "daxn          = " << daxn << "\n";
    std::cout << "T             = " << T << "\n";
    std::cout << "deltaT        = " << deltaT << "\n";
    std::cout << "alpha         = " << alpha << "\n";
    std::cout << "r             = " << r << "\n";
    std::cout << "diffusion     = " << diffusion << "\n";
    std::cout << "mass_center   = " << mass_center << "\n";
    if(diffusion == "CYLINDRICAL") std::cout << "origin        = " << origin << "\n";
    if(diffusion == "RADIAL") std::cout << "radial_center = " << radial_center << "\n";
    std::cout << "mesh_file     = " << mesh_file_name << "\n";
    std::cout << "===============================================" << std::endl;

    if (diffusion == "RADIAL") {
        RadialDiffusionTensor<DIM> diffusionTensor(dext, daxn, radial_center);
        FisherKolmogorov problem(mesh_file_name,r,T,deltaT,diffusionTensor,alpha,mass_center);
        problem.setup();
        problem.solve();
    }
    else if(diffusion == "CYLINDRICAL") {
        CylindricalDiffusionTensor diffusionTensor(dext, daxn, origin);
        FisherKolmogorov problem(mesh_file_name,r,T,deltaT,diffusionTensor,alpha,mass_center);
        problem.setup();
        problem.solve();
    }
    
    return 0;
}
