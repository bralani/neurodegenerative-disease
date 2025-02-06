
#include <fstream>
#include <iostream>

#include "DiffusionTensor.hpp"
#include "FisherKolmogorov.hpp"

#define DIMENSION 3

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  double     dext{1.0}, daxn{0.0}, T{1}, deltaT{0.1}, alpha{0.1};
  int        r             = 1;
  Point<DIMENSION> mass_center   = {0.5, 0.5, 0.5};
  Point<2>   origin        = {0, 0.1};
  Point<DIMENSION> radial_center = {0, 5, 0};

  const std::vector<std::string> mesh_files = {
    "../mesh/mesh-cube-5.msh",
    "../mesh/mesh-cube-10.msh",
    "../mesh/mesh-cube-20.msh",
    "../mesh/mesh-cube-40.msh",
  };
  const std::vector<double> h = {0.2, 0.1, 0.05, 0.025};

  std::vector<double> errors_L2_h;
  std::vector<double> errors_H1_h;

  for (const auto &mesh : mesh_files)
    {
      RadialDiffusionTensor<DIMENSION> diffusionTensor(dext, daxn, radial_center);
      FisherKolmogorov           problem(
        mesh, r, T, deltaT, diffusionTensor, alpha, mass_center);
      problem.setup();
      problem.solve();

      errors_L2_h.push_back(problem.compute_error(VectorTools::L2_norm));
      errors_H1_h.push_back(problem.compute_error(VectorTools::H1_norm));
    }


  const std::vector<double> p_orders = {
    1, 2, 3};
  std::vector<double> errors_L2_p;
  std::vector<double> errors_H1_p;
  for (const auto &p : p_orders)
    {
      RadialDiffusionTensor<DIMENSION> diffusionTensor(dext, daxn, radial_center);
      FisherKolmogorov           problem(
        "../mesh/mesh-cube-5.msh", p, T, deltaT, diffusionTensor, alpha, mass_center);
      problem.setup();
      problem.solve();

      errors_L2_p.push_back(problem.compute_error(VectorTools::L2_norm));
      errors_H1_p.push_back(problem.compute_error(VectorTools::H1_norm));
    }


  // Print the errors and estimate the convergence order.
  if (mpi_rank == 0)
    {
      std::cout << "==============================================="
                << std::endl;
      std::cout << "Convergence with respect to h" << std::endl;

      std::ofstream convergence_file("convergence_h.csv");
      convergence_file << "h,eL2,eH1" << std::endl;

      for (unsigned int i = 0; i < h.size(); ++i)
        {
          convergence_file << h[i] << "," << errors_L2_h[i] << "," << errors_H1_h[i]
                           << std::endl;

          std::cout << std::scientific << "h = " << std::setw(4)
                    << std::setprecision(2) << h[i];

          std::cout << std::scientific << " | eL2 = " << errors_L2_h[i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p = std::log(errors_L2_h[i] / errors_L2_h[i - 1]) /
                               std::log(h[i] / h[i - 1]);

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << std::scientific << " | eH1 = " << errors_H1_h[i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p = std::log(errors_H1_h[i] / errors_H1_h[i - 1]) /
                               std::log(h[i] / h[i - 1]);

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << "\n";
        }
    }


  // Print the errors and estimate the convergence order.
  if (mpi_rank == 0)
    {
      std::cout << "==============================================="
                << std::endl;
      std::cout << "Convergence with respect to p" << std::endl;

      std::ofstream convergence_file("convergence_p.csv");
      convergence_file << "p,eL2,eH1" << std::endl;

      for (unsigned int i = 0; i < p_orders.size(); ++i)
        {
          convergence_file << p_orders[i] << "," << errors_L2_p[i] << "," << errors_H1_p[i]
                           << std::endl;

          std::cout << std::scientific << "p = " << std::setw(4)
                    << std::setprecision(2) << p_orders[i];

          std::cout << std::scientific << " | eL2 = " << errors_L2_p[i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p = std::log(errors_L2_p[i] / errors_L2_p[i - 1]) /
                               std::log((1.0/p_orders[i]) / (1.0/p_orders[i - 1]));

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << std::scientific << " | eH1 = " << errors_H1_p[i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p = std::log(errors_H1_p[i] / errors_H1_p[i - 1]) /
                               std::log((1.0/p_orders[i]) / (1.0/p_orders[i - 1]));

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << "\n";
        }
    }

  return 0;
}