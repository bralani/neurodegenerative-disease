#ifndef HEAT_NON_LINEAR_HPP
#define HEAT_NON_LINEAR_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "DiffusionTensor.hpp"

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class FisherKolmogorov
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      if(CONVERGENCE_TEST) {
        double c_val = std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) * std::cos(M_PI * p[2]) * std::exp(-get_time());
        return (3 * M_PI * M_PI - 1) * c_val - 0.1 * c_val * (1 - c_val);
      }
      else {
        return 0.0;
      }
    }
  };

  // Function for initial conditions.
  class FunctionU0 : public Function<dim>
  {

  static constexpr double ray_max = 0.05;
  static constexpr double concentration = 1.0;

  public:

    FunctionU0(const Point<dim> &p) : Function<dim>(), _mass_center(p) {}

    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      double dist = p.distance(_mass_center);

      if(dist < ray_max) 
        return ((ray_max - dist) / ray_max) * concentration;

      return 0.0;
    }
  private:
    Point<dim> _mass_center;
  };


  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) * std::cos(M_PI * p[2]) * std::exp(-get_time());

    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      // duex / dx
      result[0] = -M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]) * std::cos(M_PI * p[2]) * std::exp(-get_time());

      // duex / dy
      result[1] = -M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::cos(M_PI * p[2]) * std::exp(-get_time());

      // duex / dz
      result[2] = -M_PI * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]) * std::sin(M_PI * p[2]) * std::exp(-get_time());

      return result;
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  FisherKolmogorov(const std::string  &mesh_file_name_,
                const unsigned int &r_,
                const double       &T_,
                const double       &deltat_,
                DiffusionTensor<dim>& diffusion_tensor,
                const double alpha_,
                const Point<dim> &mass_center)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , mesh(MPI_COMM_WORLD)
    , diffusion_tensor(diffusion_tensor)
    , alpha(alpha_)
    , u_0(mass_center)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();


  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type);

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the linear system associated to the tangent problem.
  void
  solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Forcing term.
  ForcingTerm forcing_term;

  // Exact solution.
  ExactSolution exact_solution;

  // Initial conditions.
  FunctionU0 u_0;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;

private:
  DiffusionTensor<dim>& diffusion_tensor;
  
  const double alpha;
};

#endif