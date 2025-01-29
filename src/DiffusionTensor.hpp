#ifndef DIFFUSION_TENSOR_HPP
#define DIFFUSION_TENSOR_HPP

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

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/tensor_function.h>

using namespace dealii;

template <unsigned int DIM>
class DiffusionTensor : public TensorFunction<2, DIM> {
public:
    DiffusionTensor(double dext, double daxn) : dext(dext), daxn(daxn), identity(unit_symmetric_tensor<DIM>()) {}

    virtual void set_mass_center(const Point<DIM>& /*_mass_center*/) {}

    Tensor<2, DIM> value(const Point<DIM> &p) const override {
        return dext*identity + daxn*computeNxN(p);
    }

protected:
    Tensor<2,DIM> computeNxN(const Point<DIM> &p) const {
        Tensor<1,DIM> fiber_n = computeFiber(p);
        return outer_product(fiber_n,fiber_n); 
    }  

    virtual Tensor<1,DIM> computeFiber(const Point<DIM> &p) const = 0; 

private:
    const double dext, daxn;
    const SymmetricTensor<2,DIM> identity;
};

template <unsigned int DIM>
class RadialDiffusionTensor : public DiffusionTensor<DIM> {
public:
    RadialDiffusionTensor(double dext, double daxn) : 
        DiffusionTensor<DIM>(dext, daxn){}

    virtual void set_mass_center(const Point<DIM> &_mass_center) override {
        mass_center=_mass_center;
    }

protected:
    Tensor<1,DIM> computeFiber(const Point<DIM> &p) const override {
        Tensor<1,DIM> fiber_n;
        for(int i = 0; i < DIM;i++) {
            fiber_n[i] = (p[i]-mass_center[i])/(p.distance(mass_center) + 1.e-6);
        }
        return fiber_n;
    }

private:
    Point<DIM> mass_center;
};

#endif