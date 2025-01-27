#ifndef DIFFUSION_TENSOR_HPP
#define DIFFUSION_TENSOR_HPP

template <unsigned int DIM>
class DiffusionTensor : public TensorFunction<2, DIM> {
public:
    DiffusionTensor(double dext, double daxn) : dext(dext), daxn(daxn), identity(unit_symmetric_tensor<DIM>()) {}

    Tensor<2, DIM> value(const Point<DIM> &p) const override {
        return dext*indentity + daxn*computeNxN(p);
    }

protected:
    Tensor<2,DIM> computeNxN(const Point<DIM> &p) const {
        computeFiber(p);
        return outer_product(fiber_n,fiber_n); 
    }  

    virtual void computeFiber(const Point<DIM> &p) = 0;
    Tensor<1,DIM> fiber_n; 

private:
    const double dext, daxn;
    const SymmetricTensor<2,DIM> identity;
};

template <unsigned int DIM>
class RadialDiffusionTensor : public DiffusionTensor<DIM> {
public:
    RadialDiffusionTensor(double dext, double daxn, double mass_center) : DiffusionTensor(dext, daxn), mass_center(mass_center){}

protected:
    void computeFiber(const Point<DIM> &p) override {
        for(int i = 0; i < DIM;i++) {
            fiber_n[i] = (p[i]-mass_center[i])/(p.distance(mass_center) + 1.e-6);
        }
    }

private:
    const Point<DIM> mass_center;
};

#endif