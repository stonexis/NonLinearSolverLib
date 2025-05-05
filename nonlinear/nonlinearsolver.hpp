#pragma once
#include "matrix.hpp"
#include "jacobian.hpp"

template<class Mat> struct ProxySimpleIter; //forward declaration
template<class Mat> struct ProxyNewton;
template<class MatF> struct ProxyNewtonModified;
template<class Mat> struct ProxyNewtonDiscrete;

template <typename T>
    struct SolverOptions {
        T eps_res_abs  = T(1e-12);
        T eps_res_rel  = T(1e-8);
        T eps_step_abs = T(1e-12);
        T eps_step_rel = T(1e-8);

        std::size_t max_iter = 100;
    };

namespace Backend {
    template<class MatF, class VecF, class VecX, 
                                     class Opt = SolverOptions<typename MatF::Scalar>>
    [[nodiscard]]VecX solve_newton_modified(
                                        const MatF& J, 
                                        const VecF& F, 
                                        VecX& x0, 
                                        const Opt& options = {}
                                    );

    template <class Mat>
    void inplace_lu_decomposition(Mat& matrix_A);
    
    template <class Mat, class Vec>
    void inplace_forward_backward_sub(const Mat& matrix_decomposition, const Vec& vector_b, Vec& solution);


}


template<class MatF>
struct Tools {
    const MatF& J;
    constexpr auto simple_iter()      const noexcept { return ProxySimpleIter {J}; }
    constexpr auto newton()           const noexcept { return ProxyNewton {J}; }
    constexpr auto newton_modified()  const noexcept { return ProxyNewtonModified {J}; }
    constexpr auto newton_discrete()  const noexcept { return ProxyNewtonDiscrete {J}; }
};
template<class MatF> Tools(const MatF&) -> Tools<MatF>; //class‑template argument deduction

// J  -> Tools<MatF> temp -> ProxyNewtonModified<MatF> temp -> backend call
//          (holds &J)             (holds &J as pointer)

//-------------------------- Proxy calls ------------------

template<class MatF>
class ProxyNewtonModified
{
    const MatF* J_;                                      
public:
    explicit constexpr ProxyNewtonModified(const MatF& J) noexcept : J_{&J} {}

    template<class VecF, class VecX,
             class Opt = SolverOptions<typename MatF::Scalar>>
    [[nodiscard]]
    constexpr auto solve(const VecF& F, VecX& x0, const Opt&  opt = {}) const { 
        return Backend::solve_newton_modified(*J_, F, x0, opt);
    }
};



#include "backimplement.tpp"