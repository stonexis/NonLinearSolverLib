#pragma once
#include "matrix.hpp"
#include "jacobian.hpp"

template<class VecF> struct ProxySimpleIter; //forward declaration
template<class MatF> struct ProxyNewtonModified;
template<class VecF> struct ProxyNewtonDiscrete;

template <typename VecX>
struct SolverOptions {
    using T = typename VecX::Scalar;
    VecX lambdas; ///Вектор множителей для каждой компоненты для метода простой итерации
    T eps_res_abs           = T(1e-12); //Допуск по невязке
    T eps_step_abs          = T(1e-12); //Допуск по размеру шага

    std::size_t k_refresh_J = 3; //Количество шагов, через которые обновляется Якобиан(для модифицированного ньютона)
    std::size_t max_iter    = 300;
    explicit SolverOptions(VecX l) : lambdas(std::move(l)){};
};

namespace nls {

    ///Модифицированный метод Ньютона при k_refresh_J = 1 - обычный Ньютон
    template<class MatF, class VecF, class VecX, 
                                     class Opt = SolverOptions<VecX>>
    [[nodiscard]] std::pair<VecX, std::size_t> newton_modified(
                                                        const MatF& J, 
                                                        const VecF& F, 
                                                        VecX x0, 
                                                        const Opt& options
                                                    );

    ///Дискретный метод Ньютона с численной аппроксимацией производных                                                        
    template<class VecF, class VecX, 
                         class Opt = SolverOptions<VecX>>
    [[nodiscard]] std::pair<VecX, std::size_t> newton_discrete(
                                                        const VecF& F, 
                                                        VecX x0, 
                                                        const Opt& options
                                                    );
    template<class VecF, class VecX, 
                         class Opt = SolverOptions<VecX>>
    [[nodiscard]] std::pair<VecX, std::size_t> simple_iter(
                                                        const VecF& F, 
                                                        VecX x0, 
                                                        const Opt& options
                                                    );

}

template<class MatF>
struct ToolsWithJ {
    const MatF& J;
    constexpr auto newton_modified()  const noexcept { return ProxyNewtonModified {J}; }
};
template<class MatF> ToolsWithJ(const MatF&) -> ToolsWithJ<MatF>; //class‑template argument deduction

// J  -> Tools<MatF> temp -> ProxyNewtonModified<MatF> temp -> backend call

template<class VecF>
struct ToolsWithF {
    const VecF& F;
    constexpr auto simple_iter()      const noexcept { return ProxySimpleIter {F}; }
    constexpr auto newton_discrete()  const noexcept { return ProxyNewtonDiscrete {F}; }
};
template<class VecF> ToolsWithF(const VecF&) -> ToolsWithF<VecF>;


//-------------------------- Proxy calls ------------------

template<class MatF>
class ProxyNewtonModified
{
    const MatF* J_;                                      
public:
    explicit constexpr ProxyNewtonModified(const MatF& J) noexcept : J_{&J} {}

    template<class VecF, class VecX,
             class Opt = SolverOptions<VecX>>
    [[nodiscard]]
    constexpr auto solve(const VecF& F, VecX x0, const Opt&  opt) const { 
        return nls::newton_modified(*J_, F, x0, opt);
    }
};

template<class VecF>
class ProxyNewtonDiscrete
{
    const VecF* F_;                                      
public:
    explicit constexpr ProxyNewtonDiscrete(const VecF& F) noexcept : F_{&F} {}

    template<class VecX, class Opt = SolverOptions<VecX>>
    [[nodiscard]]
    constexpr auto solve(VecX x0, const Opt&  opt) const { 
        return nls::newton_discrete(*F_, x0, opt);
    }
};

template<class VecF>
class ProxySimpleIter
{
    const VecF* F_;                                      
public:
    explicit constexpr ProxySimpleIter(const VecF& F) noexcept : F_{&F} {}

    template<class VecX, class Opt = SolverOptions<VecX>>
    [[nodiscard]]
    constexpr auto solve(VecX x0, const Opt&  opt) const { 
        return nls::simple_iter(*F_, x0, opt);
    }
};


#include "nlsdetail.tpp"