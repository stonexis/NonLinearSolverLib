#include <cstddef>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include "matrix.hpp"
#include "jacobian.hpp"
#include "lsutils.hpp"

namespace nls {
    
    template<class MatF, class VecF, class VecX, class Opt>
    std::pair<VecX, std::size_t> newton_modified(const MatF& J, const VecF& F, VecX x0, const Opt& opt){
        if (opt.k_refresh_J == 0) throw std::invalid_argument("k_refresh_J must > 0");
        static_assert(MatF::rows == VecF::size, "Jacobian row count must equal F size");
        static_assert(MatF::cols == VecX::size, "Jacobian col count must equal x size");

        using T = typename MatF::Scalar;
        constexpr std::size_t N = MatF::rows;
        constexpr std::size_t M = MatF::cols;      

        Matrix<T,N,N> matrix_A_SLAE{};
        VectorX<T,N> rhs{};
        VectorX<T,N> vector_step{};
        std::size_t total_iters = 0;
        for (; total_iters < opt.max_iter; ++total_iters){

            if (total_iters % opt.k_refresh_J == 0){ //Пора обновлять Якобиан

                for (std::size_t i = 0; i < N; ++i){
                    rhs[i] = -F(i)(x0); //Перевычисляем -F (покомпонентное применение функций лежащих в F к вектору)

                    for(std::size_t j = 0; j < M; ++j)
                        matrix_A_SLAE(i,j) =  J(i,j)(x0);  // J (покомпонентное применение функций лежащих в якобиане к вектору)
                }
                ls::inplace_lu_decomposition(matrix_A_SLAE); //Обновляем разложение
            }
            else //Если не пора, то обновляем только правую часть
                for (std::size_t i = 0; i < N; ++i)
                    rhs[i] = -F(i)(x0); //Перевычисляем -F (покомпонентное применение функций лежащих в F к вектору)

            ls::inplace_forward_backward_sub(matrix_A_SLAE, rhs, vector_step); //Прямой и обратный проход для вычисления s из уравнения Js = -F

            T diff_inf = T(0); //inf норма разности нового и старого приближений
            T x_inf = T(0); //inf норма вектора решения

            for (std::size_t i = 0; i < N; ++i){
                T old = x0[i]; //сохранения старого значения для вычисления разности
                x0[i] += vector_step[i];  // x_k+1 = x_k + s_k                     
                diff_inf = std::max(diff_inf, std::abs(x0[i]-old)); // L_inf(x_k+1 - x_k)
                x_inf = std::max(x_inf, std::abs(x0[i])); //L_inf(x_k)
            }
            
            const bool small_step = diff_inf < opt.eps_step_abs;
            const bool small_res = rhs.l_inf_norm() < opt.eps_res_abs;    
            if (small_step && small_res)
                break; //сошлись                             
        }

        return std::make_pair(x0, total_iters);                                      
    }


    template<class VecF, class VecX, class Opt>
    std::pair<VecX, std::size_t> newton_discrete(const VecF& F, VecX x0, const Opt& opt){
        static_assert(VecF::size == VecX::size, "F size must equal x size");
        
        using T = typename VecF::Scalar;
        constexpr std::size_t N = VecF::size;

        Matrix<T,N,N> matrix_A_SLAE{};
        VectorX<T,N> rhs{};
        VectorX<T,N> vector_step{};
        std::size_t total_iters = 0;
        for (; total_iters < opt.max_iter; ++total_iters){

            for (std::size_t i = 0; i < N; ++i)
                rhs[i] = -F(i)(x0); //Перевычисляем -F (покомпонентное применение функций лежащих в F к вектору)
            
            F.GetNumericJacInplace(matrix_A_SLAE, x0); //Перевычисляем якобиан численно inplace в заранее созданной матрице 

            ls::inplace_lu_decomposition(matrix_A_SLAE); //Обновляем разложение
            
            ls::inplace_forward_backward_sub(matrix_A_SLAE, rhs, vector_step); //Прямой и обратный проход для вычисления s из уравнения Js = -F

            T diff_inf = T(0); //inf норма разности нового и старого приближений
            T x_inf = T(0); //inf норма вектора решения

            for (std::size_t i = 0; i < N; ++i){
                T old = x0[i]; //сохранения старого значения для вычисления разности
                x0[i] += vector_step[i];  // x_k+1 = x_k + s_k                     
                diff_inf = std::max(diff_inf, std::abs(x0[i]-old)); // L_inf(x_k+1 - x_k)
                x_inf = std::max(x_inf, std::abs(x0[i])); //L_inf(x_k)
            }
            
            const bool small_step = diff_inf < opt.eps_step_abs;
            const bool small_res = rhs.l_inf_norm() < opt.eps_res_abs;    
            if (small_step && small_res)//сошлись
                break;                             
        }

        return std::make_pair(x0,total_iters);                                      
    }
    template<class VecF, class VecX, class Opt>
    std::pair<VecX, std::size_t> simple_iter(const VecF& F, VecX x0, const Opt& opt){
        static_assert(VecF::size == VecX::size, "F size must equal x size");

        using T = typename VecF::Scalar;
        constexpr std::size_t N = VecF::size;
        VectorX<T,N> rhs{}; //Для вычисления невязки
        
        std::size_t total_iters = 0;
        for (; total_iters < opt.max_iter; ++total_iters){

            T diff_inf = T(0); //inf норма разности нового и старого приближений
            T x_inf = T(0); //inf норма вектора решения

            for (std::size_t i = 0; i < N; ++i){
                T old = x0[i]; //сохранения старого значения для вычисления разности
                rhs[i] = F(i)(x0); // сохраняем значение для критерия остановки
                //G(x) = x - lambda * F(x)
                //Для сходимости норма якобиана должна быть меньше 1, Якобиан для этого метода: J_G(x) =  I - lambda*J_F(x)
                x0[i] -= opt.lambdas[i] * rhs[i]; // x_k+1 = x_k - lambda * F(x_k)                    
                diff_inf = std::max(diff_inf, std::abs(x0[i]-old)); // L_inf(x_k+1 - x_k)
                x_inf = std::max(x_inf, std::abs(x0[i])); //L_inf(x_k)
            }

            const bool small_step = diff_inf < opt.eps_step_abs;
            const bool small_res = rhs.l_inf_norm() < opt.eps_res_abs;    
            if (small_step && small_res)//сошлись
                break;



        }
        return std::make_pair(x0,total_iters);

    }
}