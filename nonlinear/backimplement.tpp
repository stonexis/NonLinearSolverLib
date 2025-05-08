#include <cstddef>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include "matrix.hpp"
#include "jacobian.hpp"

namespace Backend {
    
    template<class MatF, class VecF, class VecX, class Opt>
    std::pair<VecX, std::size_t> solve_newton_modified(const MatF& J, const VecF& F, VecX& x0, const Opt& opt){
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
                inplace_lu_decomposition(matrix_A_SLAE); //Обновляем разложение
            }
            else //Если не пора, то обновляем только правую часть
                for (std::size_t i = 0; i < N; ++i)
                    rhs[i] = -F(i)(x0); //Перевычисляем -F (покомпонентное применение функций лежащих в F к вектору)

            inplace_forward_backward_sub(matrix_A_SLAE, rhs, vector_step); //Прямой и обратный проход для вычисления s из уравнения Js = -F

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


    template<class MatF, class VecF, class VecX, class Opt>
    std::pair<VecX, std::size_t> solve_newton_discrete(const MatF& J, const VecF& F, VecX& x0, const Opt& opt){
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
                inplace_lu_decomposition(matrix_A_SLAE); //Обновляем разложение
            }
            else //Если не пора, то обновляем только правую часть
                for (std::size_t i = 0; i < N; ++i)
                    rhs[i] = -F(i)(x0); //Перевычисляем -F (покомпонентное применение функций лежащих в F к вектору)

            inplace_forward_backward_sub(matrix_A_SLAE, rhs, vector_step); //Прямой и обратный проход для вычисления s из уравнения Js = -F

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

    template <class Mat>
    void inplace_lu_decomposition(Mat& matrix_A){
        constexpr std::size_t dim_matrix = Mat::rows;
        using S = typename Mat::Scalar;

        //In-place реализация, L и U записываются поверх A. Богачев стр 20
        //В форме краута LU разложения диагональ исходной матрицы лежит в L, а диагональ U = 1 (единичная)
        //Первый столбец L равен первому столбцу A, поскольку inplace реализация, столбец остается без изменений
        for(std::size_t k = 1; k < dim_matrix; k++) //В случае с неленточной матрицей граница цикла = dim_matrix
            (matrix_A)(0,k) /= (matrix_A)(0,0); //Вычисляем элементы первой строки матрицы U

        auto col_calc = [&](std::size_t i, std::size_t k){ //Вычисление l_ik
            S sum = S(0);
            for(std::size_t j = 0; j < k; j++)
                sum += (matrix_A)(i,j) * (matrix_A)(j, k);   
            (matrix_A)(i, k) -= sum;
        };
        auto row_calc = [&](std::size_t i, std::size_t k){ //Вычисление u_ik
            S sum = S(0);
            for(std::size_t j = 0; j < i; j++)
                sum += (matrix_A)(i, j) * (matrix_A)(j , k);
            (matrix_A)(i, k) = ((matrix_A)(i, k) - sum) / (matrix_A)(i, i);
        };

        for(std::size_t i = 1; i < dim_matrix; i++){
            for(std::size_t k = 1; k <= i; k++)
                col_calc(i, k);
                
            for(std::size_t k = i + 1; k < dim_matrix; k++)
                row_calc(i, k);
        }
        
    }
    template <class Mat, class Vec>
    void inplace_forward_backward_sub(const Mat& matrix_decomposition, const Vec& vector_b, Vec& solution){

        constexpr std::size_t dim_matrix = Mat::rows;
        using S = typename Mat::Scalar;
        Vec y;                            
        
        for (std::size_t i = 0; i < dim_matrix; i++) {
            S sum = 0;
            for (std::size_t j = 0; j < i; j++)
                sum += (matrix_decomposition)(i,j) * y[j];  // L_{i,j} * y_j
            
            y[i] = (vector_b[i] - sum) / (matrix_decomposition)(i,i);
        }

        // Обратный ход
        solution[dim_matrix - 1] = y[dim_matrix - 1]; //Стартовая итерация с последней строки, Так как U_{i,i} = 1 делить не нужно
        for (int i = dim_matrix - 2; i >= 0; i--) {
            S tmp_sum = S(0);
            
            for (std::size_t j = i + 1; j < dim_matrix; j++) //Из-за ленточной структуры матрицы не нужно идти от диагонали правее, чем на длину блока(длину диагонали)
                tmp_sum += (matrix_decomposition)(i,j) * solution[j]; // U_{i,j} * x_j
            solution[i] = y[i] - tmp_sum; // Так как U_{i,i} = 1 делить не нужно
        }

    }

}
