#pragma once
#include "matrix.hpp"


namespace ls {

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