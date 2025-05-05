#pragma once
#include <cstddef>

template<typename T, std::size_t Rows, std::size_t Cols>
struct Matrix {
    using Scalar = T;
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    
    inline T& operator()(std::size_t i, std::size_t j) noexcept {return m_data[Cols * i + j];}; 
    inline const T& operator()(std::size_t i, std::size_t j) const noexcept {return m_data[Cols * i + j];};

    inline T& operator[](std::size_t i){return m_data[i];}
    inline const T& operator[](std::size_t i) const {return m_data[i];}

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    
    void Print(std::size_t width = 8) const noexcept;

    T m_data[Rows * Cols]{};
};
#include "matrix.tpp"