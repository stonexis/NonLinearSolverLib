#pragma once
#include <cstddef>
#include <iomanip>
#include <iostream>

template<typename T, std::size_t Rows, std::size_t Cols>
void Matrix<T, Rows, Cols>::Print(std::size_t width) const noexcept{
    for (std::size_t i = 0; i < Rows; ++i) {
        for (std::size_t j = 0; j < Cols; ++j)
            std::cout << std::setprecision(4) << std::setw(width) << (*this)(i,j);
        std::cout <<'\n';
    }
    std::cout << "-------------------------" << "\n";
}

