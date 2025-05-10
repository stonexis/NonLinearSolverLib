#pragma once
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "matrix.hpp"

//-------------------------------------------------

template<typename T, std::size_t N>
struct VectorX{
    using Scalar = T;
    static constexpr std::size_t size = N;

    T&       operator[](std::size_t i) { return _data[i]; }
    const T& operator[](std::size_t i) const { return _data[i]; }

    VectorX()                                    = default;
    constexpr VectorX(const VectorX&)            = default;
    constexpr VectorX& operator=(const VectorX&) = default;

    constexpr T l_inf_norm() const noexcept {
        T m = T(0);
        for(const auto& v : _data) 
            m = std::max<T>(m, std::abs(v));
        return m;
    }
    constexpr void print() const noexcept {
        for(const auto& v : _data)
            std::cout << std::left << std::setprecision(9) << std::scientific << std::setw(20) << v << " ";
    }
    T _data[N];
};

//-------------------------------------------------

template <typename R, //Тип возвращаемого значения функций, лежащих внутри Якобиана
          std::size_t Rows,
          std::size_t Cols,
          template<typename,std::size_t> class VecT = VectorX> //Список аргументов функций, лежащийх внутри Якобиана
class Jacobian{
public:
    using Vec = VecT<R,Cols>;       
    using FuncPtr = R (*)(const Vec&); // double *Func(const Vec) J_ij : Vec -> R
    using Scalar = R;

    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    
    template<typename... Fs, // Набор самых разных типов параметров, никак не связанных с FuncPtr
             typename = std::enable_if_t<(sizeof...(Fs) == Rows * Cols)>>//static assert если логическое значение равно false, подстановка этого параметра шаблона не удаётся конструктор просто исчезает из разрешения перегрузки, поэтому вызов не компилируется в случае если другие кандидаты не найдутся (SFINAE)
    constexpr explicit Jacobian(Fs... fs) noexcept : f_{ static_cast<FuncPtr>(fs)... } {} // Если невозможно привести тип Fs к FuncPtr То ошибка компиляции, неподходящий параметр, (отброс лямбд с захватом и функторов) 
    
    FuncPtr&       operator()(std::size_t i, std::size_t j) noexcept { return f_[i * Cols + j]; }
    const FuncPtr& operator()(std::size_t i, std::size_t j) const noexcept { return f_[i * Cols + j]; }
    
private:
    FuncPtr f_[Rows * Cols]{};
};

//-------------------------------------------------
//Вектор функций, каждая функция от N аргументов 
template <typename R, 
         std::size_t N, 
         template<typename,std::size_t> class VecT = VectorX,
         template<typename,std::size_t,std::size_t> class MatT = Matrix>
class VectorF
{
public:
    using Vec = VecT<R, N>; //Формат внутреннего представления аргументов функций
    using Mat = MatT<R, N, N>; //Формат результата операции взятия численного Якобиана
    using FuncPtr = R (*)(const Vec&); //F_i : Vec -> R
    using Scalar = R;

    static constexpr std::size_t size = N;

    template<typename... Fs,
             typename = std::enable_if_t<(sizeof...(Fs)==N)>> //Если передано не N функций - ошибка на этапе компиляции
    constexpr explicit VectorF(Fs... fs) noexcept : f_{static_cast<FuncPtr>(fs)...} {}

    constexpr void GetNumericJacInplace(Mat& Jac, const Vec& args, R eps = 1e-10) const noexcept {
        Vec args_copy = args;
        for (std::size_t i = 0; i < N; i++){
            for (std::size_t j = 0; j < N; j++){
                args_copy[j] += eps; //Добавление возмущения для численной производной по j компоненте
                Jac(i, j) = (f_[i](args_copy) - f_[i](args)) / eps; //[F(x+h) - F(x)]/h
                args_copy[j] -= eps; //Необходима производная по отдельной компоненте, необходимо отнять прибавленное возмущение, для вычисления следующей производной
            }
        }
    } 

    constexpr FuncPtr&       operator()(std::size_t i)       noexcept { return f_[i]; }
    constexpr const FuncPtr& operator()(std::size_t i) const noexcept { return f_[i]; }

private:
    FuncPtr f_[N]{}; //Аргументы функций не хранятся в структуре
};

