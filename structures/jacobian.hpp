#pragma once
#include <cstddef>
#include <type_traits>
#include <algorithm>


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

template <typename R, 
         std::size_t N, 
         template<typename,std::size_t> class VecT = VectorX>
class VectorF
{
public:
    using Vec = VecT<R,N>;
    using FuncPtr = R (*)(const Vec&); //F_i : Vec -> R
    using Scalar = R;

    static constexpr std::size_t size = N;

    template<typename... Fs,
             typename = std::enable_if_t<(sizeof...(Fs)==N)>>
    constexpr explicit VectorF(Fs... fs) noexcept : f_{static_cast<FuncPtr>(fs)...} {}

    constexpr FuncPtr&       operator()(std::size_t i)       noexcept { return f_[i]; }
    constexpr const FuncPtr& operator()(std::size_t i) const noexcept { return f_[i]; }

private:
    FuncPtr f_[N]{};
};

