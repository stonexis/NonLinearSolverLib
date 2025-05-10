#include <iostream>
#include <string_view>
#include "nlslib.hpp"

using namespace std;

int main(){

    using Vec2 = VectorX<double,2>;

    //Правые части
    auto F0 = [](const Vec2& v){ return std::pow(v[0],4) - v[1]; };
    auto F1 = [](const Vec2& v){ return v[0] - std::log(v[1]); };
    VectorF<double, 2>  F(F0,F1);

    //Якобиан
    auto J00 = [](const Vec2& v){ return 4 * std::pow(v[0],3); };
    auto J01 = [](const Vec2&  ){ return -1.0; };
    auto J10 = [](const Vec2&  ){ return  1.0; };
    auto J11 = [](const Vec2& v){ return -1.0/v[1]; };
    Jacobian<double, 2, 2> J(J00,J01,J10,J11);

    //Начальные приближения
    Vec2 x0_first{-0.55, 0.5};
    Vec2 x0_second{1.38, 3.99};

    //Для методов с аналитическим Якобианом
    auto toolsJ = ToolsWithJ{J}; //агрегатная инициализация, поскольку Tools не имеет конструктора, если создать конструктор для Tools, можно вызывать как ()
    //Для методов без аналитического Якобиана
    auto toolsF = ToolsWithF{F};
    SolverOptions<Vec2> options({-0.1, -0.1}); //Для отрицательного корня
    //Метод простой итерации
    auto [simple_iter_first_root, simple_iter_first_speed] = toolsF.simple_iter().solve(x0_first, options);
    options.lambdas[0] = 0.1; // Для положительного корня x1 всегда >0 поэтому его лямбда не изменяется
    auto [simple_iter_second_root, simple_iter_second_speed] = toolsF.simple_iter().solve(x0_second, options);

    //Метод Ньютона
    options.k_refresh_J=1; 
    auto [base_newton_first_root, base_newton_first_speed] = toolsJ.newton_modified().solve(F, x0_first, options);
    auto [base_newton_second_root, base_newton_second_speed] = toolsJ.newton_modified().solve(F, x0_second, options);

    //Модифицированный метод Ньютона
    options.k_refresh_J=4;
    auto [modified_newton_first_root, modified_newton_first_speed] = toolsJ.newton_modified().solve(F, x0_first, options);
    auto [modified_newton_second_root, modified_newton_second_speed] = toolsJ.newton_modified().solve(F, x0_second, options);

    //Дискретный метод Ньютона
    auto [discrete_newton_first_root, discrete_newton_first_speed] = toolsF.newton_discrete().solve(x0_first, options);
    auto [discrete_newton_second_root, discrete_newton_second_speed] = toolsF.newton_discrete().solve(x0_second, options);

    auto print = [](std::string_view method_name, const Vec2& root, std::size_t speed){
        cout << std::left << std::setw(20) << method_name << " " << "root: ";
        root.print();
        cout << std::left << std::setw(20) << " total iterations: "<< speed << '\n';
        cout << std::setfill('-') << std::setw(92) << "" << std::setfill(' ') << '\n';
    };

    print("simple_iter", simple_iter_first_root, simple_iter_first_speed);
    print("simple_iter", simple_iter_second_root, simple_iter_second_speed);

    print("base_newton", base_newton_first_root, base_newton_first_speed);
    print("base_newton", base_newton_second_root, base_newton_second_speed);

    print("modified_newton", modified_newton_first_root, modified_newton_first_speed);
    print("modified_newton", modified_newton_second_root, modified_newton_second_speed);

    print("discrete_newton", discrete_newton_first_root, discrete_newton_first_speed);
    print("discrete_newton", discrete_newton_second_root, discrete_newton_second_speed);

    return 0;
}