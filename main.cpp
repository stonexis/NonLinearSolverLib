#include <iostream>
#include "nonlinearsolver.hpp"

using namespace std;

int main(){

    using Vec2 = VectorX<double,2>;

    //Правые части
    auto F0 = [](const Vec2& v){ return std::pow(v[0],4) - v[1]; };
    auto F1 = [](const Vec2& v){ return v[0] - std::log(v[1]);     };
    VectorF<double, 2>  F(F0,F1);

    //Якобиан
    auto J00 = [](const Vec2& v){ return 4*std::pow(v[0],3); };
    auto J01 = [](const Vec2&  ){ return -1.0;               };
    auto J10 = [](const Vec2&  ){ return  1.0;               };
    auto J11 = [](const Vec2& v){ return -1.0/v[1];          };
    Jacobian<double, 2, 2> J(J00,J01,J10,J11);


    Vec2 x0_first{-0.6, 0.5};
    Vec2 x0_second{1.38, 3.99};
    
    auto tools = Tools{J}; //агрегатная инициализация, поскольку Tools не имеет конструктора, если создать конструктор для Tools, можно вызывать как ()
    auto x_first = tools.newton_modified().solve(F, x0_first);
    auto x_second = tools.newton_modified().solve(F, x0_second);

    for(std::size_t i = 0; i < 2; i++)
        cout << x_first[i] << " ";

    for(std::size_t i = 0; i < 2; i++)
        cout << x_second[i] << " ";


    return 0;
}