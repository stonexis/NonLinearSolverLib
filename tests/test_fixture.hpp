#pragma once
#include <memory>
#include "gtest/gtest.h"
#include <ceres/ceres.h>
#include "nonlinearsolver.hpp"


using T = double;
constexpr std::size_t N = 2;                    
using VecX = VectorX<T, N>;
using VecF = VectorF<T, N>;
using Jac = Jacobian<T, N, N>;
using Opt = SolverOptions<double>;

struct Functors {
    // F(x)  (return scalar)  and  J_ij(x)  (return scalar)
    static T F0 (const VecX& v){ return std::pow(v[0],4) - v[1];} 
    static T F1 (const VecX& v){ return v[0] - std::log(v[1]);}
  
    static T J00(const VecX& v){ return 4*std::pow(v[0],3);} // dF0/dx0
    static T J01(const VecX&  ){ return -1.0; } // dF0/dx1
    static T J10(const VecX&  ){ return  1.0; } // dF1/dx0
    static T J11(const VecX& v){ return -1.0/v[1];} // dF1/dx1
  };



struct NewtonFixture : public ::testing::Test {
    
    struct CeresF {
        template<typename U>
        bool operator()(const U* const x, U* r) const {
          r[0] = ceres::pow(x[0],4) - x[1];
          r[1] = x[0] - ceres::log(x[1]);
          return true;
        }
    };

    void SetUp() override {
        

        Jptr = std::make_unique<Jac>(Jac(Functors::J00, Functors::J01, Functors::J10, Functors::J11));
        Fptr = std::make_unique<VecF>(VecF(Functors::F0, Functors::F1));
        x0ptr = std::make_unique<VecX>(VecX{-0.5, 0.5});

        //CERES
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CeresF,2,2>(new CeresF), nullptr, x_ceres);
        opts.minimizer_type = ceres::TRUST_REGION;
        opts.max_num_iterations = 50;
        opts.linear_solver_type = ceres::DENSE_QR;
        opts.function_tolerance = 1e-12;
        opts.gradient_tolerance = 1e-12;
        opts.parameter_tolerance = 1e-12;
        opts.minimizer_progress_to_stdout = false;

    }

    void TearDown() override {
    }
    std::unique_ptr<Jac>  Jptr; //Оборачиваем в указатели, поскольку у обьектов Jac VecF нет конструкторов по умолчанию,
    std::unique_ptr<VecF> Fptr; //(в противном случае, чтобы скомпилировалось,
    std::unique_ptr<VecX> x0ptr; //необходимо писать явный конструктор фикстуры с какими то значениями, что на практиве не реализуется)
    
    // Ceres data
    ceres::Problem problem;
    ceres::Solver::Options opts;
    ceres::Solver::Summary summary;
    double x_ceres[2]{-0.5, 0.5};
};

