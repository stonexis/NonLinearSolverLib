#include <benchmark/benchmark.h>
#include "nonlinearsolver.hpp"
#include "benchmark_fixture.hpp"            

//Benchmark body, defined once
BENCHMARK_DEFINE_F(NewtonFixture, Solve)(benchmark::State& state) {
    Opt opt;
    opt.k_refresh_J = static_cast<std::size_t>(state.range(0));// param
    std::size_t newtonSteps = 0;
    for (auto _ : state) {
      VecX x = *x0ptr; // fresh copy each run
      auto [sol, iters] = Backend::solve_newton_modified(*Jptr, *Fptr, x, opt);
      newtonSteps += iters;
    }
    state.counters["steps/solve"] =  static_cast<double>(newtonSteps) / state.iterations();
  }
  
BENCHMARK_DEFINE_F(NewtonFixture, Ceres)(benchmark::State& st) {
    
    for (auto _ : st) {
        // reset parameters (the block is x_ceres itself)
        x_ceres[0] = -0.5;
        x_ceres[1] =  0.5;

        ceres::Solve(opts, &problem, &summary);
        benchmark::DoNotOptimize(x_ceres);
    }

}


  //registrations that exercise the same body differently
BENCHMARK_REGISTER_F(NewtonFixture, Solve)->Arg(1)->Unit(benchmark::kNanosecond);//Base Newton
  
BENCHMARK_REGISTER_F(NewtonFixture, Solve)->Arg(5)->Unit(benchmark::kNanosecond);//Modified Newton

BENCHMARK_REGISTER_F(NewtonFixture, Ceres)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();