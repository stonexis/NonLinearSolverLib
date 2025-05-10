#include <benchmark/benchmark.h>
#include "nlslib.hpp"
#include "benchmark_fixture.hpp"            

//Benchmark body, defined once
BENCHMARK_DEFINE_F(NewtonFixture, Newton)(benchmark::State& state) {
    options.k_refresh_J = static_cast<std::size_t>(state.range(0));// param
    std::size_t newtonSteps = 0;
    for (auto _ : state) {
        VecX x = *x0ptr; // fresh copy each run
        auto [sol, iters] = nls::newton_modified(*Jptr, *Fptr, x, options);
        newtonSteps += iters;
    }
    state.counters["steps/solve"] =  static_cast<double>(newtonSteps) / state.iterations();
}

BENCHMARK_DEFINE_F(NewtonFixture, NewtonDiscrete)(benchmark::State& state) {
    std::size_t newtonSteps = 0;
    for (auto _ : state) {
        VecX x = *x0ptr; // fresh copy each run
        auto [sol, iters] = nls::newton_discrete( *Fptr, x, options);
        newtonSteps += iters;
    }
    state.counters["steps/solve"] =  static_cast<double>(newtonSteps) / state.iterations();
}

static const VectorX<double, 4> lambdas = {0.1, 0.5, 0.6, 0.9};

BENCHMARK_DEFINE_F(NewtonFixture, SimpleIter)(benchmark::State& state) {
    std::size_t Steps = 0;
    options.lambdas[0] = lambdas[static_cast<std::size_t>(state.range(0))];// param;
    options.lambdas[1] = -lambdas[static_cast<std::size_t>(state.range(0))];
    options.max_iter = 300;
    for (auto _ : state) {
        VecX x = *x0ptr; // fresh copy each run
        auto [sol, iters] = nls::simple_iter( *Fptr, x, options);
        Steps += iters;
    }
    state.counters["steps/solve"] =  static_cast<double>(Steps) / state.iterations();
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
BENCHMARK_REGISTER_F(NewtonFixture, Newton)->Arg(1)->Unit(benchmark::kNanosecond);//Base Newton
  
BENCHMARK_REGISTER_F(NewtonFixture, Newton)->Arg(5)->Unit(benchmark::kNanosecond);//Modified Newton

BENCHMARK_REGISTER_F(NewtonFixture, NewtonDiscrete)->Unit(benchmark::kNanosecond); //Discrete Newton

BENCHMARK_REGISTER_F(NewtonFixture, Ceres)->Unit(benchmark::kNanosecond);

BENCHMARK_REGISTER_F(NewtonFixture, SimpleIter)->Arg(0)->Unit(benchmark::kNanosecond);//Simple Iter lambda = 0.1

BENCHMARK_REGISTER_F(NewtonFixture, SimpleIter)->Arg(1)->Unit(benchmark::kNanosecond);//Simple Iter lambda = 0.5

BENCHMARK_REGISTER_F(NewtonFixture, SimpleIter)->Arg(2)->Unit(benchmark::kNanosecond);//Simple Iter lambda = 0.6

BENCHMARK_REGISTER_F(NewtonFixture, SimpleIter)->Arg(3)->Unit(benchmark::kNanosecond);//Simple Iter lambda = 0.9

BENCHMARK_MAIN();