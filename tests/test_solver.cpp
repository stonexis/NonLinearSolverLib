#include <ceres/ceres.h>
#include "test_fixture.hpp"

TEST_F(NewtonFixture, NewtonModified) {

    double epsilon = 1e-12;
    auto tools = ToolsWithJ {*Jptr};
    auto x = tools.newton_modified().solve(*Fptr, *x0ptr, options);
    ceres::Solve(opts, &problem, &summary); //inplace method
    for(std::size_t i = 0; i < N; i++)
    EXPECT_TRUE(abs(x.first[i] - x_ceres[i]) < epsilon) << "solverNewtonModified() gave wrong result:\n"
                                              << "Expected: " << x_ceres[i] << "\n"
                                              << "Got     : " << x.first[i];
}

TEST_F(NewtonFixture, NewtonDiscrete) {

    double epsilon = 1e-12;
    auto tools = ToolsWithF {*Fptr};
    auto x = tools.newton_discrete().solve(*x0ptr, options);
    ceres::Solve(opts, &problem, &summary); //inplace method
    for(std::size_t i = 0; i < N; i++)
    EXPECT_TRUE(abs(x.first[i] - x_ceres[i]) < epsilon) << "solverNewtonDiscrete() gave wrong result:\n"
                                              << "Expected: " << x_ceres[i] << "\n"
                                              << "Got     : " << x.first[i];
}

TEST_F(NewtonFixture, SimpleIter) {

    double epsilon = 1e-2;
    auto tools = ToolsWithF {*Fptr};
    auto x = tools.simple_iter().solve(*x0ptr, options);
    ceres::Solve(opts, &problem, &summary); //inplace method
    for(std::size_t i = 0; i < N; i++)
    EXPECT_TRUE(abs(x.first[i] - x_ceres[i] ) < epsilon) << "solverSimpleIter() gave wrong result:\n"
                                              << "Expected: " << x_ceres[i] << "\n"
                                              << "Got     : " << x.first[i];
}