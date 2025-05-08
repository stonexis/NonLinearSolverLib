#include <ceres/ceres.h>
#include "test_fixture.hpp"

TEST_F(NewtonFixture, NewtonCorrectly) {

    double epsilon = 1e-9;
    auto tools = Tools {*Jptr};
    auto x = tools.newton_modified().solve(*Fptr, *x0ptr);
    ceres::Solve(opts, &problem, &summary); //inplace method
    for(std::size_t i = 0; i < N; i++)
    EXPECT_TRUE((x.first[i] - x_ceres[i]) < epsilon) << "solverNewton() gave wrong result:\n"
                                              << "Expected: " << x_ceres[i] << "\n"
                                              << "Got     : " << x.first[i];
}
