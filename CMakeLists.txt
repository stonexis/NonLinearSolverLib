cmake_minimum_required(VERSION 3.14)
project(NonLinearSolver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

set(INSTALL_GTEST OFF)
set(gtest_build_tests OFF)
# --- GoogleTest ---
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

# --- Google Benchmark ---
# FetchContent_Declare(
#   benchmark
#   URL https://github.com/google/benchmark/archive/refs/heads/main.zip
#   DOWNLOAD_EXTRACT_TIMESTAMP TRUE
# )
# FetchContent_MakeAvailable(benchmark)

# FetchContent_Declare(
#   eigen
#   URL https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip
#   DOWNLOAD_EXTRACT_TIMESTAMP TRUE
# )
# FetchContent_MakeAvailable(eigen)

add_executable(${PROJECT_NAME}
  main.cpp
)

target_include_directories(
  ${PROJECT_NAME} 
  PRIVATE 
  #${eigen_SOURCE_DIR}  
  ${CMAKE_SOURCE_DIR}/structures
  ${CMAKE_SOURCE_DIR}/nonlinear
)

find_package(Ceres REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Ceres::ceres)


#--- Тесты ---
# enable_testing()

# add_executable(tests
#   tests/test_main.cpp
#   tests/test_solver.cpp
#   tests/test_utils.cpp
# )

# target_include_directories(
#   tests 
#   PRIVATE 
#   ${eigen_SOURCE_DIR}  
#   ${CMAKE_SOURCE_DIR}/matrix
#   ${CMAKE_SOURCE_DIR}/nonlinear
# )
# target_link_libraries(tests gtest_main)

# add_test(NAME AllTests COMMAND tests)

# --- Бенчмарки ---
# add_executable(benchmarks benchmark/sample_benchmark.cpp)
# target_link_libraries(benchmarks benchmark::benchmark)
