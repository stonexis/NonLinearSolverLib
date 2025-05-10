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

#--- Google Benchmark ---
set(BENCHMARK_ENABLE_TESTING        OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS    OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(benchmark)

#--- Ceres ----
set(CERES_USE_CXSPARSE OFF CACHE BOOL "" FORCE)

set(BUILD_TESTING    OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  ceres
  GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
  GIT_TAG        2.2.0
)

FetchContent_MakeAvailable(ceres)


add_library(core INTERFACE)

target_include_directories(core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/structures>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nonlinear>
)

add_executable(${PROJECT_NAME}
  main.cpp
)

include_directories(${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE core PRIVATE ceres)


#--- Тесты ---
enable_testing()
include(GoogleTest)

add_executable(unit_tests
  tests/test_main.cpp
  tests/test_solver.cpp
)

target_include_directories(
  unit_tests 
  PRIVATE  
  ${CMAKE_SOURCE_DIR}/structures
  ${CMAKE_SOURCE_DIR}/nonlinear
)
target_link_libraries(unit_tests PRIVATE gtest_main core ceres)

#add_test(NAME AllTests COMMAND tests)

gtest_discover_tests(unit_tests
  PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    LABELS "unit"
)


# --- Бенчмарки ---
find_package(Threads REQUIRED)
add_executable(benchmarks
benchmarks/benchmark_solver.cpp
)
target_include_directories(
  benchmarks
  PRIVATE
  ${CMAKE_SOURCE_DIR}/structures
  ${CMAKE_SOURCE_DIR}/nonlinear
)
target_link_libraries(benchmarks
  PRIVATE
    benchmark::benchmark      # <-- this target carries include dirs
    Threads::Threads         # <-- pthread
    ceres
)

