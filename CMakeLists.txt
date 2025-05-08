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

add_library(core INTERFACE)

# Where users can #include your headers
target_include_directories(core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/structures>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nonlinear>
)

add_executable(${PROJECT_NAME}
  main.cpp
)

find_package(Ceres REQUIRED)
find_package(TBB CONFIG REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE core PRIVATE Ceres::ceres PRIVATE TBB::tbb)


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
target_link_libraries(unit_tests PRIVATE gtest_main core Ceres::ceres)

#add_test(NAME AllTests COMMAND tests)

gtest_discover_tests(unit_tests
  PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    LABELS "unit"
)


# --- Бенчмарки ---

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
    Ceres::ceres
)

