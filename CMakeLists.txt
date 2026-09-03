cmake_minimum_required(VERSION 3.5)
project(KPAX LANGUAGES CXX CUDA)

# No CMAKE_BUILD_TYPE means no optimization flags at all. Every scripts/run_*.sh sets
# -DCMAKE_BUILD_TYPE=Release explicitly, but JetsonSmokeTest is meant to be built by hand
# (see scripts/run_jetson_smoke_test.sh, which also sets it) -- default it so a bare
# `cmake ..` does not silently produce an unoptimized build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    include 
    ${EIGEN3_INCLUDE_DIRS} 
    ${BOOST_INCLUDE_DIRS}  
    ${CMAKE_SOURCE_DIR}/src 
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)

set(SOURCE_FILES
    src/planners/KPAX.cu
    src/planners/KPAXCap.cu
    src/planners/KinoPaxPlus.cu
    src/planners/originalKPAX.cu
    src/planners/originalPlanner.cu
    src/planners/KinoPaxSTAR.cu
    src/planners/KinoPaxSTARcostprune.cu
    src/planners/KinoPaxSTARWeightedCost.cu
    src/planners/KinoPaxSTARCleanCost.cu
    src/planners/KinoPaxSTARCOMBO.cu
    src/planners/CountingStars.cu
    src/planners/KinoPaxSTARTrue.cu
    src/planners/KinoPaxSTARTrueWeightedCost.cu
    src/planners/KinoPaxSTARcostprunenoseed.cu
    src/planners/KinoPaxSTARNoGoalBias.cu
    src/planners/KinoPaxSTARNoPruneNoSpatialHash.cu
    src/planners/KinoPaxSTARnoseed.cu
    src/planners/KinoPaxSTARsparsefill.cu
    src/planners/PruneKPAX.cu
    src/planners/KinoPaxPlusSpatialHash.cu
    src/planners/Planner.cu
    src/ReKino/ReKino.cu
    src/ReKino/ReKinoLite.cu
    src/collisionCheck/collisionCheck.cu
    src/collisionCheck/spatialHash.cu
    src/graphs/Graph.cu
    src/graphs/KinoPaxPlusRegions.cu
    src/helper/helper.cu
    src/statePropagator/statePropagator.cu
    src/statePropagator/statePropagatorSpatialHash.cu
)
add_library(KPAX_lib ${SOURCE_FILES})


if (OpenMP_CXX_FOUND)
    target_link_libraries(KPAX_lib PUBLIC OpenMP::OpenMP_CXX)
endif()


# -- GPU EXECUTABLES --
add_executable(main examples/gpu/main.cu)
target_link_libraries(main
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(benchMain examples/gpu/benchMain.cu)
target_link_libraries(benchMain
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(executionTimeMain examples/gpu/executionTimeMain.cu)
target_link_libraries(executionTimeMain
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(ReKinoMain examples/gpu/rekino_main.cu)
target_link_libraries(ReKinoMain
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxPlusMain examples/gpu/kinopaxplus_main.cu)
target_link_libraries(KinoPaxPlusMain
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(ComprehensiveBenchmark examples/gpu/comprehensive_benchmark.cu)
target_link_libraries(ComprehensiveBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxPlusComprehensiveBenchmark examples/gpu/kinopaxplus_comprehensive_benchmark.cu)
target_link_libraries(KinoPaxPlusComprehensiveBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(PlannerComparisonBenchmark examples/gpu/planner_comparison_benchmark.cu)
target_link_libraries(PlannerComparisonBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(JetsonSmokeTest examples/gpu/jetson_smoke_test.cu)
target_link_libraries(JetsonSmokeTest
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxPlusDeltaBenchmark examples/gpu/kinopaxplus_delta_benchmark.cu)
target_link_libraries(KinoPaxPlusDeltaBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxPlusSHBenchmark examples/gpu/kinopaxplus_sh_benchmark.cu)
target_link_libraries(KinoPaxPlusSHBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxStarLargeBenchmark examples/gpu/kinopaxstar_large_benchmark.cu)
target_link_libraries(KinoPaxStarLargeBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxStarCostBenchmark examples/gpu/kinopaxstar_cost_benchmark.cu)
target_link_libraries(KinoPaxStarCostBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxStarCostTuningSweep examples/gpu/kinopaxstar_cost_tuning_sweep.cu)
target_link_libraries(KinoPaxStarCostTuningSweep
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxStarComboTuningSweep examples/gpu/kinopaxstar_combo_tuning_sweep.cu)
target_link_libraries(KinoPaxStarComboTuningSweep
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(CountingStarsSweep examples/gpu/countingstars_sweep.cu)
target_link_libraries(CountingStarsSweep
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxStarComparison examples/gpu/kinopaxstar_comparison.cu)
target_link_libraries(KinoPaxStarComparison
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(KinoPaxStarAcceptBreakdown examples/gpu/kinopaxstar_accept_breakdown.cu)
target_link_libraries(KinoPaxStarAcceptBreakdown
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(TreeGrowthDump examples/gpu/tree_growth_dump.cu)
target_link_libraries(TreeGrowthDump
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

set_target_properties(KPAX_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(benchMain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(executionTimeMain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ReKinoMain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxPlusMain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ComprehensiveBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxPlusComprehensiveBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PlannerComparisonBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(JetsonSmokeTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxPlusDeltaBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxPlusSHBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxStarLargeBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxStarCostBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxStarCostTuningSweep PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxStarComboTuningSweep PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CountingStarsSweep PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxStarComparison PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KinoPaxStarAcceptBreakdown PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TreeGrowthDump PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -diag-suppress=20012,20013,20014,20015 --expt-relaxed-constexpr --ptxas-options=-v")
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET benchMain PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET executionTimeMain PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KPAX_lib PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET ReKinoMain PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxPlusMain PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET ComprehensiveBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxPlusComprehensiveBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET PlannerComparisonBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET JetsonSmokeTest PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxPlusDeltaBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxPlusSHBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxStarLargeBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
# These seven had CUDA_SEPARABLE_COMPILATION set above but no CUDA_ARCHITECTURES here, so with
# CMP0104 left OLD by cmake_minimum_required(3.5), CMAKE_CUDA_ARCHITECTURES is never initialized
# and they fell back to nvcc's default (sm_52) -- device-linked against KPAX_lib, which IS built
# 61/86/87. On an Orin Nano (sm_87) that meant no native code path in these binaries at all.
set_property(TARGET KinoPaxStarCostBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxStarCostTuningSweep PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxStarComboTuningSweep PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET CountingStarsSweep PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxStarComparison PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxStarAcceptBreakdown PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET TreeGrowthDump PROPERTY CUDA_ARCHITECTURES 61 86 87)
add_definitions(-DEIGEN_NO_DEBUG)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-function")
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267")
endif()
add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
add_definitions(-D_FORCE_INLINES)