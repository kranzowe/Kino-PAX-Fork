cmake_minimum_required(VERSION 3.5)
project(KPAX LANGUAGES CXX CUDA)

# No CMAKE_BUILD_TYPE means no optimization flags at all. Every scripts/paper/run_*.sh sets
# -DCMAKE_BUILD_TYPE=Release explicitly, but a hand-built target would otherwise get an
# unoptimized build -- default it so a bare `cmake ..` does not silently do that.
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

# Planners in this branch: KPAX, KinoPaxPlus, CountingStars (Kino-PAX#), ChainReaction, FissionReaction.
# Every other planner variant lives (unbuilt) under legacy/ -- see legacy/README.md.
set(SOURCE_FILES
    src/planners/KPAX.cu
    src/planners/KinoPaxPlus.cu
    src/planners/CountingStars.cu
    src/planners/ChainReaction.cu
    src/planners/FissionReaction.cu
    src/planners/Planner.cu
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

# --- Paper targets (sources in examples/gpu/paper/, runners in scripts/paper/; see PAPER.md) ---
add_executable(PaperBenchmarkV2 examples/gpu/paper/paper_benchmark_v2.cu)
target_link_libraries(PaperBenchmarkV2
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(PaperBenchmarkV3 examples/gpu/paper/paper_benchmark_v3.cu)
target_link_libraries(PaperBenchmarkV3
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(JetsonPaperBenchmarkV3 examples/gpu/paper/jetson_paper_benchmark_v3.cu)
target_link_libraries(JetsonPaperBenchmarkV3
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(PaperBenchmarkV3Quad examples/gpu/paper/paper_benchmark_v3_quad.cu)
target_link_libraries(PaperBenchmarkV3Quad
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(TreeCheckpointDump examples/gpu/paper/tree_checkpoint_dump.cu)
target_link_libraries(TreeCheckpointDump
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(ChainReactionBenchmark examples/gpu/paper/chain_reaction_benchmark.cu)
target_link_libraries(ChainReactionBenchmark
    PUBLIC KPAX_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

add_executable(TreeGrowthAnimationDump examples/gpu/paper/tree_growth_animation_dump.cu)
target_link_libraries(TreeGrowthAnimationDump
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
set_target_properties(KinoPaxPlusMain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PaperBenchmarkV2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PaperBenchmarkV3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(JetsonPaperBenchmarkV3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PaperBenchmarkV3Quad PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TreeCheckpointDump PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TreeGrowthAnimationDump PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ChainReactionBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -diag-suppress=20012,20013,20014,20015 --expt-relaxed-constexpr --ptxas-options=-v")
# With CMP0104 left OLD by cmake_minimum_required(3.5), CMAKE_CUDA_ARCHITECTURES is never initialized, so a target
# without an explicit CUDA_ARCHITECTURES falls back to nvcc's default (sm_52) and device-links against KPAX_lib
# (built 61/86/87) with no native code path on e.g. an Orin Nano (sm_87). Every target below sets it explicitly.
set_property(TARGET KPAX_lib PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET benchMain PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET executionTimeMain PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET KinoPaxPlusMain PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET PaperBenchmarkV2 PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET PaperBenchmarkV3 PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET JetsonPaperBenchmarkV3 PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET PaperBenchmarkV3Quad PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET TreeCheckpointDump PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET TreeGrowthAnimationDump PROPERTY CUDA_ARCHITECTURES 61 86 87)
set_property(TARGET ChainReactionBenchmark PROPERTY CUDA_ARCHITECTURES 61 86 87)
add_definitions(-DEIGEN_NO_DEBUG)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-function")
endif()
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267")
endif()
add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
add_definitions(-D_FORCE_INLINES)
