cmake_minimum_required(VERSION 3.15)
project(machtopo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------
# Output directory
# -----------------------------
set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})

# -----------------------------
# Python and pybind11
# -----------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# -----------------------------
# OpenMP
# -----------------------------
find_package(OpenMP REQUIRED)

# -----------------------------
# Eigen: first try system, fallback to header-only
# -----------------------------
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching header-only version...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
    include_directories(${eigen_SOURCE_DIR})
else()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# -----------------------------
# Include directories
# -----------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)

# -----------------------------
# Collect all source files
# -----------------------------
file(GLOB_RECURSE MODULE_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# -----------------------------
# Build Python module
# -----------------------------
pybind11_add_module(machtopo ${MODULE_SOURCES})
target_link_libraries(machtopo PRIVATE OpenMP::OpenMP_CXX)

# Keep prefix empty for Python module
set_target_properties(machtopo PROPERTIES PREFIX "")

# -----------------------------
# Compiler optimizations
# -----------------------------
if(MSVC)

    # Place .pyd directly in bin/ for Release
    set_target_properties(machtopo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIR}
    )

    # High-performance MSVC preset (Release only)
    target_compile_options(machtopo PRIVATE
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/fp:fast>
        $<$<CONFIG:Release>:/arch:AVX2>
    )

    target_link_options(machtopo PRIVATE
        $<$<CONFIG:Release>:/LTCG>
    )

    # Disable expensive safety/debug features
    target_compile_definitions(machtopo PRIVATE
        $<$<CONFIG:Release>:_ITERATOR_DEBUG_LEVEL=0>
        $<$<CONFIG:Release>:_CRT_SECURE_NO_WARNINGS>
        $<$<CONFIG:Release>:_SCL_SECURE_NO_WARNINGS>
    )

else()

    # GCC / Clang
    target_compile_options(machtopo PRIVATE
        -O3
        -march=native
        -fopenmp
    )
    
    # Optional: portable version
    # target_compile_options(machtopo PRIVATE -O3 -march=x86-64-v3 -fopenmp)
    
    # Optional LTO
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-flto=4" COMPILER_SUPPORTS_FLTO4)
    if(COMPILER_SUPPORTS_FLTO4)
        message(STATUS "Enabling LTO with 4 threads")
        set_target_properties(machtopo PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        target_compile_options(machtopo PRIVATE -flto=4)
        target_link_options(machtopo PRIVATE -flto=4)
    endif()

endif()

# -----------------------------
# Eigen performance tuning
# -----------------------------
target_compile_definitions(machtopo PRIVATE
    EIGEN_NO_DEBUG
    EIGEN_FAST_MATH
)

