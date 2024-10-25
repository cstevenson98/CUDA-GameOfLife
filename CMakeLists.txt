# Project cuda-gol
cmake_minimum_required(VERSION 3.10)
# SET DEFAULT CUDA ARCHITECTURE

# set cuda compiler for .cu files
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# set the cuda architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Set the project name
project(cuda-gol LANGUAGES CXX CUDA)

# include cuda headers
include_directories(/usr/local/cuda-12.6/include)

set( _glfw3_HEADER_SEARCH_DIRS
        "/usr/include"
        "/usr/local/include"
        "${CMAKE_SOURCE_DIR}/includes"
        "C:/Program Files (x86)/glfw/include" )
set( _glfw3_LIB_SEARCH_DIRS
        "/usr/lib"
        "/usr/local/lib"
        "${CMAKE_SOURCE_DIR}/lib"
        "C:/Program Files (x86)/glfw/lib-msvc110" )

# Check environment for root search directory
set( _glfw3_ENV_ROOT $ENV{GLFW3_ROOT} )
if( NOT GLFW3_ROOT AND _glfw3_ENV_ROOT )
    set(GLFW3_ROOT ${_glfw3_ENV_ROOT} )
endif()

# Put user specified location at beginning of search
if( GLFW3_ROOT )
    list( INSERT _glfw3_HEADER_SEARCH_DIRS 0 "${GLFW3_ROOT}/include" )
    list( INSERT _glfw3_LIB_SEARCH_DIRS 0 "${GLFW3_ROOT}/lib" )
endif()

# Search for the header
FIND_PATH(GLFW3_INCLUDE_DIR "GLFW/glfw3.h"
        PATHS ${_glfw3_HEADER_SEARCH_DIRS} )

# Search for the library
FIND_LIBRARY(GLFW3_LIBRARY NAMES glfw3 glfw
        PATHS ${_glfw3_LIB_SEARCH_DIRS} )
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLFW3 DEFAULT_MSG
        GLFW3_LIBRARY GLFW3_INCLUDE_DIR)

message(STATUS "GLFW3_LIBRARY: ${GLFW3_LIBRARY}")

# Set the source files
set(SOURCE_FILES
        src/main.cpp
        src/golCUDA.cu
        src/golPipeline.cu
        src/openGLutils.cpp)

set(OpenGL_GL_PREFERENCE "GLVND")

# Set the libraries
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Set the executable
include_directories(src)
add_executable(cuda-gol ${SOURCE_FILES})
target_link_libraries(cuda-gol ${GLFW3_LIBRARY} ${OPENGL_LIBRARIES} GLEW::GLEW)