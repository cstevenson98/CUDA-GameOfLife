# Project cuda-gol
cmake_minimum_required(VERSION 3.22)
# SET DEFAULT CUDA ARCHITECTURE

# set cuda compiler for .cu files
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# set the cuda architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Set the project name
project(cuda-gol LANGUAGES CXX CUDA)

# include cuda headers
include_directories(/usr/local/cuda-12.6/include)
include(cmake/get_glfw3.cmake)

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