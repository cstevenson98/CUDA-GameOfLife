# Project cuda-gol
cmake_minimum_required(VERSION 3.22)
include(FetchContent)

# SET DEFAULT CUDA ARCHITECTURE

# set cuda compiler for .cu files
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# set the cuda architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Set the project name
project(cuda-gol LANGUAGES CXX CUDA)

message(STATUS "GLFW3_LIBRARY: ${GLFW3_LIBRARY}")

# Set the source files
set(SOURCE_FILES
        src/main.cpp
        src/golCUDA.cu
        src/golPipeline.cu
        src/openGLutils.cpp)

set(OpenGL_GL_PREFERENCE "GLVND")

fetchcontent_populate(imgui
        URL https://github.com/ocornut/imgui/archive/docking.zip
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui
)

add_library(imgui_glfw STATIC
        imgui/imgui.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_demo.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_widgets.cpp

        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_opengl3.cpp
)

target_link_libraries(imgui_glfw PUBLIC glfw ${OPENGL_LIBRARIES})
target_include_directories(imgui_glfw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/imgui
        ${CMAKE_CURRENT_LIST_DIR}/imgui/backends
)
# Set the libraries
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
include(cmake/get_glfw3.cmake)
find_package(glfw3 REQUIRED)
include_directories(/usr/local/cuda-12.6/include)
add_executable(cuda-gol ${SOURCE_FILES})

target_include_directories(cuda-gol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
target_link_libraries(cuda-gol ${GLFW3_LIBRARY} ${OPENGL_LIBRARIES} GLEW::GLEW imgui_glfw)