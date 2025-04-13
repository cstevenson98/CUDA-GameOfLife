# Project cuda-gol
cmake_minimum_required(VERSION 3.29)

# Set the project name
project(cuda-gol LANGUAGES CXX CUDA)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

add_subdirectory(test)

# # Set the source files
# set(SOURCE_FILES
#         src/main.cpp
#         src/golCUDA.cu
#         src/golPipeline.cu
#         src/openGLutils.cpp
#         src/MaxwellPipeline.cu
#         src/MaxwellPipeline.h
#         src/maxwellCUDA.cu)

# # Add ImGui library
# add_library(imgui_glfw STATIC
#         imgui/imgui.cpp
#         imgui/imgui_draw.cpp
#         imgui/imgui_demo.cpp
#         imgui/imgui_tables.cpp
#         imgui/imgui_widgets.cpp
#         imgui/backends/imgui_impl_glfw.cpp
#         imgui/backends/imgui_impl_opengl3.cpp
# )

# target_link_libraries(imgui_glfw PUBLIC glfw OpenGL::GL)
# target_include_directories(imgui_glfw
#         PUBLIC
#         ${CMAKE_CURRENT_LIST_DIR}/imgui
#         ${CMAKE_CURRENT_LIST_DIR}/imgui/backends
# )

# # Create the main executable
# add_executable(cuda-gol ${SOURCE_FILES})

# target_include_directories(cuda-gol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
# target_link_libraries(cuda-gol PRIVATE 
#     glfw 
#     OpenGL::GL 
#     GLEW::GLEW 
#     imgui_glfw
# )