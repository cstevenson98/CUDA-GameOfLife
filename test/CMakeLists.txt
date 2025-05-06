add_executable(glfw-test main.cpp)

target_link_libraries(glfw-test PRIVATE 
    glfw
    OpenGL::GL
    GLEW::GLEW
    imgui_glfw
) 

target_include_directories(glfw-test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/imgui
    ${CMAKE_CURRENT_LIST_DIR}/imgui/backends
)