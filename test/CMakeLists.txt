add_executable(glfw-test main.cpp)

target_link_libraries(glfw-test PRIVATE 
    glfw
    OpenGL::GL
    GLEW::GLEW
) 