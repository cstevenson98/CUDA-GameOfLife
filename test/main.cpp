#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Error callback for GLFW
void error_callback(int error, const char* description) {
  std::cerr << "Error: " << description << std::endl;
}

// Function to load shader source from file
std::string loadShaderSource(const char* filePath) {
  std::filesystem::path home = std::getenv("HOME");
  std::filesystem::path path =
      home / "dev/CUDA-GameOfLife/test/shaders" / filePath;
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Failed to open shader file: " << path << std::endl;
    return "";
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

// Function to compile shader
GLuint compileShader(GLenum type, const char* source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  // Check for compilation errors
  GLint success;
  GLchar infoLog[512];
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(shader, 512, nullptr, infoLog);
    std::cerr << "Shader compilation error: " << infoLog << std::endl;
  }

  return shader;
}

int main() {
  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  // Set error callback
  glfwSetErrorCallback(error_callback);

  // Configure GLFW
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create a window
  GLFWwindow* window =
      glfwCreateWindow(800, 600, "Strobing Triangle", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  // Make the window's context current
  glfwMakeContextCurrent(window);

  // Initialize GLEW
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    glfwTerminate();
    return -1;
  }

  // Print OpenGL version
  std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

  // Load and compile shaders
  std::string vertexSource = loadShaderSource("vertex.glsl");
  std::string fragmentSource = loadShaderSource("fragment.glsl");

  if (vertexSource.empty() || fragmentSource.empty()) {
    std::cerr << "Failed to load shader sources" << std::endl;
    glfwTerminate();
    return -1;
  }

  GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource.c_str());
  GLuint fragmentShader =
      compileShader(GL_FRAGMENT_SHADER, fragmentSource.c_str());

  // Create shader program
  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  // Check for linking errors
  GLint success;
  GLchar infoLog[512];
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
    std::cerr << "Shader program linking error: " << infoLog << std::endl;
  }

  // Clean up shaders
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // Triangle vertices
  float vertices[] = {-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f};

  // Create and bind VAO and VBO
  GLuint VAO, VBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // Set vertex attribute pointers
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // Unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // Get uniform location
  GLint timeLoc = glGetUniformLocation(shaderProgram, "time");

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    // Clear the screen
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Use shader program
    glUseProgram(shaderProgram);

    // Update time uniform
    float time = glfwGetTime();
    glUniform1f(timeLoc, time);

    // Draw triangle
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Clean up
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}