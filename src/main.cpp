#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>
#include <chrono>

#include "golPipeline.h"
#include "Utility.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

const char *WindowTitle = "Lattice Boltzmann - GPU";

///////////////////////////////////////////////////
///////////////////////////////////////////////////
unsigned int pointSize = 2;

const unsigned int threadsPerBlockX = 20;
const unsigned int blockCountX = 22;

const unsigned int threadsPerBlockY = 20;
const unsigned int blockCountY = 22;

const unsigned int widthX = threadsPerBlockX * blockCountX;
const unsigned int widthY = threadsPerBlockY * blockCountY;

size_t GOLBufferSizeUint = widthX * widthY * sizeof(unsigned int);

dim3 threadSize(threadsPerBlockX, threadsPerBlockY);
dim3 blockSize(blockCountX, blockCountY);

///////////////////////////////////////////////////
///////////////////////////////////////////////////

// Some event flags.
bool RUN_SIMULATION = false;
bool SKIP_FORWARD = false;
bool READ_INSERT = false;

// Callbacks.
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_E && action == GLFW_PRESS)
        RUN_SIMULATION = !RUN_SIMULATION;
    if (key == GLFW_KEY_F && action == GLFW_PRESS)
        SKIP_FORWARD = true;
    if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        READ_INSERT = true;
}

///////////////////////////////////////////////////

void rle2state(std::string &rle, std::vector<unsigned int> &in, int x, int y) {
    std::string nums = "1234567890";
    std::string stateString = "bo";
    char lineEnd = '$';

    int num_to_write;
    std::string currentNums = "";
    for (auto i = rle.begin(); i != rle.end(); ++i) {
        // First check if number and count them
        while ((nums.find(*i) != std::string::npos)) {
            currentNums += *i;
            ++i;
        }

        if (currentNums != "")
            num_to_write = std::stoi(currentNums);
        else
            num_to_write = 1;

        // If now an entry, append to vector
        if (*i == 'b' || *i == 'o') {
            for (int j = 0; j < num_to_write; j++)
                in.push_back((*i == 'o'));
        }
        // If an endl,
        else if (*i == '$') {
            for (int j = 0; j < x * (num_to_write - 1); j++)
                in.push_back(0);
        }
        currentNums = "";
    }
    //std::cout << currentNums;
}

///////////////////////////////////////////////////
// Vertex Shader source code
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
void main()
{
    gl_Position = vec4(aPos, 1.0);
}
)";

// Fragment Shader source code
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
)";

// Function to compile shader and check for errors
unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    return shader;
}

// Function to create shader program
unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cout << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Function to set up vertex data and buffers
unsigned int setupTriangle() {
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return VAO;
}


int main(void) {
    GLFWwindow *window;
    if (!glfwInit()) { return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    window = glfwCreateWindow(widthX * pointSize, widthY * pointSize, WindowTitle, /*glfwGetPrimaryMonitor()*/NULL,
                              NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glViewport(0, 0, widthX * pointSize, widthY * pointSize);

    if (glewInit() != GLEW_OK)
        std::cout << "Something went wrong in GLEW init" << std::endl;


    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // IF using Docking Branch

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init("#version 130");
    unsigned int shaderProgram = createShaderProgram();
    unsigned int VAO = setupTriangle();

    glfwSetKeyCallback(window, key_callback);
    {
        GoLPipeline Scene(threadSize, blockSize, widthX, widthY, pointSize);
        assert(Scene.Init());
        while (!glfwWindowShouldClose(window)) {
            GLCall(glfwPollEvents());
            if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
            {
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }


            // Rendering
            // if (RUN_SIMULATION)
            //     Scene.Update();
            // if (SKIP_FORWARD)
            //     Scene.Update();
            // SKIP_FORWARD = false;
            // if (READ_INSERT) {
            //     std::string filename;
            //     std::vector<unsigned int> state;
            //     std::cin >> filename;
            //     rle2state(filename, state, 0, 0);
            //     for (const auto &i: state)
            //         std::cout << i;
            // }
            glClear(GL_COLOR_BUFFER_BIT );

            //
            Scene.Draw();
            // In your main function, before the rendering loop

            // In your rendering loop, replace the commented Scene.Draw(); line with the following
            // glUseProgram(shaderProgram);
            // glBindVertexArray(VAO);
            // glDrawArrays(GL_TRIANGLES, 0, 3);
            // glBindVertexArray(0);
            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
            {
                static float f = 0.0f;
                static int counter = 0;

                ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

                ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)

                ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f

                if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                    counter++;
                ImGui::SameLine();
                ImGui::Text("counter = %d", counter);

                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
                ImGui::End();
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            GLCall(glfwSwapBuffers(window));
        }


    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}
