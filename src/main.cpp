#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>
#include <chrono>

#include "golPipeline.h"
#include "MaxwellPipeline.h"
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

enum sim {
    GOL,
    MAXWELL
};

sim currentSim = GOL;

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
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // IF using Docking Branch

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init("#version 130");

    glfwSetKeyCallback(window, nullptr); // key_callback);
    {
        GoLPipeline gol_pipeline(threadSize, blockSize, widthX, widthY, pointSize);
        MaxwellPipeline maxwell_pipeline(threadSize, blockSize, widthX, widthY);
        assert(maxwell_pipeline.Init());
        assert(gol_pipeline.Init());
        auto generations = new int(1);
        while (!glfwWindowShouldClose(window)) {
            GLCall(glfwPollEvents());
            if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            if (RUN_SIMULATION) {
                switch (currentSim) {
                    case GOL:
                        gol_pipeline.Update(*generations);
                        break;
                    case MAXWELL:
                        maxwell_pipeline.UpdateField();
                        break;
                }
            }

            switch (currentSim) {
                case GOL:
                    gol_pipeline.Draw();
                    break;
                case MAXWELL:
                    maxwell_pipeline.Draw();
                    break;
            }

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame(); {
                static float f = 0.0f;
                static int counter = 0;

                // tabs

                ImGui::Begin("Settings"); // Create a window called "Hello, world!" and append into it.
                ImGui::Text("Choose simulation:"); // Display some text (you can use a format strings too)
                ImGui::RadioButton("Game of Life", (int *) &currentSim, GOL);
                ImGui::RadioButton("Maxwell", (int *) &currentSim, MAXWELL);

                if (ImGui::Button(RUN_SIMULATION ? "Pause" : "Run") || ImGui::IsKeyPressed(ImGuiKey_E))
                    RUN_SIMULATION = !RUN_SIMULATION;

                // Reset button
                if (ImGui::Button("Reset")) {
                    gol_pipeline.Reset();
                }

                // Number of generations
                ImGui::SliderInt("Generations", generations, 1, 499);

                // Frames per second
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                            1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
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
