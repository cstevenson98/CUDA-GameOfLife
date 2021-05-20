// C Stevenson 2021

#include <GL/glew.h>	
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"

#include <cuda.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string> 
#include <chrono>

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "VertexBufferLayout.h"
#include "VertexArray.h"
#include "Shader.h"
#include "Renderer.h"
#include "Vendor/imgui/imgui.h"
#include "Vendor/imgui/imgui_impl_glfw.h"
#include "Vendor/imgui/imgui_impl_opengl3.h"

#include "GoLCudaBuffer.h"
#include "ColorData.h"
#include "Utility.h"
#include "Fluid2D.h"

/* =========================================================================== */

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

/* =========================================================================== */
/* =========================================================================== */
/* =========================================================================== */


#define IG_POINT_SIZE 1
#define DENSITY 0.75

#define xmin -1.0
#define xmax 1.0
#define ymin -1.0
#define ymax 1.0

#define threadsPerBlock 22
#define blockCount 40

unsigned int fullCellWidth = threadsPerBlock * blockCount;

size_t GOLBufferSizeUint = fullCellWidth * fullCellWidth * sizeof(unsigned int);

dim3 threadSize(threadsPerBlock, threadsPerBlock);
dim3 blockSize(blockCount, blockCount);

/* =========================================================================== */
/* =========================================================================== */
/* =========================================================================== */

int main(void)
{
	GLFWwindow* window;
	// Initialize the library
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a windowed mode window and its OpenGL context
	std::cout << "Window size = " 
		<< fullCellWidth * IG_POINT_SIZE << " x " << fullCellWidth * IG_POINT_SIZE
		<< std::endl;

	window = glfwCreateWindow(fullCellWidth * (IG_POINT_SIZE), fullCellWidth * (IG_POINT_SIZE), 
								"Fucking about", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	// Make the window's context current 
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (glewInit() != GLEW_OK)
		std::cout << "Something went wrong in GLEW init" << std::endl;

	glClearColor(.0, .0, .0, .0);

	// New scope so the glfw doesn't remove context before stack
	// is cleaned up
	{
		// Initialise rendering
		Renderer renderer;

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui::StyleColorsDark();
		ImGui_ImplOpenGL3_Init((char*)glGetString(330));

		std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point new_time;

		float dt = std::chrono::duration_cast<std::chrono::microseconds>
						(std::chrono::steady_clock::now() - current_time).count();
		float t = 0.;
		float step = 1.;

		Fluid2D fluid(threadSize, blockSize, fullCellWidth);
		while (!glfwWindowShouldClose(window))
		{
			dt = std::chrono::duration_cast<std::chrono::microseconds>
						(std::chrono::steady_clock::now() - current_time).count();
			current_time = std::chrono::steady_clock::now();

			t += step * dt / 1000000.0;

			fluid.next(t);

			renderer.Clear();
			fluid.draw(renderer, fullCellWidth);

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			// ImGui scope -- not quite sure 
			// why this is completely necessary
			{
				static float f = 0.1;
				static float density = 0.5;
				ImGui::Begin("Cuda/GL: Game of Life");

				ImGui::SliderFloat("Speed", &f, 0.1f, 3.0f);
				step = f;

				ImGui::SliderFloat("Density", &density, 0.0f, 1.0f);

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 
							1000.0f / ImGui::GetIO().Framerate, 
							ImGui::GetIO().Framerate);

				ImGui::End();
			}
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			GLCall( glfwSwapBuffers(window) );
			GLCall( glfwPollEvents() );
		}
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	glfwTerminate();
	return 0;
}
