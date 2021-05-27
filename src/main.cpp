#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>
#include <chrono>

#include "lbPipeline.h"

const char* WindowTitle = "Lattice Boltzmann - GPU";

///////////////////////////////////////////////////
///////////////////////////////////////////////////
unsigned int pointSize = 2;

const unsigned int threadsPerBlockX = 20;
const unsigned int blockCountX = 30;

const unsigned int threadsPerBlockY = 20;
const unsigned int blockCountY = 15;

const unsigned int widthX = threadsPerBlockX * blockCountX;
const unsigned int widthY = threadsPerBlockY * blockCountY;

size_t GOLBufferSizeUint = widthX * widthY * sizeof(unsigned int);

dim3 threadSize(threadsPerBlockX, threadsPerBlockY);
dim3 blockSize(blockCountX, blockCountY);

///////////////////////////////////////////////////
///////////////////////////////////////////////////

int main(void)
{
	GLFWwindow* window;
	if (!glfwInit()) { return -1; }
	window = glfwCreateWindow(widthX*pointSize, widthY*pointSize, WindowTitle, NULL, NULL);
	if (!window){ glfwTerminate(); return -1; }

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (glewInit() != GLEW_OK)
		std::cout << "Something went wrong in GLEW init" << std::endl;

    {
        LatticeBoltzmannPipeline Scene(threadSize, blockSize, widthX, widthY, pointSize);
        assert( Scene.Init() );
		while (!glfwWindowShouldClose(window))
		{
			Scene.Draw();
			GLCall( glfwSwapBuffers(window) );
			GLCall( glfwPollEvents() );
		}
	}

	glfwTerminate();
	return 0;
}
