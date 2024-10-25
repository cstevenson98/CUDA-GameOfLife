#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <cassert>
#include <chrono>

#include "golPipeline.h"
#include "Utility.h"

const char* WindowTitle = "Lattice Boltzmann - GPU";

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
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_E && action == GLFW_PRESS)
        RUN_SIMULATION = !RUN_SIMULATION;
	if (key == GLFW_KEY_F && action == GLFW_PRESS)
		SKIP_FORWARD = true;
	if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
		READ_INSERT = true;
}

///////////////////////////////////////////////////

void rle2state(std::string& rle, std::vector<unsigned int> &in, int x, int y)
{
	std::string nums = "1234567890";
	std::string stateString = "bo";
	char lineEnd = '$';

	int num_to_write; 
	std::string currentNums = "";
	for (auto i = rle.begin(); i != rle.end(); ++i)
	{
		// First check if number and count them
		while( (nums.find(*i) != std::string::npos) )
		{
			currentNums += *i;
			++i;
		}

		if (currentNums != "")
			num_to_write = std::stoi(currentNums);
		else
			num_to_write = 1;

		// If now an entry, append to vector
		if (*i == 'b' || *i == 'o')
		{
			for (int j = 0; j < num_to_write; j++)
				in.push_back( (*i == 'o') );
		} 
		// If an endl,
		else if (*i == '$') 
		{
			for (int j = 0; j < x * (num_to_write-1); j++)
				in.push_back( 0 );
		}
		currentNums = "";
	}
	//std::cout << currentNums;
}

///////////////////////////////////////////////////

int main(void)
{
	GLFWwindow* window;
	if (!glfwInit()) { return -1; }

	window = glfwCreateWindow(widthX*pointSize, widthY*pointSize, WindowTitle, /*glfwGetPrimaryMonitor()*/NULL, NULL);
	if (!window){ glfwTerminate(); return -1; }

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glViewport(0, 0, widthX*pointSize, widthY*pointSize);

	if (glewInit() != GLEW_OK)
		std::cout << "Something went wrong in GLEW init" << std::endl;

   
	glfwSetKeyCallback(window, key_callback);

	{
        GoLPipeline Scene(threadSize, blockSize, widthX, widthY, pointSize);
        assert( Scene.Init() );
		while (!glfwWindowShouldClose(window))
		{
			if (RUN_SIMULATION)
				Scene.Update();
			if (SKIP_FORWARD)
				Scene.Update();
				SKIP_FORWARD = false;
			if (READ_INSERT)
			{
				std::string filename;
				std::vector<unsigned int> state;
				std::cin >> filename;
				rle2state(filename, state, 0, 0);
				for(auto &i : state)
					std::cout << i;
			}
			Scene.Draw();
			
			GLCall( glfwSwapBuffers(window) );
			GLCall( glfwPollEvents() );
		}
	}

	glfwTerminate();
	return 0; 
}
