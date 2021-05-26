// C Stevenson 2021

#pragma once

/*

	FILE READING AND WRITING

*/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CSV -> vect<float>
std::vector<float> csvToVectf(std::string& file)
{
	std::ifstream myFile(file); 
	std::vector<float> out;
	std::string line;

	if(!myFile.is_open()) 
		throw std::runtime_error("Could not open color data file requested: " + file);

	// Loading line-by-line the .csv file:
	while (std::getline(myFile, line))
	{
		// Scan the line and separate by ','
		size_t pos = 0;
		while ((pos = line.find(",")) != std::string::npos)
		{
			out.push_back(stof(line.substr(0, pos)));
			line.erase(0, pos + 1);
		}
		out.push_back(stof(line.substr(0, pos)));
	}
	myFile.close();

	return out;
}

// CSV -> vect<int, uint, complex>