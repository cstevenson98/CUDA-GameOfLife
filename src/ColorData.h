#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "Utility.h"

struct ColorData
{
	ColorData(std::string& file);

	std::string fileName;
	std::string schemeName;
	std::vector<float> data;
};

ColorData::ColorData(std::string& file)
{

	fileName = file;
	schemeName = fileName.substr(0, fileName.length() - 4);

	data = csvToVectf(file);
}