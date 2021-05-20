#pragma once

#include <string>
#include <vector>

struct RunLengthEncode
{
	RunLengthEncode(std::string& in)
	{
		rle = in;

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
					data.push_back( (*i == 'o') );
			} 
			// If an endl,
			else if (*i == '$') 
			{
				for (int j = 0; j < x * (num_to_write-1); j++)
					data.push_back( 0 );
			}
			currentNums = "";
		}
	};

	std::string rle;
	std::string nums = "1234567890";
	std::vector<unsigned int> data;

	int x, y;
};


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
