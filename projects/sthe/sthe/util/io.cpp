#define STB_IMAGE_IMPLEMENTATION

#include "io.hpp"
#include <sthe/config/debug.hpp>
#include <stb_image.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>

namespace sthe
{

std::string readFile(const std::string& t_file)
{
	std::ifstream stream{ t_file };
	std::string source;

	if (stream.is_open())
	{
		std::string line;

		while (!stream.eof())
		{
			std::getline(stream, line);
			source += line + "\n";
		}

		stream.close();
	}

	return source;
}

std::shared_ptr<unsigned char> readImage2D(const std::string& t_file, int& t_width, int& t_height, int& t_channels, const int t_requestedChannels)
{
	stbi_set_flip_vertically_on_load(1);
	stbi_uc* const source{ stbi_load(t_file.c_str(), &t_width, &t_height, &t_channels, t_requestedChannels) };
	
	STHE_ASSERT(source != nullptr, "Failed to read image");

	return std::shared_ptr<unsigned char>{ source, free };
}

}
