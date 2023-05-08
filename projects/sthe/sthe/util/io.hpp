#pragma once

#include <memory>
#include <string>

namespace sthe
{

std::string readFile(const std::string& t_file);
std::shared_ptr<unsigned char> readImage2D(const std::string& t_file, int& t_width, int& t_height, int& t_channels, const int t_requestedChannels = 0);
std::string getModelPath();
std::string getMeshPath();
std::string getShaderPath();
std::string getResourcePath();

}

#include "io.inl"
