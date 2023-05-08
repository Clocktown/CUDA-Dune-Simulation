#include "image.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <glad/glad.h>
#include <memory>
#include <string>

namespace sthe
{
namespace gl
{

std::shared_ptr<unsigned char> readImage2D(const std::string& t_file, int& t_width, int& t_height, GLenum& t_internalFormat, GLenum& t_pixelFormat, GLenum& t_pixelType)
{
	int channels;
	const std::shared_ptr<unsigned char> source{ sthe::readImage2D(t_file, t_width, t_height, channels) };

	switch (channels)
	{
	case 1:
		t_internalFormat = GL_R8;
		t_pixelFormat = GL_RED;
		break;
	case 2:
		t_internalFormat = GL_RG8;
		t_pixelFormat = GL_RG;
		break;
	case 3:
		t_internalFormat = GL_RGB8;
		t_pixelFormat = GL_RGB;
		break;
	case 4:
		t_internalFormat = GL_RGBA8;
		t_pixelFormat = GL_RGBA;
		break;
	default:
		STHE_ERROR("Channels must be between 1 and 4");

		break;
	}

	t_pixelType = GL_UNSIGNED_BYTE;

	return source;
}

int getMipCapacity(const int t_width, const int t_height)
{
	return getMipCapacity(t_width, t_height, 0);
}

int getMipCapacity(const int t_width, const int t_height, const int t_depth)
{
	return 1 + static_cast<int>(floor(log2f(static_cast<float>(std::max(t_width, std::max(t_height, t_depth))))));
}

// Getters
int Image::getWidth() const
{
    return 0;
}

int Image::getHeight() const
{
    return 0;
}

int Image::getDepth() const
{
    return 0;
}

int Image::getLayerCount() const
{
    return 1;
}

int Image::getMipCount() const
{
    return 1;
}

bool Image::hasStorage() const
{
    return getWidth() > 0 || getHeight() > 0 || getDepth() > 0;
}

bool Image::isLayered() const
{
	return getLayerCount() > 1;
}

bool Image::isMipmapped() const
{
    return getMipCount() > 1;
}

}
}
