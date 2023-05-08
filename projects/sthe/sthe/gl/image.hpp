#pragma once

#include "handle.hpp"
#include <glad/glad.h>
#include <memory>
#include <string>

namespace sthe
{
namespace gl
{

std::shared_ptr<unsigned char> readImage2D(const std::string& t_file, int& t_width, int& t_height, GLenum& t_internalFormat, GLenum& t_pixelFormat, GLenum& t_pixelType);
int getMipCapacity(const int t_width, const int t_height);
int getMipCapacity(const int t_width, const int t_height, const int t_depth);

class Image : public Handle
{
public:
	// Constructors
	Image() = default;
	Image(const Image& t_image) = delete;
	Image(Image&& t_image) = delete;

	// Destructor
	virtual ~Image() = default;

	// Operators
	Image& operator=(const Image& t_image) = delete;
	Image& operator=(Image&& t_image) = delete;

	// Functionality
	virtual void bind() const = 0;
	virtual void release() = 0;

	// Getters
	virtual GLenum getTarget() const = 0;
	virtual int getWidth() const;
	virtual int getHeight() const;
	virtual int getDepth() const;
	virtual GLenum getFormat() const = 0;
	virtual int getLayerCount() const;
	virtual int getMipCount() const;
	bool hasStorage() const;
	bool isLayered() const;
	bool isMipmapped() const;
};

}
}
