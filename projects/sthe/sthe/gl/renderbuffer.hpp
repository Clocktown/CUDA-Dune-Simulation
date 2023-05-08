#pragma once

#include "image.hpp"
#include <glad/glad.h>

namespace sthe
{
namespace gl
{

class Renderbuffer : public Image
{
public:
	// Constructors
	Renderbuffer();
	Renderbuffer(const int t_width, const int t_height, const GLenum t_format);
	Renderbuffer(const Renderbuffer& t_renderbuffer) noexcept;
	Renderbuffer(Renderbuffer&& t_renderbuffer) noexcept;

	// Destructor
	~Renderbuffer();

	// Operators
	Renderbuffer& operator=(const Renderbuffer& t_renderbuffer) noexcept;
	Renderbuffer& operator=(Renderbuffer&& t_renderbuffer) noexcept;

	// Functionality
	void bind() const override;
	void reinitialize(const int t_width, const int t_height, const GLenum t_format);
	void release() override;

	// Getters
	GLuint getHandle() const override;
	GLenum getTarget() const override;
	int getWidth() const override;
	int getHeight() const override;
	GLenum getFormat() const override;
private:
	// Attributes
	GLuint m_handle;
	int m_width;
	int m_height;
	GLenum m_format;
};

}
}
