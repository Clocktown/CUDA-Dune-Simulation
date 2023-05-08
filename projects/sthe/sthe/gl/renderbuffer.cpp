#include "renderbuffer.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <utility>

namespace sthe
{
namespace gl
{

// Constructors
Renderbuffer::Renderbuffer() :
	m_width{ 0 },
	m_height{ 0 },
	m_format{ GL_NONE }
{
	GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));
}

Renderbuffer::Renderbuffer(const int t_width, const int t_height, const GLenum t_format) :
	m_width{ t_width },
	m_height{ t_height }
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");

	GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));
	
	if (m_width > 0 || m_height > 0)
	{
		GL_CHECK_ERROR(glGetInternalformativ(GL_RENDERBUFFER, t_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glNamedRenderbufferStorage(m_handle, m_format, m_width, m_height));
	}
	else
	{
		m_format = t_format;
	}
}

Renderbuffer::Renderbuffer(const Renderbuffer& t_renderbuffer) noexcept :
	m_width{ t_renderbuffer.m_width },
	m_height{ t_renderbuffer.m_height },
	m_format{ t_renderbuffer.m_format }
{
	GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));

	if (t_renderbuffer.hasStorage())
	{
		GL_CHECK_ERROR(glNamedRenderbufferStorage(m_handle, m_format, m_width, m_height));
		GL_CHECK_ERROR(glCopyImageSubData(t_renderbuffer.m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_width, m_height, 1));
	}
}

Renderbuffer::Renderbuffer(Renderbuffer&& t_renderbuffer) noexcept :
	m_handle{ std::exchange(t_renderbuffer.m_handle, GL_NONE) },
	m_width{ std::exchange(t_renderbuffer.m_width, 0) },
	m_height{ std::exchange(t_renderbuffer.m_height, 0) },
	m_format{ std::exchange(t_renderbuffer.m_format, GL_NONE) }
{

}

// Destructor
Renderbuffer::~Renderbuffer()
{
	GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));
}

// Operators
Renderbuffer& Renderbuffer::operator=(const Renderbuffer& t_renderbuffer) noexcept
{
	if (this != &t_renderbuffer)
	{
		reinitialize(t_renderbuffer.m_width, t_renderbuffer.m_height, t_renderbuffer.m_format);

		if (t_renderbuffer.hasStorage())
		{
			GL_CHECK_ERROR(glCopyImageSubData(t_renderbuffer.m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_handle, GL_RENDERBUFFER, 0, 0, 0, 0, m_width, m_height, 1));
		}
	}

	return *this;
}

Renderbuffer& Renderbuffer::operator=(Renderbuffer&& t_renderbuffer) noexcept
{
	if (this != &t_renderbuffer)
	{
		GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));

		m_handle = std::exchange(t_renderbuffer.m_handle, GL_NONE);
		m_width = std::exchange(t_renderbuffer.m_width, 0);
		m_height = std::exchange(t_renderbuffer.m_height, 0);
		m_format = std::exchange(t_renderbuffer.m_format, GL_NONE);
	}

	return *this;
}

// Functionality
void Renderbuffer::bind() const
{
	GL_CHECK_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, m_handle));
}

void Renderbuffer::reinitialize(const int t_width, const int t_height, const GLenum t_format)
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");

	if (m_width == t_width && m_height == t_height && m_format == t_format)
	{
		return;
	}

	if (hasStorage())
	{
		GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));
		GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));
	}
	
	m_width = t_width;
	m_height = t_height;

	if (m_width > 0 || m_height > 0)
	{
		GL_CHECK_ERROR(glGetInternalformativ(GL_RENDERBUFFER, t_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glNamedRenderbufferStorage(m_handle, m_format, m_width, m_height));
	}
	else
	{
		m_format = t_format;
	}
}

void Renderbuffer::release()
{
	if (hasStorage())
	{
		GL_CHECK_ERROR(glDeleteRenderbuffers(1, &m_handle));
		GL_CHECK_ERROR(glCreateRenderbuffers(1, &m_handle));
		m_width = 0;
		m_height = 0;
	}
}

// Getters
GLuint Renderbuffer::getHandle() const
{
	return m_handle;
}

GLenum Renderbuffer::getTarget() const
{
	return GL_RENDERBUFFER;
}

int Renderbuffer::getWidth() const
{
	return m_width;
}

int Renderbuffer::getHeight() const
{
	return m_height;
}

GLenum Renderbuffer::getFormat() const
{
	return m_format;
}

}
}
