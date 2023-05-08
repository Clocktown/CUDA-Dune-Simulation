#include "texture2d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <glad/glad.h>
#include <utility>
#include <string>

namespace sthe
{
namespace gl
{

// Constructors
Texture2D::Texture2D() :
	Texture{ GL_TEXTURE_2D },
	m_width{ 0 },
	m_height{ 0 },
	m_format{ GL_NONE },
	m_mipCount{ 0 }
{

}

Texture2D::Texture2D(const int t_width, const int t_height, const GLenum t_format, const bool t_hasMipmap) :
	Texture2D{ t_width, t_height, t_format, t_hasMipmap ? getMipCapacity(t_width, t_height) : 1 }
{

}

Texture2D::Texture2D(const int t_width, const int t_height, const GLenum t_format, const int t_mipCount) :
	Texture{ GL_TEXTURE_2D },
	m_width{ t_width },
	m_height{ t_height },
	m_mipCount{ t_mipCount }
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_mipCount >= 0, "MipCount must be greater than or equal to 0");

	if (m_width > 0 || m_height > 0)
	{
		GL_CHECK_ERROR(glGetInternalformativ(GL_TEXTURE_2D, t_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glTextureStorage2D(getHandle(), m_mipCount, m_format, m_width, m_height));
	}
	else
	{
		m_format = t_format;
	}
}

Texture2D::Texture2D(const std::string& t_file, const bool t_hasMipmap) :
	Texture{ GL_TEXTURE_2D }
{
	GLenum pixelFormat;
	GLenum pixelType;
	const std::shared_ptr<unsigned char> source{ readImage2D(t_file, m_width, m_height, m_format, pixelFormat, pixelType) };

	GL_CHECK_ERROR(glGetInternalformativ(GL_TEXTURE_2D, m_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
	m_mipCount = t_hasMipmap ? getMipCapacity(m_width, m_height) : 1;

	GL_CHECK_ERROR(glTextureStorage2D(getHandle(), m_mipCount, m_format, m_width, m_height));
	GL_CHECK_ERROR(glTextureSubImage2D(getHandle(), 0, 0, 0, m_width, m_height, pixelFormat, pixelType, source.get()));
	GL_CHECK_ERROR(glGenerateTextureMipmap(getHandle()));
}

Texture2D::Texture2D(const std::string& t_file, const int t_mipCount) :
	Texture{ GL_TEXTURE_2D },
	m_mipCount{ t_mipCount }
{
	GLenum pixelFormat;
	GLenum pixelType;
	const std::shared_ptr<unsigned char> source{ readImage2D(t_file, m_width, m_height, m_format, pixelFormat, pixelType) };

	GL_CHECK_ERROR(glGetInternalformativ(GL_TEXTURE_2D, m_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
	GL_CHECK_ERROR(glTextureStorage2D(getHandle(), m_mipCount, m_format, m_width, m_height));
	GL_CHECK_ERROR(glTextureSubImage2D(getHandle(), 0, 0, 0, m_width, m_height, pixelFormat, pixelType, source.get()));
	GL_CHECK_ERROR(glGenerateTextureMipmap(getHandle()));
}

Texture2D::Texture2D(const Texture2D& t_texture2D) noexcept :
	Texture{ t_texture2D },
	m_width{ t_texture2D.m_width },
	m_height{ t_texture2D.m_height },
	m_format{ t_texture2D.m_format },
	m_mipCount{ t_texture2D.m_mipCount }
{
	if (t_texture2D.hasStorage())
	{
		GL_CHECK_ERROR(glTextureStorage2D(getHandle(), m_mipCount, m_format, m_width, m_height));
		
		for (int i{ 0 }, width{ m_width }, height{ m_height }; i < m_mipCount; ++i, width = std::max(width / 2, 1), height = std::max(height / 2, 1))
		{
			GL_CHECK_ERROR(glCopyImageSubData(t_texture2D.getHandle(), GL_TEXTURE_2D, i, 0, 0, 0, getHandle(), GL_TEXTURE_2D, i, 0, 0, 0, width, height, 1));
		}
	}
}

Texture2D::Texture2D(Texture2D&& t_texture2D) noexcept :
	Texture{ t_texture2D },
	m_width{ std::exchange(t_texture2D.m_width, 0) },
	m_height{ std::exchange(t_texture2D.m_height, 0) },
	m_format{ std::exchange(t_texture2D.m_format, GL_NONE) },
	m_mipCount{ std::exchange(t_texture2D.m_mipCount, 0) }
{

}

// Operators
Texture2D& Texture2D::operator=(const Texture2D& t_texture2D) noexcept
{
	if (this != &t_texture2D)
	{
		reinitialize(t_texture2D.m_width, t_texture2D.m_height, t_texture2D.m_format, t_texture2D.m_mipCount);
		Texture::operator=(t_texture2D);

		if (t_texture2D.hasStorage())
		{
			for (int i{ 0 }, width{ m_width }, height{ m_height }; i < m_mipCount; ++i, width = std::max(width / 2, 1), height = std::max(height / 2, 1))
			{
				GL_CHECK_ERROR(glCopyImageSubData(t_texture2D.getHandle(), GL_TEXTURE_2D, i, 0, 0, 0, getHandle(), GL_TEXTURE_2D, i, 0, 0, 0, width, height, 1));
			}
		}
	}

	return *this;
}

Texture2D& Texture2D::operator=(Texture2D&& t_texture2D) noexcept
{
	if (this != &t_texture2D)
	{
		Texture::operator=(t_texture2D);
		m_width = std::exchange(t_texture2D.m_width, 0);
		m_height = std::exchange(t_texture2D.m_height, 0);
		m_format = std::exchange(t_texture2D.m_format, GL_NONE);
		m_mipCount = std::exchange(t_texture2D.m_mipCount, 0);
	}

	return *this;
}

// Functionality
void Texture2D::reinitialize(const int t_width, const int t_height, const GLenum t_format, const bool t_hasMipmap)
{
	reinitialize(t_width, t_height, t_format, t_hasMipmap ? getMipCapacity(t_width, t_height) : 1);
}

void Texture2D::reinitialize(const int t_width, const int t_height, const GLenum t_format, const int t_mipCount)
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_mipCount >= 0, "MipCount must be greater than or equal to 0");

	if (m_width == t_width && m_height == t_height && m_format == t_format)
	{
		return;
	}

	if (hasStorage())
	{
		recreate();
	}
	
	m_width = t_width;
	m_height = t_height;
	m_mipCount = t_mipCount;

	if (m_width > 0 || m_height > 0)
	{
		GL_CHECK_ERROR(glGetInternalformativ(GL_TEXTURE_2D, t_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glTextureStorage2D(getHandle(), m_mipCount, m_format, m_width, m_height));
	}
	else
	{
		m_format = t_format;
	}
}

void Texture2D::release()
{
	if (hasStorage())
	{
		recreate();
		m_width = 0;
		m_height = 0;
	}
}

// Getters
int Texture2D::getWidth() const
{
	return m_width;
}

int Texture2D::getHeight() const
{
	return m_height;
}

GLenum Texture2D::getFormat() const
{
	return m_format;
}

int Texture2D::getMipCount() const
{
	return m_mipCount;
}

}
}
