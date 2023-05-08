#include "texture3d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <glad/glad.h>
#include <utility>

namespace sthe
{
namespace gl
{

// Constructors
Texture3D::Texture3D() :
	Texture{ GL_TEXTURE_3D },
	m_width{ 0 },
	m_height{ 0 },
	m_depth{ 0 },
	m_format{ GL_NONE },
	m_mipCount{ 0 }
{
	
}

Texture3D::Texture3D(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const bool t_hasMipmap) :
	Texture3D{ t_width, t_height, t_depth, t_format, t_hasMipmap ? getMipCapacity(t_width, t_height, t_depth) : 1 }
{

}

Texture3D::Texture3D(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const int t_mipCount) :
	Texture{ GL_TEXTURE_3D },
	m_width{ t_width },
	m_height{ t_height },
	m_depth{ t_depth },
	m_mipCount{ t_mipCount }
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_depth >= 0, "Depth must be greater than or equal to 0");
	STHE_ASSERT(t_mipCount >= 0, "MipCount must be greater than or equal to 0");

	if (m_width > 0 || m_height > 0 || m_depth > 0)
	{
		GL_CHECK_ERROR(glGetInternalformativ(GL_TEXTURE_3D, t_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glTextureStorage3D(getHandle(), m_mipCount, m_format, m_width, m_height, m_depth));
	}
	else
	{
		m_format = t_format;
	}
}

Texture3D::Texture3D(const Texture3D& t_texture3D) noexcept :
	Texture{ GL_TEXTURE_3D },
	m_width{ t_texture3D.m_width },
	m_height{ t_texture3D.m_height },
	m_depth{ t_texture3D.m_depth },
	m_format{ t_texture3D.m_format },
	m_mipCount{ t_texture3D.m_mipCount }
{
	if (t_texture3D.hasStorage())
	{
		GL_CHECK_ERROR(glTextureStorage3D(getHandle(), m_mipCount, m_format, m_width, m_height, m_depth));
	
		for (int i{ 0 }, width{ m_width }, height{ m_height }, depth{ m_depth }; i < m_mipCount; ++i, width = std::max(width / 2, 1), height = std::max(height / 2, 1), depth = std::max(depth / 2, 1))
		{
			GL_CHECK_ERROR(glCopyImageSubData(t_texture3D.getHandle(), GL_TEXTURE_3D, i, 0, 0, 0, getHandle(), GL_TEXTURE_3D, i, 0, 0, 0, width, height, depth));
		}
	}
}

Texture3D::Texture3D(Texture3D&& t_texture3D) noexcept :
	Texture{ t_texture3D },
	m_width{ std::exchange(t_texture3D.m_width, 0) },
	m_height{ std::exchange(t_texture3D.m_height, 0) },
	m_depth{ std::exchange(t_texture3D.m_depth, 0) },
	m_format{ std::exchange(t_texture3D.m_format, GL_NONE) },
	m_mipCount{ std::exchange(t_texture3D.m_mipCount, 0) }
{

}

// Operators
Texture3D& Texture3D::operator=(const Texture3D& t_texture3D) noexcept
{
	if (this != &t_texture3D)
	{
		reinitialize(t_texture3D.m_width, t_texture3D.m_height, t_texture3D.m_depth, t_texture3D.m_format, t_texture3D.m_mipCount);
		Texture::operator=(t_texture3D);

		if (t_texture3D.hasStorage())
		{
			for (int i{ 0 }, width{ m_width }, height{ m_height }, depth{ m_depth }; i < m_mipCount; ++i, width = std::max(width / 2, 1), height = std::max(height / 2, 1), depth = std::max(depth / 2, 1))
			{
				GL_CHECK_ERROR(glCopyImageSubData(t_texture3D.getHandle(), GL_TEXTURE_3D, i, 0, 0, 0, getHandle(), GL_TEXTURE_3D, i, 0, 0, 0, width, height, depth));
			}
		}
	}

	return *this;
}

Texture3D& Texture3D::operator=(Texture3D&& t_texture3D) noexcept
{
	if (this != &t_texture3D)
	{
		Texture::operator=(t_texture3D);
		m_width = std::exchange(t_texture3D.m_width, 0);
		m_height = std::exchange(t_texture3D.m_height, 0);
		m_depth = std::exchange(t_texture3D.m_depth, 0);
		m_format = std::exchange(t_texture3D.m_format, GL_NONE);
		m_mipCount = std::exchange(t_texture3D.m_mipCount, 0);
	}

	return *this;
}

// Functionality
void Texture3D::reinitialize(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const bool t_hasMipmap)
{
	reinitialize(t_width, t_height, t_depth, t_format, t_hasMipmap ? getMipCapacity(t_width, t_height, t_depth) : 1);
}

void Texture3D::reinitialize(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const int t_mipCount)
{
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_depth >= 0, "Depth must be greater than or equal to 0");
	STHE_ASSERT(t_mipCount >= 0, "MipCount must be greater than or equal to 0");

	if (m_width == t_width && m_height == t_height && m_depth == t_depth && m_format == t_format)
	{
		return;
	}

	if (hasStorage())
	{
		recreate();
	}

	m_width = t_width;
	m_height = t_height;
	m_depth = t_depth;
	m_mipCount = t_mipCount;

	if (m_width > 0 || m_height > 0 || m_depth > 0)
	{
		GL_CHECK_ERROR(glGetInternalformativ(GL_TEXTURE_3D, t_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
		GL_CHECK_ERROR(glTextureStorage3D(getHandle(), m_mipCount, m_format, m_width, m_height, m_depth));
	}
	else
	{
		m_format = t_format;
	}
}

void Texture3D::release()
{
	if (hasStorage())
	{
		recreate();
		m_width = 0;
		m_height = 0;
		m_depth = 0;
	}
}

// Getters
int Texture3D::getWidth() const
{
	return m_width;
}

int Texture3D::getHeight() const
{
	return m_height;
}

int Texture3D::getDepth() const
{
	return m_depth;
}

GLenum Texture3D::getFormat() const
{
	return m_format;
}

int Texture3D::getMipCount() const
{
	return m_mipCount;
}

}
}
