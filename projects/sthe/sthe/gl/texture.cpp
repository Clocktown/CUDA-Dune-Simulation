#include "texture.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <utility>

namespace sthe
{
namespace gl
{

// Constructors
Texture::Texture(const GLenum t_target) :
	m_target{ t_target },
	m_minFilter{ GL_NEAREST_MIPMAP_LINEAR },
	m_magFilter{ GL_LINEAR },
	m_wrapModeU{ GL_REPEAT },
	m_wrapModeV{ GL_REPEAT },
	m_wrapModeW{ GL_REPEAT }
{
	GL_CHECK_ERROR(glCreateTextures(m_target, 1, &m_handle));
}

Texture::Texture(const Texture& t_texture) noexcept :
	m_target{ t_texture.m_target },
	m_minFilter{ t_texture.m_minFilter },
	m_magFilter{ t_texture.m_magFilter },
	m_wrapModeU{ t_texture.m_wrapModeU },
	m_wrapModeV{ t_texture.m_wrapModeV },
	m_wrapModeW{ t_texture.m_wrapModeW }
{
	GL_CHECK_ERROR(glCreateTextures(m_target, 1, &m_handle));
	update();
}

Texture::Texture(Texture&& t_texture3D) noexcept :
	m_handle{ std::exchange(t_texture3D.m_handle, GL_NONE) },
	m_target{ std::exchange(t_texture3D.m_target, GL_NONE) },
	m_minFilter{ std::exchange(t_texture3D.m_minFilter, GL_NONE) },
	m_magFilter{ std::exchange(t_texture3D.m_magFilter, GL_NONE) },
	m_wrapModeU{ std::exchange(t_texture3D.m_wrapModeU, GL_NONE) },
	m_wrapModeV{ std::exchange(t_texture3D.m_wrapModeV, GL_NONE) },
	m_wrapModeW{ std::exchange(t_texture3D.m_wrapModeW, GL_NONE) }
{
	
}

// Operators
Texture& Texture::operator=(const Texture& t_texture) noexcept
{
	if (this != &t_texture)
	{
		setMinFilter(t_texture.m_minFilter);
		setMagFilter(t_texture.m_magFilter);
		setWrapModeU(t_texture.m_wrapModeU);
		setWrapModeV(t_texture.m_wrapModeV);
		setWrapModeW(t_texture.m_wrapModeW);
	}

	return *this;
}

Texture& Texture::operator=(Texture&& t_texture) noexcept
{
	if (this != &t_texture)
	{
		GL_CHECK_ERROR(glDeleteTextures(1, &m_handle));

		m_handle = std::exchange(t_texture.m_handle, GL_NONE);
		m_target = std::exchange(t_texture.m_target, GL_NONE);
		m_minFilter = std::exchange(t_texture.m_minFilter, GL_NONE);
		m_magFilter = std::exchange(t_texture.m_magFilter, GL_NONE);
		m_wrapModeU = std::exchange(t_texture.m_wrapModeU, GL_NONE);
		m_wrapModeV = std::exchange(t_texture.m_wrapModeV, GL_NONE);
		m_wrapModeW = std::exchange(t_texture.m_wrapModeW, GL_NONE);
	}

	return *this;
}

// Functionality
void Texture::bind() const
{
	GL_CHECK_ERROR(glBindTexture(m_target, m_handle));
}

void Texture::bind(const int t_unit) const
{
	STHE_ASSERT(t_unit >= 0, "Unit must be greater than or equal to 0");

	GL_CHECK_ERROR(glBindTextureUnit(t_unit, m_handle));
}

void Texture::recalculateMipmap()
{
	GL_CHECK_ERROR(glGenerateTextureMipmap(m_handle));
}

void Texture::recreate()
{
	GL_CHECK_ERROR(glDeleteTextures(1, &m_handle));
	GL_CHECK_ERROR(glCreateTextures(m_target, 1, &m_handle));
	update();
}

void Texture::update()
{
	if (m_minFilter != GL_NEAREST_MIPMAP_LINEAR)
	{
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_MIN_FILTER, &m_minFilter));
	}

	if (m_magFilter != GL_LINEAR)
	{
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_MAG_FILTER, &m_magFilter));
	}

	if (m_wrapModeU != GL_REPEAT)
	{
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_S, &m_wrapModeU));
	}

	if (m_wrapModeV != GL_REPEAT)
	{
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_T, &m_wrapModeV));
	}

	if (m_wrapModeW != GL_REPEAT)
	{
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_R, &m_wrapModeW));
	}
}

// Setters
void Texture::setMinFilter(const GLenum t_minFilter)
{
	if (m_minFilter != t_minFilter)
	{
		m_minFilter = t_minFilter;
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_MIN_FILTER, &m_minFilter));
	}
}

void Texture::setMagFilter(const GLenum t_magFilter)
{
	if (m_magFilter != t_magFilter)
	{
		m_magFilter = t_magFilter;
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_MIN_FILTER, &m_magFilter));
	}
}

void Texture::setWrapModeU(const GLenum t_wrapModeU)
{
	if (m_wrapModeU != t_wrapModeU)
	{
		m_wrapModeU = t_wrapModeU;
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_S, &m_wrapModeU));
	}
}

void Texture::setWrapModeV(const GLenum t_wrapModeV)
{
	if (m_wrapModeV != t_wrapModeV)
	{
		m_wrapModeV = t_wrapModeV;
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_T, &m_wrapModeV));
	}
}

void Texture::setWrapModeW(const GLenum t_wrapModeW)
{
	if (m_wrapModeW != t_wrapModeW)
	{
		m_wrapModeW = t_wrapModeW;
		GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_R, &m_wrapModeW));
	}
}

// Getters
GLuint Texture::getHandle() const
{
	return m_handle;
}

GLenum Texture::getTarget() const
{
	return m_target;
}

GLenum Texture::getMinFilter() const
{
	return m_minFilter;
}

GLenum Texture::getMagFilter() const
{
	return m_magFilter;
}

GLenum Texture::getWrapModeU() const
{
	return m_wrapModeU;
}

GLenum Texture::getWrapModeV() const
{
	return m_wrapModeV;
}

GLenum Texture::getWrapModeW() const
{
	return m_wrapModeW;
}

}
}
