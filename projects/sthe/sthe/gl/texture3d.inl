#include "texture3d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <glad/glad.h>
#include <vector>

namespace sthe
{
namespace gl
{

// Functionality
template<typename T>
inline void Texture3D::upload(const std::vector<T>& t_source, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel)
{
	GL_CHECK_ERROR(glTextureSubImage3D(getHandle(), t_mipLevel, 0, 0, 0, t_width, t_height, t_depth, t_format, t_type, t_source.data()));
}

template<typename T>
inline void Texture3D::upload(const std::vector<T>& t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel)
{
	GL_CHECK_ERROR(glTextureSubImage3D(getHandle(), t_mipLevel, t_x, t_y, t_z, t_width, t_height, t_depth, t_format, t_type, t_source.data()));
}

template<typename T>
inline void Texture3D::upload(const T* const t_source, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel)
{
	GL_CHECK_ERROR(glTextureSubImage3D(getHandle(), t_mipLevel, 0, 0, 0, t_width, t_height, t_depth, t_format, t_type, t_source));
}

template<typename T>
inline void Texture3D::upload(const T* const t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel)
{
	GL_CHECK_ERROR(glTextureSubImage3D(getHandle(), t_mipLevel, t_x, t_y, t_z, t_width, t_height, t_depth, t_format, t_type, t_source));
}

template<typename T>
inline void Texture3D::download(std::vector<T>& t_destination, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel) const
{
	GL_CHECK_ERROR(glGetTextureSubImage(getHandle(), t_mipLevel, 0, 0, 0, t_width, t_height, t_depth, t_format, t_type, static_cast<GLsizei>(t_destination.size() * sizeof(T)), t_destination.data()));
}

template<typename T>
inline void Texture3D::download(std::vector<T>& t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel) const
{
	GL_CHECK_ERROR(glGetTextureSubImage(getHandle(), t_mipLevel, t_x, t_y, t_z, t_width, t_height, t_depth, t_format, t_type, static_cast<GLsizei>(t_destination.size() * sizeof(T)), t_destination.data()));
}

template<typename T>
inline void Texture3D::download(T* const t_destination, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel) const
{
	const GLsizei size{ t_width * t_height * t_depth * static_cast<GLsizei>(sizeof(T)) };
	GL_CHECK_ERROR(glGetTextureSubImage(getHandle(), t_mipLevel, 0, 0, 0, t_width, t_height, t_depth, t_format, t_type, size, t_destination));
}

template<typename T>
inline void Texture3D::download(T* const t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel) const
{
	const GLsizei size{ t_width * t_height * t_depth * static_cast<GLsizei>(sizeof(T)) };
	GL_CHECK_ERROR(glGetTextureSubImage(getHandle(), t_mipLevel, t_x, t_y, t_z, t_width, t_height, t_depth, t_format, t_type, size, t_destination));
}

}
}
