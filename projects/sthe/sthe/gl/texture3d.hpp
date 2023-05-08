#pragma once

#include "texture.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace sthe
{
namespace gl
{

class Texture3D : public Texture
{
public:
	// Constructors
	Texture3D();
	Texture3D(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const bool t_hasMipmap = true);
	Texture3D(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const int t_mipCount);
	Texture3D(const Texture3D& t_texture3D) noexcept;
	Texture3D(Texture3D&& t_texture3D) noexcept;

	// Destructor
	~Texture3D() = default;

	// Operators
	Texture3D& operator=(const Texture3D& t_texture3D) noexcept;
	Texture3D& operator=(Texture3D&& t_texture3D) noexcept;
	
	// Functionality
	void reinitialize(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const bool t_hasMipmap = true);
	void reinitialize(const int t_width, const int t_height, const int t_depth, const GLenum t_format, const int t_mipCount);
	void release() override;

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);
	
	template<typename T>
	void upload(const T* const t_source, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);
	
	template<typename T>
	void upload(const T* const t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);

	template<typename T>
	void download(std::vector<T>& t_destination, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;
	
	template<typename T>
	void download(std::vector<T>& t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;
	
	template<typename T>
	void download(T* const t_destination, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;
	
	template<typename T>
	void download(T* const t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;

	// Getters
	int getWidth() const override;
	int getHeight() const override;
	int getDepth() const override;
	GLenum getFormat() const override;
	int getMipCount() const override;
private:
	// Attributes
	int m_width;
	int m_height;
	int m_depth;
	GLenum m_format;
	int m_mipCount;
};

}
}

#include "texture3d.inl"
