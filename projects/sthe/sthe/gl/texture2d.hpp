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

class Texture2D : public Texture
{
public:
	// Constructors
	Texture2D();
	Texture2D(const int t_width, const int t_height, const GLenum t_format, const bool t_hasMipmap);
	Texture2D(const int t_width, const int t_height, const GLenum t_format, const int t_mipCount);
	Texture2D(const std::string& t_file, const bool t_hasMipmap = true);
	Texture2D(const std::string& t_file, const int t_mipCount);
	Texture2D(const Texture2D& t_texture2D) noexcept;
	Texture2D(Texture2D&& t_texture2D) noexcept;

	// Destructor
	~Texture2D() = default;

	// Operators
	Texture2D& operator=(const Texture2D& t_texture2D) noexcept;
	Texture2D& operator=(Texture2D&& t_texture2D) noexcept;
	
	// Functionality
	void reinitialize(const int t_width, const int t_height, const GLenum t_format, const bool t_hasMipmap = true);
	void reinitialize(const int t_width, const int t_height, const GLenum t_format, const int t_mipCount);
	void release() override;
	
	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);

	template<typename T>
	void upload(const std::vector<T>& t_source, const int t_x, const int t_y, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);
	
	template<typename T>
	void upload(const T* const t_source, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);
	
	template<typename T>
	void upload(const T* const t_source, const int t_x, const int t_y, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0);

	template<typename T>
	void download(std::vector<T>& t_destination, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;
	
	template<typename T>
	void download(std::vector<T>& t_destination, const int t_x, const int t_y, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;

	template<typename T>
	void download(T* const t_destination, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;
	
	template<typename T>
	void download(T* const t_destination, const int t_x, const int t_y, const int t_width, const int t_height, const GLenum t_format, const GLenum t_type, const int t_mipLevel = 0) const;

	// Getters
	int getWidth() const override;
	int getHeight() const override;
	GLenum getFormat() const override;
	int getMipCount() const override;
private:
	// Attributes
	int m_width;
	int m_height;
	GLenum m_format;
	int m_mipCount;
};

}
}

#include "texture2d.inl"
