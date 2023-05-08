#pragma once

#include "image.hpp"
#include <glad/glad.h>

namespace sthe
{
namespace gl
{

class Texture : public Image
{
public:
	// Constructors
	explicit Texture(const GLenum t_target);
	Texture(const Texture& t_texture) noexcept;
	Texture(Texture&& t_texture) noexcept;

	// Destructor
	virtual ~Texture() = default;

	// Operators
	Texture& operator=(const Texture& t_texture) noexcept;
	Texture& operator=(Texture&& t_texture) noexcept;

	// Functionality
	void bind() const override;
	void bind(const int t_unit) const;
	void recalculateMipmap();
	
	// Setters
	void setMinFilter(const GLenum t_minFilter);
	void setMagFilter(const GLenum t_magFilter);
	void setWrapModeU(const GLenum t_wrapModeU);
	void setWrapModeV(const GLenum t_wrapModeV);
	void setWrapModeW(const GLenum t_wrapModeW);

	// Getters
	GLuint getHandle() const override;
	GLenum getTarget() const override;
	GLenum getMinFilter() const;
	GLenum getMagFilter() const;
	GLenum getWrapModeU() const;
	GLenum getWrapModeV() const;
	GLenum getWrapModeW() const;
protected:
	// Functionality
	void recreate();
private:
	// Functionality
	void update();

	// Attributes
	GLuint m_handle;
	GLenum m_target;
	GLenum m_minFilter;
	GLenum m_magFilter;
	GLenum m_wrapModeU;
	GLenum m_wrapModeV;
	GLenum m_wrapModeW;
};

}
}
