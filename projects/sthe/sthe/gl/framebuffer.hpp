#pragma once

#include "handle.hpp"
#include "texture.hpp"
#include "texture2d.hpp"
#include "renderbuffer.hpp"
#include <glad/glad.h>
#include <memory>
#include <vector>
#include <map>

namespace sthe
{
namespace gl
{

namespace DefaultFramebuffer
{
    void bind(const GLenum t_target = GL_DRAW_FRAMEBUFFER);
    constexpr GLuint getHandle();
}

class Framebuffer : public Handle
{
public:
	// Constructors
	Framebuffer();
	Framebuffer(const int t_width, const int t_height);
	Framebuffer(const Framebuffer& t_framebuffer) = delete;
	Framebuffer(Framebuffer&& t_framebuffer) noexcept;
	
	// Destructor
	~Framebuffer();

	// Operators
	Framebuffer& operator=(const Framebuffer& t_framebuffer) = delete;
	Framebuffer& operator=(Framebuffer&& t_framebuffer) noexcept;
	
	// Functionality
	void bind(const GLenum t_target = GL_DRAW_FRAMEBUFFER) const;
	void blit(Framebuffer& t_framebuffer, const GLbitfield t_mask = GL_COLOR_BUFFER_BIT, const GLenum t_filter = GL_NEAREST) const;
	void blit(Framebuffer& t_framebuffer, const int t_width, const int t_height, const GLbitfield t_mask = GL_COLOR_BUFFER_BIT, const GLenum t_filter = GL_NEAREST) const;
	void blit(Framebuffer& t_framebuffer, const int t_x, const int t_heightOffset, const int t_width, const int t_height, const GLbitfield t_mask = GL_COLOR_BUFFER_BIT, const GLenum t_filter = GL_NEAREST) const;
	void blit(const GLuint t_framebuffer = DefaultFramebuffer::getHandle(), const GLbitfield t_mask = GL_COLOR_BUFFER_BIT) const;
	void blit(const GLuint t_framebuffer, const int t_width, const int t_height, const GLbitfield t_mask = GL_COLOR_BUFFER_BIT, const GLenum t_filter = GL_NEAREST) const;
	void blit(const GLuint t_framebuffer, const int t_x, const int t_heightOffset, const int t_width, const int t_height, const GLbitfield t_mask = GL_COLOR_BUFFER_BIT, const GLenum t_filter = GL_NEAREST) const;
	void resize(const int t_width, const int t_height);
	void enableReadBuffer(const GLenum t_attachment);
	void enableDrawBuffer(const GLenum t_attachment);
	void enableDrawBuffers(const std::vector<GLenum>& t_attachments);
	void enableDrawBuffers(const GLenum* const t_attachments, const int t_count);
	void disableReadBuffer();
	void disableDrawBuffers();
	void attachTexture(const GLenum t_attachment, Texture& t_texture, const int t_mipLevel = 0);
	void attachTexture(const GLenum t_attachment, const GLuint t_texture, const int t_mipLevel = 0);
	void attachRenderbuffer(const GLenum t_attachment, Renderbuffer& t_renderbuffer);
	void attachRenderbuffer(const GLenum t_attachment, const GLuint t_renderbuffer);
	void detach(const GLenum t_attachment);

	// Getters
	GLuint getHandle() const override;
	int getWidth() const;
	int getHeight() const;
private:
	// Attributes
	GLuint m_handle;
	int m_width;
	int m_height;
};

}
}
