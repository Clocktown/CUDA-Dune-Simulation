#include "handle.hpp"
#include "framebuffer.hpp"
#include "texture.hpp"
#include "texture2d.hpp"
#include "renderbuffer.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <utility>
#include <memory>
#include <vector>
#include <map>

namespace sthe
{
namespace gl
{

void DefaultFramebuffer::bind(const GLenum t_target)
{
	GL_CHECK_ERROR(glBindFramebuffer(t_target, 0));
}

constexpr GLuint DefaultFramebuffer::getHandle()
{
	return 0;
}

// Constructors
Framebuffer::Framebuffer() :
	m_width{ 0 },
	m_height{ 0 }
{
	GL_CHECK_ERROR(glCreateFramebuffers(1, &m_handle));
}

Framebuffer::Framebuffer(const int t_width, const int t_height) :
	m_width{ t_width },
	m_height{ t_height }
{
	STHE_ASSERT((t_width > 0 && t_height > 0) || (t_width == 0 && t_height == 0), "Width and height must be simultaneously greater than or equal to 0");

	GL_CHECK_ERROR(glCreateFramebuffers(1, &m_handle));
}

Framebuffer::Framebuffer(Framebuffer&& t_framebuffer) noexcept :
	m_handle{ std::exchange(t_framebuffer.m_handle, GL_NONE) },
	m_width{ std::exchange(t_framebuffer.m_width, 0) },
	m_height{ std::exchange(t_framebuffer.m_height, 0) }
{

}

// Destructor
Framebuffer::~Framebuffer()
{
	GL_CHECK_ERROR(glDeleteFramebuffers(1, &m_handle));
}

// Operators
Framebuffer& Framebuffer::operator=(Framebuffer&& t_framebuffer) noexcept
{
	if (this != &t_framebuffer)
	{
		GL_CHECK_ERROR(glDeleteFramebuffers(1, &m_handle));

		m_handle = std::exchange(t_framebuffer.m_handle, GL_NONE);
		m_width = std::exchange(t_framebuffer.m_width, 0);
		m_height = std::exchange(t_framebuffer.m_height, 0);
	}

	return *this;
}

// Functionality
void Framebuffer::bind(const GLenum t_target) const
{
	GL_CHECK_FRAMEBUFFER(m_handle, t_target);
	GL_CHECK_ERROR(glBindFramebuffer(t_target, m_handle));
}

void Framebuffer::blit(Framebuffer& t_framebuffer, const GLbitfield t_mask, const GLenum t_filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, t_framebuffer.getHandle(), 0, 0, m_width, m_height, 0, 0, t_framebuffer.getWidth(), t_framebuffer.getHeight(), t_mask, t_filter));
}

void Framebuffer::blit(Framebuffer& t_framebuffer, const int t_width, const int t_height, const GLbitfield t_mask, const GLenum t_filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, t_framebuffer.getHandle(), 0, 0, m_width, m_height, 0, 0, t_width, t_height, t_mask, t_filter));
}

void Framebuffer::blit(Framebuffer& t_framebuffer, const int t_x, const int t_heightOffset, const int t_width, const int t_height, const GLbitfield t_mask, const GLenum t_filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, t_framebuffer.getHandle(), 0, 0, m_width, m_height, t_x, t_heightOffset, t_width, t_height, t_mask, t_filter));
}

void Framebuffer::blit(const GLuint t_framebuffer, const GLbitfield t_mask) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, t_framebuffer, 0, 0, m_width, m_height, 0, 0, m_width, m_height, t_mask, GL_NEAREST));
}

void Framebuffer::blit(const GLuint t_framebuffer, const int t_width, const int t_height, const GLbitfield t_mask, const GLenum t_filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, t_framebuffer, 0, 0, m_width, m_height, 0, 0, t_width, t_height, t_mask, t_filter));
}

void Framebuffer::blit(const GLuint t_framebuffer, const int t_x, const int t_heightOffset, const int t_width, const int t_height, const GLbitfield t_mask, const GLenum t_filter) const
{
	GL_CHECK_ERROR(glBlitNamedFramebuffer(m_handle, t_framebuffer, 0, 0, m_width, m_height, t_x, t_heightOffset, t_width, t_height, t_mask, t_filter));
}

void Framebuffer::enableReadBuffer(const GLenum t_attachment)
{
	GL_CHECK_ERROR(glNamedFramebufferReadBuffer(m_handle, t_attachment));
}

void Framebuffer::enableDrawBuffer(const GLenum t_attachment)
{
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffer(m_handle, t_attachment));
}

void Framebuffer::enableDrawBuffers(const std::vector<GLenum>& t_attachments)
{
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffers(m_handle, static_cast<int>(t_attachments.size()), t_attachments.data()));
}

void Framebuffer::enableDrawBuffers(const GLenum* const t_attachments, const int t_count)
{
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffers(m_handle, t_count, t_attachments));
}

void Framebuffer::disableReadBuffer()
{
	GL_CHECK_ERROR(glNamedFramebufferReadBuffer(m_handle, GL_NONE));
}

void Framebuffer::disableDrawBuffers()
{
	GL_CHECK_ERROR(glNamedFramebufferDrawBuffer(m_handle, GL_NONE));
}

void Framebuffer::resize(const int t_width, const int t_height)
{
	STHE_ASSERT((t_width > 0 && t_height > 0) || (t_width == 0 && t_height == 0), "Width and height must be simultaneously greater than or equal to 0");

	m_width = t_width;
	m_height = t_height;
}

void Framebuffer::attachTexture(const GLenum t_attachment, Texture& t_texture, const int t_mipLevel)
{
	GL_CHECK_ERROR(glNamedFramebufferTexture(m_handle, t_attachment, t_texture.getHandle(), t_mipLevel));
}

void Framebuffer::attachTexture(const GLenum t_attachment, const GLuint t_texture, const int t_mipLevel)
{
	GL_CHECK_ERROR(glNamedFramebufferTexture(m_handle, t_attachment, t_texture, t_mipLevel));
}

void Framebuffer::attachRenderbuffer(const GLenum t_attachment, Renderbuffer& t_renderbuffer)
{
	GL_CHECK_ERROR(glNamedFramebufferRenderbuffer(m_handle, t_attachment, GL_RENDERBUFFER, t_renderbuffer.getHandle()));
}

void Framebuffer::attachRenderbuffer(const GLenum t_attachment, const GLuint t_renderbuffer)
{
	GL_CHECK_ERROR(glNamedFramebufferRenderbuffer(m_handle, t_attachment, GL_RENDERBUFFER, t_renderbuffer));
}

void Framebuffer::detach(const GLenum t_attachment)
{
	GL_CHECK_ERROR(glNamedFramebufferRenderbuffer(m_handle, t_attachment, GL_RENDERBUFFER, GL_NONE));
}

// Getters
GLuint Framebuffer::getHandle() const
{
	return m_handle;
}

int Framebuffer::getWidth() const
{
	return m_width;
}

int Framebuffer::getHeight() const
{
	return m_height;
}

}
}
