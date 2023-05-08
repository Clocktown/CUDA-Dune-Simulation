#include "handle.hpp"
#include "vertex_array.hpp"
#include "buffer.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <utility>

namespace sthe
{
namespace gl
{

// Constructors
VertexArray::VertexArray()
{
	GL_CHECK_ERROR(glCreateVertexArrays(1, &m_handle));
}

VertexArray::VertexArray(VertexArray&& t_vertexArray) noexcept :
	m_handle{ std::exchange(t_vertexArray.m_handle, GL_NONE) }
{

}

// Destructor
VertexArray::~VertexArray()
{
	GL_CHECK_ERROR(glDeleteVertexArrays(1, &m_handle));
}

// Operator
VertexArray& VertexArray::operator=(VertexArray&& t_vertexArray) noexcept
{
	if (this != &t_vertexArray)
	{
		GL_CHECK_ERROR(glDeleteVertexArrays(1, &m_handle));
		m_handle = std::exchange(t_vertexArray.m_handle, GL_NONE);
	}

	return *this;
}

// Functionality
void VertexArray::bind() const
{
	GL_CHECK_ERROR(glBindVertexArray(m_handle));
}

void VertexArray::attachIndexBuffer(const Buffer& t_indexBuffer)
{
	GL_CHECK_ERROR(glVertexArrayElementBuffer(m_handle, t_indexBuffer.getHandle()));
}

void VertexArray::attachIndexBuffer(const GLuint t_indexBuffer) 
{
	GL_CHECK_ERROR(glVertexArrayElementBuffer(m_handle, t_indexBuffer));
}

void VertexArray::attachVertexBuffer(const int t_location, const Buffer& t_vertexBuffer, const int t_offset)
{
	attachVertexBuffer(t_location, t_vertexBuffer.getHandle(), t_offset * static_cast<long long int>(t_vertexBuffer.getStride()), t_vertexBuffer.getStride());
}

void VertexArray::attachVertexBuffer(const int t_location, const GLuint t_vertexBuffer, const int t_stride)
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");

	GL_CHECK_ERROR(glVertexArrayVertexBuffer(m_handle, static_cast<GLuint>(t_location), t_vertexBuffer, 0, t_stride));
}

void VertexArray::attachVertexBuffer(const int t_location, const GLuint t_vertexBuffer, const long long int t_offset, const int t_stride)
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");

	GL_CHECK_ERROR(glVertexArrayVertexBuffer(m_handle, static_cast<GLuint>(t_location), t_vertexBuffer, t_offset, t_stride));
}

void VertexArray::detachIndexBuffer()
{
	GL_CHECK_ERROR(glVertexArrayElementBuffer(m_handle, GL_NONE));
}

void VertexArray::detachVertexBuffer(const int t_location)
{
	GL_CHECK_ERROR(glVertexArrayVertexBuffer(m_handle, static_cast<GLuint>(t_location), GL_NONE, 0, 0));
}

// Setters
void VertexArray::setVertexAttributeFormat(const int t_location, const int t_count, const GLenum t_type, const bool t_isNormalized, const int t_relativeStride)
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");
	STHE_ASSERT(t_relativeStride >= 0, "Relative stride must be greater than or equal to 0");

	GL_CHECK_ERROR(glVertexArrayAttribFormat(m_handle, static_cast<GLuint>(t_location), t_count, t_type, t_isNormalized, static_cast<GLuint>(t_relativeStride)));
}

void VertexArray::setVertexAttributeDivisor(const int t_location, const int t_divisor)
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");
	STHE_ASSERT(t_divisor >= 0, "Divisor must be greater than or equal to 0");

	GL_CHECK_ERROR(glVertexArrayBindingDivisor(m_handle, static_cast<GLuint>(t_location), static_cast<GLuint>(t_divisor)));
}

void VertexArray::enableVertexAttribute(const int t_location)
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");

	GL_CHECK_ERROR(glEnableVertexArrayAttrib(m_handle, static_cast<GLuint>(t_location)));
}

void VertexArray::disableVertexAttribute(const int t_location)
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");

	GL_CHECK_ERROR(glDisableVertexArrayAttrib(m_handle, static_cast<GLuint>(t_location)));
}

// Getter
GLuint VertexArray::getHandle() const
{
	return m_handle;
}

}
}
