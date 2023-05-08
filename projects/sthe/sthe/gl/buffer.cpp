#include "buffer.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <utility>

namespace sthe
{
namespace gl
{

// Constructors
Buffer::Buffer() :
	m_count{ 0 },
	m_stride{ 0 }
{
	GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
}

Buffer::Buffer(const int t_count, const int t_stride) :
	m_count{ t_count },
	m_stride{ t_stride }
{
	STHE_ASSERT(t_count >= 0, "Count must be greater than or equal to 0");
	STHE_ASSERT(t_stride >= 0, "Stride must be greater than or equal to 0");

	GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));

	if (m_count > 0)
	{
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, m_count * static_cast<GLsizeiptr>(m_stride), nullptr, GL_DYNAMIC_STORAGE_BIT));
	}
}

Buffer::Buffer(const Buffer& t_buffer) noexcept :
	m_count{ t_buffer.m_count },
	m_stride{ t_buffer.m_stride }
{
	GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));

	if (t_buffer.hasStorage())
	{
		const GLsizeiptr size{ m_count * static_cast<GLsizeiptr>(m_stride) };
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, size, nullptr, GL_DYNAMIC_STORAGE_BIT));
		GL_CHECK_ERROR(glCopyNamedBufferSubData(t_buffer.m_handle, m_handle, 0, 0, size));
	}
}

Buffer::Buffer(Buffer&& t_buffer) noexcept :
	m_handle{ std::exchange(t_buffer.m_handle, GL_NONE) },
	m_count{ std::exchange(t_buffer.m_count, 0) },
	m_stride{ std::exchange(t_buffer.m_stride, 0) }
{

}

// Destructor
Buffer::~Buffer()
{
	GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));
}

// Operators
Buffer& Buffer::operator=(const Buffer& t_buffer) noexcept
{
	if (this != &t_buffer)
	{
		reinitialize(t_buffer.m_count, t_buffer.m_stride);

		if (!t_buffer.hasStorage())
		{
			GL_CHECK_ERROR(glCopyNamedBufferSubData(t_buffer.m_handle, m_handle, 0, 0, m_count * static_cast<GLsizeiptr>(m_stride)));
		}
	}

	return *this;
}

Buffer& Buffer::operator=(Buffer&& t_buffer) noexcept
{
	if (this != &t_buffer)
	{
		GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));

		m_handle = std::exchange(t_buffer.m_handle, GL_NONE);
		m_count = std::exchange(t_buffer.m_count, 0);
		m_stride = std::exchange(t_buffer.m_stride, 0);
	}

	return *this;
}

// Functionality
void Buffer::bind(const GLenum t_target) const
{
	GL_CHECK_ERROR(glBindBuffer(t_target, m_handle));
}

void Buffer::bind(const GLenum t_target, const int t_location) const
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");

	GL_CHECK_ERROR(glBindBufferBase(t_target, static_cast<GLuint>(t_location), m_handle));
}

void Buffer::bind(const GLenum t_target, const int t_location, const int t_count) const
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");

	GL_CHECK_ERROR(glBindBufferRange(t_target, static_cast<GLuint>(t_location), m_handle, 0, t_count * static_cast<GLsizeiptr>(m_stride)));
}

void Buffer::bind(const GLenum t_target, const int t_location, const int t_offset, const int t_count) const
{
	STHE_ASSERT(t_location >= 0, "Location must be greater than or equal to 0");

	const GLsizeiptr stride{ m_stride };
	GL_CHECK_ERROR(glBindBufferRange(t_target, static_cast<GLuint>(t_location), m_handle, t_offset * stride, t_count * stride));
}

void Buffer::reinitialize(const int t_count, const int t_stride)
{
	STHE_ASSERT(t_count >= 0, "Count must be greater than or equal to 0");
	STHE_ASSERT(t_stride >= 0, "Stride must be greater than or equal to 0");
	STHE_ASSERT(t_count == 0 || t_stride > 0, "If count is greater than 0 than stride must be greater than 0");

	const GLsizeiptr currentSize{ m_count * static_cast<GLsizeiptr>(m_stride) };
	const GLsizeiptr newSize{ t_count * static_cast<GLsizeiptr>(t_stride) };

	if (currentSize != newSize)
	{
		if (hasStorage())
		{
			GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));
			GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		}

		if (newSize > 0)
		{
			GL_CHECK_ERROR(glNamedBufferStorage(m_handle, newSize, nullptr, GL_DYNAMIC_STORAGE_BIT));
		}
	}

	m_count = t_count;
	m_stride = t_stride;
}

void Buffer::reinterpret(const int t_stride)
{
	STHE_ASSERT(t_stride > 0, "Stride must be greater than 0");

	const GLsizeiptr size{ m_count * static_cast<GLsizeiptr>(m_stride) };

	STHE_ASSERT(size % t_stride == 0, "Size must be reinterpretable with stride");

	m_count = static_cast<int>(size / t_stride);
	m_stride = t_stride;
}

void Buffer::release()
{
	if (hasStorage())
	{
		GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));
		GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		m_count = 0;
	}
}

// Getters
GLuint Buffer::getHandle() const
{
	return m_handle;
}

int Buffer::getCount() const
{
	return m_count;
}

int Buffer::getStride() const
{
	return m_stride;
}

bool Buffer::hasStorage() const
{
	return m_count > 0;
}

}
}
