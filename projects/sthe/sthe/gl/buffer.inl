#include "buffer.hpp"
#include <sthe/config/debug.hpp>
#include <glad/glad.h>
#include <vector>

namespace sthe
{
namespace gl
{

// Constructor
template<typename T>
inline Buffer::Buffer(const std::vector<T>& t_source) :
	m_count{ static_cast<int>(t_source.size()) },
	m_stride{ sizeof(T) }
{
	if (!t_source.empty())
	{
		GL_CHECK_ERROR(glCreateBuffers(1, &m_handle));
		GL_CHECK_ERROR(glNamedBufferStorage(m_handle, m_count * static_cast<GLsizeiptr>(sizeof(T)), t_source.data(), GL_DYNAMIC_STORAGE_BIT));
	}
}

// Functionality
template<typename T>
void Buffer::reinitialize(const std::vector<T>& t_source)
{
	reinitialize(static_cast<int>(t_source.size()), sizeof(T));

	if (!t_source.empty())
	{
		GL_CHECK_ERROR(glNamedBufferSubData(m_handle, 0, m_count * static_cast<GLsizeiptr>(sizeof(T)), t_source.data()));
	}
}

template<typename T>
inline void Buffer::upload(const std::vector<T>& t_source)
{
	upload(t_source.data(), 0, static_cast<int>(t_source.size()));
}

template<typename T>
inline void Buffer::upload(const std::vector<T>& t_source, const int t_count)
{
	upload(t_source.data(), 0, t_count);
}

template<typename T>
inline void Buffer::upload(const std::vector<T>& t_source, const int t_offset, const int t_count)
{
	upload(t_source.data(), t_offset, t_count);
}

template<typename T>
inline void Buffer::upload(const T* const t_source, const int t_count)
{
	upload(t_source, 0, t_count);
}

template<typename T>
inline void Buffer::upload(const T* const t_source, const int t_offset, const int t_count)
{
	STHE_ASSERT(sizeof(T) == m_stride, "Size of T must be equal to stride");

	constexpr GLsizeiptr stride{ sizeof(T) };
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, t_offset * stride, t_count * stride, t_source));
}

template<typename T>
inline void Buffer::download(std::vector<T>& t_destination) const
{
	download(t_destination.data(), 0, static_cast<int>(t_destination.size()));
}

template<typename T>
inline void Buffer::download(std::vector<T>& t_destination, const int t_count) const
{
	download(t_destination.data(), 0, t_count);
}

template<typename T>
inline void Buffer::download(std::vector<T>& t_destination, const int t_offset, const int t_count) const
{
	download(t_destination.data(), t_offset, t_count);
}

template<typename T>
inline void Buffer::download(T* const t_destination, const int t_count) const
{
	download(t_destination, 0, t_count);
}

template<typename T>
inline void Buffer::download(T* const t_destination, const int t_offset, const int t_count) const
{
	STHE_ASSERT(sizeof(T) == m_stride, "Size of T must be equal to stride");

	constexpr GLsizeiptr stride{ sizeof(T) };
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, t_offset * stride, t_count * stride, t_destination));
}

}
}
