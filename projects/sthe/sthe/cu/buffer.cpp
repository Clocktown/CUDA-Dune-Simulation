#include "buffer.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/gl/buffer.hpp>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <utility>

namespace sthe
{
namespace cu
{

// Constructors
Buffer::Buffer() :
	m_data{ nullptr },
	m_count{ 0 },
	m_stride{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	
}

Buffer::Buffer(const int t_count, const int t_stride) :
	m_count{ t_count },
	m_stride{ t_stride },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	STHE_ASSERT(t_count >= 0, "Count must be greater than or equal to 0");
	STHE_ASSERT(t_stride >= 0, "Stride must be greater than or equal to 0");

	if (m_count > 0)
	{
		CU_CHECK_ERROR(cudaMalloc(&m_data, static_cast<size_t>(m_count) * static_cast<size_t>(m_stride)));
	}
	else
	{
		m_data = nullptr;
	}
}

Buffer::Buffer(gl::Buffer& t_buffer, const unsigned int t_flags) :
	m_data{ nullptr },
	m_count{ 0 },
	m_stride{ 0 },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, t_buffer.getHandle(), t_flags));
}

Buffer::Buffer(const GLuint t_buffer, const unsigned int t_flags) :
	m_data{ nullptr },
	m_count{ 0 },
	m_stride{ 0 },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, t_buffer, t_flags));
}

Buffer::Buffer(const Buffer& t_buffer) noexcept :
	m_count{ t_buffer.m_count },
	m_stride{ t_buffer.m_stride },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	if (t_buffer.hasStorage())
	{
		const size_t size{ static_cast<size_t>(m_count) * static_cast<size_t>(m_stride) };
		CU_CHECK_ERROR(cudaMalloc(&m_data, size));
		CU_CHECK_ERROR(cudaMemcpy(m_data, t_buffer.m_data, size, cudaMemcpyDeviceToDevice));
	}
	else
	{
		m_data = nullptr;
	}
}

Buffer::Buffer(Buffer&& t_buffer) noexcept :
	m_data{ std::exchange(t_buffer.m_data, nullptr) },
	m_count{ std::exchange(t_buffer.m_count, 0) },
	m_stride{ std::exchange(t_buffer.m_stride, 0) },
	m_graphicsResource{ std::exchange(t_buffer.m_graphicsResource, nullptr) },
	m_isMapped{ std::exchange(t_buffer.m_isMapped, false) }
{

}

// Destructor
Buffer::~Buffer()
{
	release();
}

// Operators
Buffer& Buffer::operator=(const Buffer& t_buffer) noexcept
{
	if (this != &t_buffer)
	{
		reinitialize(t_buffer.m_count, t_buffer.m_stride);

		if (t_buffer.hasStorage())
		{
			CU_CHECK_ERROR(cudaMemcpy(m_data, t_buffer.m_data, static_cast<size_t>(m_count) * static_cast<size_t>(m_stride), cudaMemcpyDeviceToDevice));
		}
	}

	return *this;
}

Buffer& Buffer::operator=(Buffer&& t_buffer) noexcept
{
	if (this != &t_buffer)
	{
		if (!m_isMapped)
		{
			CU_CHECK_ERROR(cudaFree(m_data));
		}

		m_data = std::exchange(t_buffer.m_data, nullptr);
		m_count = std::exchange(t_buffer.m_count, 0);
		m_stride = std::exchange(t_buffer.m_stride, 0);
		m_graphicsResource = std::exchange(t_buffer.m_graphicsResource, nullptr);
		m_isMapped = std::exchange(t_buffer.m_isMapped, false);
	}

	return *this;
}

// Functionality
void Buffer::reinitialize(const int t_count, const int t_stride)
{
	STHE_ASSERT(t_count >= 0, "Count must be greater than or equal to 0");
	STHE_ASSERT(t_stride >= 0, "Stride must be greater than or equal to 0");
	STHE_ASSERT(t_count == 0 || t_stride > 0, "If count is greater than 0 than stride must be greater than 0");

	const size_t currentSize{ static_cast<size_t>(m_count) * static_cast<size_t>(m_stride) };
	const size_t newSize{ static_cast<size_t>(t_count) * static_cast<size_t>(t_stride) };
	m_count = t_count;
	m_stride = t_stride;

	if (m_graphicsResource != nullptr)
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = nullptr;
	}
	else if (currentSize == newSize)
	{
		return;
	}

	if (m_count > 0)
	{
		CU_CHECK_ERROR(cudaMalloc(&m_data, newSize));
	}
	else
	{
		m_data = nullptr;
	}
}

void Buffer::reinitialize(gl::Buffer& t_buffer, const unsigned int t_flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, t_buffer.getHandle(), t_flags));
}

void Buffer::reinitialize(const GLuint t_buffer, const unsigned int t_flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, t_buffer, t_flags));
}

void Buffer::reinterpret(const int t_stride)
{
	STHE_ASSERT(t_stride > 0, "Stride must be greater than 0");

	const size_t stride{ static_cast<size_t>(t_stride) };
	const size_t size{ static_cast<size_t>(m_count) * static_cast<size_t>(m_stride) };
	
	STHE_ASSERT(size % stride == 0, "Size must be reinterpretable with stride");

	m_count = static_cast<int>(size / stride);
	m_stride = t_stride;
}

void Buffer::release()
{
	if (m_graphicsResource != nullptr)
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = nullptr;
	}
	else
	{
		CU_CHECK_ERROR(cudaFree(m_data));
	}

	m_data = nullptr;
	m_count = 0;
}

void Buffer::map(const int t_stride)
{
	STHE_ASSERT(m_graphicsResource != nullptr, "Graphics resource must be registered");
	STHE_ASSERT(!m_isMapped, "Buffer must be unmapped");

	CU_CHECK_ERROR(cudaGraphicsMapResources(1, &m_graphicsResource));

	size_t size;
	CU_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&m_data, &size, m_graphicsResource));

	const size_t stride{ static_cast<size_t>(t_stride) };

	STHE_ASSERT(size % stride == 0, "Size must be reinterpretable with stride");

	m_count = static_cast<int>(size / stride);
	m_stride = t_stride;
	m_isMapped = true;
}

void Buffer::map()
{
	map(m_stride);
}

void Buffer::unmap()
{
	STHE_ASSERT(m_graphicsResource != nullptr, "Graphics resource must be registered");
	STHE_ASSERT(m_isMapped, "Buffer must be mapped");

	CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));

	m_count = 0;
	m_isMapped = false;
}

// Getters
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
