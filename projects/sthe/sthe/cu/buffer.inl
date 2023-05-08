#include "buffer.hpp"
#include <sthe/config/debug.hpp>
#include <cuda_runtime.h>
#include <vector>

namespace sthe
{
namespace cu
{

// Constructor
template<typename T>
Buffer::Buffer(const std::vector<T>& t_source) :
	m_count{ static_cast<int>(t_source.size()) },
	m_stride{ sizeof(T) },
	m_isMapped{ false }
{
	if (!t_source.empty())
	{
		const size_t size{ t_source.size() * sizeof(T) };

		CU_CHECK_ERROR(cudaMalloc(&m_data, size));
		CU_CHECK_ERROR(cudaMemcpy(m_data, t_source.data(), size, cudaMemcpyHostToDevice));
	}
}

// Functionality
template<typename T>
inline void Buffer::reinitialize(const std::vector<T>& t_source)
{
	reinitialize(static_cast<int>(t_source.size()), sizeof(T));

	if (!t_source.empty())
	{
		CU_CHECK_ERROR(cudaMemcpy(m_data, t_source.data(), t_source.size() * sizeof(T), cudaMemcpyHostToDevice));
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
	STHE_ASSERT(t_offset >= 0, "Offset must be greater than or equal to 0");
	STHE_ASSERT(t_count >= 0, "Count must be greater than or equal to 0");
	STHE_ASSERT(sizeof(T) == m_stride, "Size of T must be equal to stride");
	
	CU_CHECK_ERROR(cudaMemcpy(static_cast<T*>(m_data) + static_cast<size_t>(t_offset), t_source, static_cast<size_t>(t_count) * sizeof(T), cudaMemcpyHostToDevice));
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
	STHE_ASSERT(t_offset >= 0, "Offset must be greater than or equal to 0");
	STHE_ASSERT(t_count >= 0, "Count must be greater than or equal to 0");
	STHE_ASSERT(sizeof(T) == m_stride, "Size of T must be equal to stride");

	CU_CHECK_ERROR(cudaMemcpy(t_destination, static_cast<T*>(m_data) + static_cast<size_t>(t_offset), static_cast<size_t>(t_count) * sizeof(T), cudaMemcpyDeviceToHost));
}

// Getters
template<typename T>
inline const T* Buffer::getData() const
{
	STHE_ASSERT(sizeof(T) == m_stride, "Size of T must be equal to stride");

	return static_cast<T*>(m_data);
}

template<typename T>
inline T* Buffer::getData()
{
	STHE_ASSERT(sizeof(T) == m_stride, "Size of T must be equal to stride");

	return static_cast<T*>(m_data);
}

}
}
