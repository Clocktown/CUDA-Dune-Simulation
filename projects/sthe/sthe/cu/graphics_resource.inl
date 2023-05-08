#include "graphics_resource.hpp"
#include <sthe/config/debug.hpp>
#include <cuda_runtime.h>

namespace sthe
{
namespace cu
{

// Functionality
template<typename T>
inline void GraphicsResource::mapBuffer(T*& t_pointer)
{
	map();
	[[maybe_unused]] size_t size;
	CU_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&t_pointer), &size, m_handle));

	STHE_ASSERT(size % sizeof(T) == 0, "Size must be reinterpretable with stride");
}

template<typename T>
inline void GraphicsResource::mapBuffer(T*& t_pointer, int& t_count)
{
	map();
	size_t size;
	CU_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&t_pointer), &size, m_handle));

	STHE_ASSERT(size % sizeof(T) == 0, "Size must be reinterpretable with stride");

	t_count = static_cast<int>(size / sizeof(T));
}

}
}
