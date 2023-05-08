#include "array3d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <cuda_runtime.h>
#include <vector>

namespace sthe
{
namespace cu
{

template<typename T>
void Array3D::upload(const std::vector<T>& t_source, const int t_width, const int t_height, const int t_depth)
{
	upload(t_source.data(), 0, 0, 0, t_width, t_height, t_depth);
}

template<typename T>
void Array3D::upload(const std::vector<T>& t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth)
{
	upload(t_source.data(), t_x, t_y, t_z, t_width, t_height, t_depth);
}

template<typename T>
inline void Array3D::upload(const T* const t_source, const int t_width, const int t_height, const int t_depth)
{
	upload(t_source, 0, 0, 0, t_width, t_height, t_depth);
}

template<typename T>
inline void Array3D::upload(const T* const t_source, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth)
{
	STHE_ASSERT(t_x >= 0, "X must be greater than or equal to 0");
	STHE_ASSERT(t_y >= 0, "Y must be greater than or equal to 0");
	STHE_ASSERT(t_z >= 0, "Z must be greater than or equal to 0");
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_depth >= 0, "Depth must be greater than or equal to 0");
	STHE_ASSERT(sizeof(T) == getStride(), "Size of T must be equal to stride");

	const cudaPos offset{ static_cast<size_t>(t_x), static_cast<size_t>(t_y), static_cast<size_t>(t_z) };
	const cudaExtent extent{ static_cast<size_t>(t_width), static_cast<size_t>(t_height), static_cast<size_t>(t_depth) };
	const cudaMemcpy3DParms parameter{ .srcPtr{ make_cudaPitchedPtr(const_cast<void*>(t_source), extent.width * sizeof(T), extent.width, extent.height)},
									   .dstArray{ m_handle },
									   .dstPos{ offset },
									   .extent{ extent },
									   .kind{ cudaMemcpyHostToDevice } };

	CU_CHECK_ERROR(cudaMemcpy3D(&parameter));
}

template<typename T>
void Array3D::download(std::vector<T>& t_destination, const int t_width, const int t_height, const int t_depth) const
{
	download(t_destination.data(), 0, 0, 0, t_width, t_height, t_depth);
}

template<typename T>
void Array3D::download(std::vector<T>& t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth) const
{
	download(t_destination.data(), t_x, t_y, t_z, t_width, t_height, t_depth);
}

template<typename T>
inline void Array3D::download(T* const t_destination, const int t_width, const int t_height, const int t_depth) const
{
	download(t_destination, 0, 0, 0, t_width, t_height, t_depth);
}

template<typename T>
inline void Array3D::download(T* const t_destination, const int t_x, const int t_y, const int t_z, const int t_width, const int t_height, const int t_depth) const
{
	STHE_ASSERT(t_x >= 0, "X must be greater than or equal to 0");
	STHE_ASSERT(t_y >= 0, "Y must be greater than or equal to 0");
	STHE_ASSERT(t_z >= 0, "Z must be greater than or equal to 0");
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(t_depth >= 0, "Depth must be greater than or equal to 0");
	STHE_ASSERT(sizeof(T) == getStride(), "Size of T must be equal to stride");

	const cudaPos offset{ static_cast<size_t>(t_x), static_cast<size_t>(t_y), static_cast<size_t>(t_z) };
	const cudaExtent extent{ static_cast<size_t>(t_width), static_cast<size_t>(t_height), static_cast<size_t>(t_depth) };
	const cudaMemcpy3DParms parameter{ .srcArray{ m_handle },
									   .srcPos{ offset },
									   .dstPtr{ make_cudaPitchedPtr(t_destination, extent.width * sizeof(T), extent.width, extent.height)},
									   .extent{ extent },
									   .kind{ cudaMemcpyDeviceToHost } };

	CU_CHECK_ERROR(cudaMemcpy3D(&parameter));
}

}
}
