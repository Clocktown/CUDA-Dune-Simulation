#include "array2d.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/util/io.hpp>
#include <sthe/gl/texture2d.hpp>
#include <sthe/gl/renderbuffer.hpp>
#include <cuda_runtime.h>
#include <vector>

namespace sthe
{
namespace cu
{

template<typename T>
void Array2D::upload(const std::vector<T>& t_source, const int t_width, const int t_height)
{
	upload(t_source.data(), 0, 0, t_width, t_height);
}

template<typename T>
void Array2D::upload(const std::vector<T>& t_source, const int t_x, const int t_y, const int t_width, const int t_height)
{
	upload(t_source.data(), t_x, t_y, t_width, t_height);
}

template<typename T>
inline void Array2D::upload(const T* const t_source, const int t_width, const int t_height)
{
	upload(t_source, 0, 0, t_width, t_height);
}

template<typename T>
inline void Array2D::upload(const T* const t_source, const int t_x, const int t_y, const int t_width, const int t_height)
{
	STHE_ASSERT(t_x >= 0, "X must be greater than or equal to 0");
	STHE_ASSERT(t_y >= 0, "Y must be greater than or equal to 0");
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(sizeof(T) == getStride(), "Size of T must be equal to stride");

	const size_t pitch{ static_cast<size_t>(t_width) * sizeof(T) };
	CU_CHECK_ERROR(cudaMemcpy2DToArray(m_handle, static_cast<size_t>(t_x) * sizeof(T), static_cast<size_t>(t_y), t_source, pitch, pitch, static_cast<size_t>(t_height), cudaMemcpyHostToDevice));
}

template<typename T>
void Array2D::download(std::vector<T>& t_destination, const int t_width, const int t_height) const
{
	download(t_destination.data(), 0, 0, t_width, t_height);
}

template<typename T>
void Array2D::download(std::vector<T>& t_destination, const int t_x, const int t_y, const int t_width, const int t_height) const
{
	download(t_destination.data(), t_x, t_y, t_width, t_height);
}

template<typename T>
inline void Array2D::download(T* const t_destination, const int t_width, const int t_height) const
{
	download(t_destination, 0, 0, t_width, t_height);
}

template<typename T>
inline void Array2D::download(T* const t_destination, const int t_x, const int t_y, const int t_width, const int t_height) const
{
	STHE_ASSERT(t_x >= 0, "X must be greater than or equal to 0");
	STHE_ASSERT(t_y >= 0, "Y must be greater than or equal to 0");
	STHE_ASSERT(t_width >= 0, "Width must be greater than or equal to 0");
	STHE_ASSERT(t_height >= 0, "Height must be greater than or equal to 0");
	STHE_ASSERT(sizeof(T) == getStride(), "Size of T must be equal to stride");

	const size_t pitch{ static_cast<size_t>(t_width) * sizeof(T) };
	CU_CHECK_ERROR(cudaMemcpy2DFromArray(t_destination, pitch, m_handle, static_cast<size_t>(t_x) * sizeof(T), static_cast<size_t>(t_y), pitch, static_cast<size_t>(t_height), cudaMemcpyDeviceToHost));
}

}
}
