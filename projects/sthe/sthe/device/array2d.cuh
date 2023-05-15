#pragma once

#include <cuda_runtime.h>

namespace sthe
{
namespace device
{

template<typename T>
struct Array2D
{
	__forceinline__ __device__ T write(const int t_x, const int t_y, const T& t_value, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap)
	{
		surf2Dwrite<T>(t_value, surface, t_x * static_cast<int>(sizeof(T)), t_y, t_boundaryMode);
	}

	__forceinline__ __device__ T write(const int2& t_position, const T& t_value, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap)
	{
		surf2Dwrite<T>(t_value, surface, t_position.x * static_cast<int>(sizeof(T)), t_position.y, t_boundaryMode);
	}

	__forceinline__ __device__ T read(const int t_x, const int t_y, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap) const
	{
		return surf2Dread<T>(surface, t_x * static_cast<int>(sizeof(T)), t_y, t_boundaryMode);
	}

	__forceinline__ __device__ T read(const int2& t_position, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap) const
	{
		return surf2Dread<T>(surface, t_position.x * static_cast<int>(sizeof(T)), t_position.y, t_boundaryMode);
	}

	__forceinline__ __device__ T sample(const float t_x, const float t_y) const
	{
		return tex2D<T>(texture, t_x, t_y);
	}

	__forceinline__ __device__ T sample(const float2& t_position) const
	{
		return tex2D<T>(texture, t_position.x, t_position.y);
	}

	cudaSurfaceObject_t surface;
	cudaTextureObject_t texture;
};

}
}
