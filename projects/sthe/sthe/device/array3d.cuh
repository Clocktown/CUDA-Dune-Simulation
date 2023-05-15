#pragma once

#include <cuda_runtime.h>

namespace sthe
{
namespace device
{

template<typename T>
struct Array3D
{
	__forceinline__ __device__ T write(const int t_x, const int t_y, const int t_z, const T& t_value, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap)
	{
		surf3Dwrite<T>(t_value, surface, t_x * static_cast<int>(sizeof(T)), t_y, t_z, t_boundaryMode);
	}

	__forceinline__ __device__ T write(const int3& t_position, const T& t_value, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap)
	{
		surf3Dwrite<T>(t_value, surface, t_position.x * static_cast<int>(sizeof(T)), t_position.y, t_position.z, t_boundaryMode);
	}

	__forceinline__ __device__ T read(const int t_x, const int t_y, const int t_z, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap) const
	{
		return surf3Dread<T>(surface, t_x * static_cast<int>(sizeof(T)), t_y, t_z, t_boundaryMode);
	}

	__forceinline__ __device__ T read(const int3& t_position, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap) const
	{
		return surf3Dread<T>(surface, t_position.x * static_cast<int>(sizeof(T)), t_position.y, t_position.z, t_boundaryMode);
	}

	__forceinline__ __device__ T sample(const float t_x, const float t_y, const float t_z) const
	{
		return tex3D<T>(texture, t_x, t_y, t_z);
	}

	__forceinline__ __device__ T sample(const float3& t_position) const
	{
		return tex3D<T>(texture, t_position.x, t_position.y, t_position.z);
	}

	cudaSurfaceObject_t surface;
	cudaTextureObject_t texture;
};

}
}
