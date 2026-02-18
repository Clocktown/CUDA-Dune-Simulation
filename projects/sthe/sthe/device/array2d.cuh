#pragma once

#include <sthe/device/vector_extension.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace sthe
{
namespace device
{

template<typename T>
struct Array2D
{
	__forceinline__ __device__ void write(const int t_x, const int t_y, const T& t_value, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap)
	{
        if constexpr(true == std::is_same_v<T, half2>)
        {
            surf2Dwrite(
                    *reinterpret_cast<const uint32_t*>(&t_value), surface, t_x * static_cast<int>(sizeof(T)), t_y, t_boundaryMode);
		}
        else if constexpr(true == std::is_same_v<T, half4>)
        {
            surf2Dwrite(*reinterpret_cast<const float2*>(&t_value),
                        surface,
                        t_x * static_cast<int>(sizeof(T)),
                        t_y,
                        t_boundaryMode);
        }
        else
        {
            surf2Dwrite<T>(
                    t_value, surface, t_x * static_cast<int>(sizeof(T)), t_y, t_boundaryMode);
        }
}

	__forceinline__ __device__ void write(const int2& t_position, const T& t_value, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap)
	{
            if constexpr(true == std::is_same_v<T, half2>)
            {
                surf2Dwrite(*reinterpret_cast<const uint32_t*>(&t_value),
                            surface,
                            t_position.x * static_cast<int>(sizeof(T)),
                            t_position.y,
                            t_boundaryMode);
            }
            else if constexpr(true == std::is_same_v<T, half4>)
            {
                surf2Dwrite(*reinterpret_cast<const float2*>(&t_value),
                            surface,
                            t_position.x * static_cast<int>(sizeof(T)),
                            t_position.y,
                            t_boundaryMode);
            }
            else
            {
                surf2Dwrite<T>(t_value,
                               surface,
                               t_position.x * static_cast<int>(sizeof(T)),
                               t_position.y,
                               t_boundaryMode);
            }
	}

	__forceinline__ __device__ T read(const int t_x, const int t_y, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap) const
	{
        if constexpr(true == std::is_same_v<T, half2>)
        {
            uint32_t val = surf2Dread<uint32_t>(surface, t_x * static_cast<int>(sizeof(T)), t_y, t_boundaryMode);
            return *reinterpret_cast<half2*>(&val);
        }
        else if constexpr(true == std::is_same_v<T, half4>)
        {
            float2 val = surf2Dread<float2>(surface,
                                            t_x * static_cast<int>(sizeof(T)),
                                            t_y,
                                            t_boundaryMode);
            return *reinterpret_cast<half4*>(&val);
        }
        else
        {
            return surf2Dread<T>(
                    surface, t_x * static_cast<int>(sizeof(T)), t_y, t_boundaryMode);
        }
	}

	__forceinline__ __device__ T read(const int2& t_position, const cudaSurfaceBoundaryMode t_boundaryMode = cudaBoundaryModeTrap) const
	{
            if constexpr(true == std::is_same_v<T, half2>)
            {
                uint32_t val = surf2Dread<uint32_t>(surface,
                                                t_position.x * static_cast<int>(sizeof(T)),
                                                t_position.y,
                                                t_boundaryMode);
                return *reinterpret_cast<half2*>(&val);
            }
            else if constexpr(true == std::is_same_v<T, half4>)
            {
                float2 val = surf2Dread<float2>(surface,
                                                    t_position.x * static_cast<int>(sizeof(T)),
                                                    t_position.y,
                                                    t_boundaryMode);
                return *reinterpret_cast<half4*>(&val);
            }
            else
            {
                return surf2Dread<T>(surface,
                                     t_position.x * static_cast<int>(sizeof(T)),
                                     t_position.y,
                                     t_boundaryMode);
            }
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
