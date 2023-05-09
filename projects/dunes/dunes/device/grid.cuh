#pragma once

#include <cuda_runtime.h>
#include <sthe/device/vector_extension.hpp>

namespace dunes
{
namespace device
{

__forceinline__ __device__ int getLinearIndex(const int2& t_cell);
__forceinline__ __device__ int2 getWrappedCell(const int2& t_cell);
__forceinline__ __device__ int2 getNearestCell(const float2& t_position);
__forceinline__ __device__ bool isOutside(const int2& t_cell);

}
}

