#pragma once

#include "constants.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <device_launch_parameters.h>

namespace dunes
{

__forceinline__ __device__ int getGlobalIndex1D()
{
	return static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
}

__forceinline__ __device__ int2 getGlobalIndex2D()
{
	return int2{ static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x),
				 static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y) };
}

__forceinline__ __device__ int getGridStride1D()
{
	return static_cast<int>(blockDim.x * gridDim.x);
}

__forceinline__ __device__ int2 getGridStride2D()
{
	return int2{ static_cast<int>(blockDim.x * gridDim.x),
				 static_cast<int>(blockDim.y * gridDim.y) };
}

__forceinline__ __device__ int2 getWrappedCell(const int2& t_cell)
{
	return int2{ (t_cell.x + c_parameters.gridSize.x) % c_parameters.gridSize.x,
				 (t_cell.y + c_parameters.gridSize.y) % c_parameters.gridSize.y };
}

__forceinline__ __device__ int2 getNearestCell(const float2& t_position)
{
	return make_int2(roundf(t_position));
}

__forceinline__ __device__ int getCellIndex(const int2& t_cell)
{
	return t_cell.x + t_cell.y * c_parameters.gridSize.x;
}

__forceinline__ __device__ bool isOutside(const int t_cellIndex)
{
	return t_cellIndex >= c_parameters.cellCount;
}

__forceinline__ __device__ bool isOutside(const int2& t_cell)
{
	return t_cell.x < 0 || t_cell.y < 0 || t_cell.x >= c_parameters.gridSize.x || t_cell.y >= c_parameters.gridSize.y;
}

}
