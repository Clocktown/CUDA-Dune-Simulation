#pragma once

#include "constants.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <device_launch_parameters.h>

namespace dunes
{

	__forceinline__ __device__ int getThreadIndex1D()
	{
		return static_cast<int>(threadIdx.x);
	}

	__forceinline__ __device__ int2 getThreadIndex2D()
	{
		return int2{ static_cast<int>(threadIdx.x),
					 static_cast<int>(threadIdx.y) };
	}

	__forceinline__ __device__ int getBlockIndex1D()
	{
		return static_cast<int>(blockIdx.x);
	}

	__forceinline__ __device__ int2 getBlockIndex2D()
	{
		return int2{ static_cast<int>(blockIdx.x),
					 static_cast<int>(blockIdx.y) };
	}

	__forceinline__ __device__ int getBlockSize1D()
	{
		return static_cast<int>(blockDim.x);
	}

	__forceinline__ __device__ int2 getBlockSize2D()
	{
		return int2{ static_cast<int>(blockDim.x),
					 static_cast<int>(blockDim.y) };
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

	__forceinline__ __device__ int getGlobalIndex1D()
	{
		return static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
	}

	__forceinline__ __device__ int2 getGlobalIndex2D()
	{
		return int2{ static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x),
					 static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y) };
	}

	__forceinline__ __device__ int2 getWrappedCell(const int2& t_cell, const int2& t_gridSize = c_parameters.gridSize)
	{
		int2 sign;
		sign.x = (int)(t_cell.x < 0) - (int)(t_cell.x >= t_gridSize.x);
		sign.y = (int)(t_cell.y < 0) - (int)(t_cell.y >= t_gridSize.y);

		return t_cell + sign * t_gridSize;

		//return int2{ (t_cell.x + t_gridSize.x) % t_gridSize.x,
		//			   (t_cell.y + t_gridSize.y) % t_gridSize.y };
	}

	__forceinline__ __device__ int2 getNearestCell(const float2& t_position)
	{
		return make_int2(roundf(t_position));
	}

	__forceinline__ __device__ int getCellIndex(const int2& t_cell, const int2 t_gridSize = c_parameters.gridSize)
	{
		return t_cell.x + t_cell.y * t_gridSize.x;
	}

	__forceinline__ __device__ bool isOutside(const int t_cellIndex, const int t_cellCount = c_parameters.cellCount)
	{
		return t_cellIndex >= t_cellCount;
	}

	__forceinline__ __device__ bool isOutside(const int2& t_cell, const int2& t_gridSize = c_parameters.gridSize)
	{
		return t_cell.x >= t_gridSize.x || t_cell.y >= t_gridSize.y;
	}

	template<bool TUseBilinear>
	__forceinline__ __device__  float2 sampleLinearOrNearest(const Array2D<float2>& arr, const float2& pos) {
		if constexpr (TUseBilinear) {
			return arr.sample(pos);
		}
		else {
			return arr.read(getWrappedCell(getNearestCell(pos - 0.5f)));
		}
	}

}
