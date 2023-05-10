#pragma once

#include "simulation_parameter.cuh"
#include "constant.cuh"
#include <sthe/device/vector_extension.cuh>
#include <device_launch_parameters.h>

namespace dunes
{

__forceinline__ __device__ int getLinearIndex(const int2& t_cell)
{
	return t_cell.x + t_cell.y * c_simulationParameter.gridSize.x;
}

__forceinline__ __device__ int2 getCell()
{
	return make_int2(threadIdx.x + blockIdx.x * blockDim.x,
					 threadIdx.y + blockIdx.y * blockDim.y);
}

__forceinline__ __device__ int2 getWrappedCell(const int2& t_cell)
{ 
	return int2{ (t_cell.x + c_simulationParameter.gridSize.x) % c_simulationParameter.gridSize.x,
				 (t_cell.y + c_simulationParameter.gridSize.y) % c_simulationParameter.gridSize.y };
}

__forceinline__ __device__ int2 getNearestCell(const float2& t_position)
{
	return make_int2(roundf(t_position));
}

__forceinline__ __device__ bool isOutside(const int2& t_cell)
{
	return t_cell.x < 0 || t_cell.y < 0 || t_cell.x >= c_simulationParameter.gridSize.x || t_cell.y >= c_simulationParameter.gridSize.y;
}

}
