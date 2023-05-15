#pragma once

#include <sthe/device/buffer.cuh>
#include <sthe/device/array2d.cuh>
#include <cuda_runtime.h>

namespace dunes
{

template<typename T>
using Array2D = sthe::device::Array2D<T>;

template<typename T>
using Buffer = sthe::device::Buffer<T>;

struct LaunchParameters
{
	unsigned int blockSize1D;
	dim3 blockSize2D;
	unsigned int gridSize1D;
	dim3 gridSize2D;
	unsigned int optimalGridSize1D;
	dim3 optimalGridSize2D;

	int avalancheIterations{ 50 };

	Array2D<float2> terrainArray;
	Array2D<float2> windArray;
	Array2D<float4> resistanceArray; // .x = wind shadow, .y = vegetation, .z = erosion
	Buffer<float> tmpBuffer; // 8 * gridSize.x * gridSize.y
};

}
