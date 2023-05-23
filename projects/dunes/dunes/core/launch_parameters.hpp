#pragma once

#include "simulation_parameters.hpp"
#include <cuda_runtime.h>

namespace dunes
{

enum class TimeMode : unsigned char
{
	DeltaTime, FixedDeltaTime
};

enum class AvalancheMode : unsigned char
{
	AtomicBuffered, AtomicInPlace, SharedAtomicInPlace, MixedInPlace
};

struct LaunchParameters
{
	unsigned int blockSize1D;
	dim3 blockSize2D;
	unsigned int gridSize1D;
	dim3 gridSize2D;

	unsigned int optimalBlockSize1D;
	dim3 optimalBlockSize2D;
	unsigned int optimalGridSize1D;
	dim3 optimalGridSize2D;

	AvalancheMode avalancheMode{ AvalancheMode::AtomicInPlace };
	int avalancheIterations{ 50 };
	int avalancheSoftIterationModulus{ 10 };
	int avalancheFinalSoftIterations{ 5 };
	TimeMode timeMode{ TimeMode::DeltaTime };

	Array2D<float2> terrainArray;
	Array2D<float2> windArray;
	Array2D<float4> resistanceArray; // .x = wind shadow, .y = vegetation, .z = erosion
	Buffer<float> tmpBuffer; // 8 * gridSize.x * gridSize.y
};

}
