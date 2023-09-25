#include "continuous_reptation.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <sthe/config/debug.hpp>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <cstdio>

namespace dunes
{

__global__ void setupContinuousReptationKernel(Buffer<float> t_reptationBuffer)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < c_parameters.cellCount; cellIndex += stride)
	{
		t_reptationBuffer[cellIndex] = 0.0f;
	}
}

__global__ void continuousReptationKernel(const Array2D<float2> t_terrainArray, Buffer<float> t_slabBuffer, Buffer<float> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float2 terrain{ t_terrainArray.read(cell) };
	const float height{ terrain.x + terrain.y };

	float slab{ t_slabBuffer[cellIndex] };

	float change{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		const float nextSlab{ t_slabBuffer[getCellIndex(nextCell)] };

		const float2 nextTerrain{ t_terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ (nextHeight - height) * c_parameters.rGridScale * c_rDistances[i]};
		const float heightScale = 1.f;// fmaxf(c_parameters.avalancheAngle - abs(heightDifference), 0.f) / c_parameters.avalancheAngle;

		// Enforce symmetric additive and subtractive changes, avoiding any atomics
		float step = fmaxf(0.5f * heightScale * (slab + nextSlab) * c_parameters.reptationStrength, 0.f);
        change += signbit(heightDifference) ? -fminf(step, terrain.y) : fminf(step, nextTerrain.y);
	}

	t_reptationBuffer[cellIndex] = change * 0.125;
}

__global__ void finishContinuousReptationKernel(Array2D<float2> t_terrainArray, Buffer<float> t_reptationBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			float2 terrain{ t_terrainArray.read(cell) };
			terrain.y += t_reptationBuffer[getCellIndex(cell)];

			t_terrainArray.write(cell, terrain);
		}
	}
}

void continuousReptation(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	if (t_simulationParameters.reptationStrength > 0.f) {
		Buffer<float> reptationBuffer{ t_launchParameters.tmpBuffer + t_simulationParameters.cellCount };

		//if (t_simulationParameters.timestep == 0) {
		//	setupContinuousReptationKernel << <t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D >> > (reptationBuffer);
		//}
		continuousReptationKernel << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (t_launchParameters.terrainArray, t_launchParameters.tmpBuffer, reptationBuffer);
		finishContinuousReptationKernel << <t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D >> > (t_launchParameters.terrainArray, reptationBuffer);
	}
}

}
