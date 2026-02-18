#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupAtomicInPlaceAvalanchingKernel(Array2D<half2> t_terrainArray, Buffer<half2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	t_terrainBuffer[cellIndex] = t_terrainArray.read(cell);
}

template <bool TUseAvalancheStrength>
__global__ void atomicInPlaceAvalanchingKernel(Buffer<half2> t_terrainBuffer, const Buffer<half> t_reptationBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float avalancheAngle{ __half2float(t_reptationBuffer[cellIndex]) };

	const float2 terrain{ __half22float2(t_terrainBuffer[cellIndex]) };
	const float height{ terrain.x + terrain.y };
	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
        const float2 nextTerrain{ __half22float2(t_terrainBuffer[nextCellIndices[i]]) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf((TUseAvalancheStrength ? c_parameters.avalancheStrength : 1.0f) * maxAvalanche /
										 (1.0f + maxAvalanche * rAvalancheSum), terrain.y) };


		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(&t_terrainBuffer[nextCellIndices[i]].y, __float2half(scale * avalanches[i]));
			}
		}

		atomicAdd(&t_terrainBuffer[cellIndex].y, __float2half(- avalancheSize));
	}
}

__global__ void finishAtomicInPlaceAvalanchingKernel(Array2D<half2> t_terrainArray, Buffer<half2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	t_terrainArray.write(cell, t_terrainBuffer[cellIndex]);
}

void avalanching(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
    Buffer<half2> terrainBuffer {reinterpret_cast<Buffer<half2>>(t_launchParameters.tmpBuffer)};
	Buffer<half> reptationBuffer{ t_launchParameters.tmpBuffer + 2 * t_simulationParameters.cellCount };

	switch (t_launchParameters.avalancheMode)
	{
	case AvalancheMode::Jacobi:
        // Removed, does nothing

		break;
	case AvalancheMode::AtomicBuffered:
        // Removed, does nothing

	    break;
	case AvalancheMode::AtomicInPlace:
		setupAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

		if (t_simulationParameters.avalancheStrength == 1.f) {
			for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
			{
				atomicInPlaceAvalanchingKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (terrainBuffer, reptationBuffer);
			}
		}
		else {
			for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
			{
				if (i % t_launchParameters.avalancheSoftIterationModulus == 0 ||
					i >= t_launchParameters.avalancheIterations - t_launchParameters.avalancheFinalSoftIterations)
				{
					atomicInPlaceAvalanchingKernel<true> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (terrainBuffer, reptationBuffer);
				}
				else
				{
					atomicInPlaceAvalanchingKernel<false> << <t_launchParameters.gridSize2D, t_launchParameters.blockSize2D >> > (terrainBuffer, reptationBuffer);
				}
			}
		}

		finishAtomicInPlaceAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

		break;
	case AvalancheMode::SharedAtomicInPlace:
        // Removed, does nothing

		break;
	case AvalancheMode::MixedInPlace:
        // Removed, does nothing

		break;
	case AvalancheMode::Multigrid:
	    // Removed, does nothing

	    break;
	case AvalancheMode::Taylor:
        // Removed, does nothing

		break;
	}
}

}
