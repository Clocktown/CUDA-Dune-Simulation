#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupReptationKernel(Buffer<float> t_reptationBuffer)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < c_parameters.cellCount; cellIndex += stride)
	{
		t_reptationBuffer[cellIndex] = 0.0f;
	}
}

__global__ void reptationKernel(const Array2D<float2> t_terrainArray, Buffer<float> t_slabBuffer, Buffer<float> t_reptationBuffer)
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

	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		nextCellIndices[i] = getCellIndex(nextCell);

		const float2 nextTerrain{ t_terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - c_parameters.avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
	}

	if (avalancheSum > 0.0f)
	{
		const float reptation{ clamp(c_parameters.reptationStrength * slab, 0.0f, slab) };
		slab -= reptation;

		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float scale{ reptation * rAvalancheSum };
		
		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(t_reptationBuffer + nextCellIndices[i], scale * avalanches[i]);
			}
		}
	}

	atomicAdd(t_reptationBuffer + cellIndex, slab);
}

__global__ void finishReptationKernel(Array2D<float2> t_terrainArray, Buffer<float> t_reptationBuffer)
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

void reptation(const LaunchParameters& t_launchParameters)
{
	Buffer<float> reptationBuffer{ t_launchParameters.tmpBuffer + t_launchParameters.multigrid[0].cellCount };

	setupReptationKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(reptationBuffer);
	reptationKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer, reptationBuffer);
	finishReptationKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, reptationBuffer);
}

}
