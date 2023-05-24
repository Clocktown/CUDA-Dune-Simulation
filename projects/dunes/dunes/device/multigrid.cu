#include "multigrid.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <sthe/config/debug.hpp>
#include <cstdio>

namespace dunes
{

__global__ void setupMultigridAvalanchingKernel(const Array2D<float2> t_terrainArray, const MultigridLevel t_level)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };
	
	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			t_level.terrainBuffer[cellIndex] = t_terrainArray.read(cell);
			t_level.fluxBuffer[cellIndex] = 0.0f;
			t_level.avalancheBuffer[cellIndex] = 0.0f;
		}
	}
}

__global__ void multigridAvalanchingKernel(const MultigridLevel t_level)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell, t_level.gridSize))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell, t_level.gridSize) };
	const float2 terrain{ t_level.terrainBuffer[cellIndex] + make_float2(0.0f, t_level.fluxBuffer[cellIndex]) };
	const float height{ terrain.x + terrain.y };

	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i], t_level.gridSize) };
		nextCellIndices[i] = getCellIndex(nextCell, t_level.gridSize);

		const float2 nextTerrain{ t_level.terrainBuffer[nextCellIndices[i]] + make_float2(0.0f, t_level.fluxBuffer[nextCellIndices[i]]) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - c_parameters.avalancheAngle * c_distances[i] * t_level.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ fminf(c_parameters.avalancheStrength * maxAvalanche /
										 (1.0f + maxAvalanche * rAvalancheSum), terrain.y) };

		atomicAdd(t_level.avalancheBuffer + cellIndex, -avalancheSize);

		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				atomicAdd(t_level.avalancheBuffer + nextCellIndices[i], scale * avalanches[i]);
			}
		}
	}
}

__global__ void applyMultigridAvalanchingKernel(const MultigridLevel t_level)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < t_level.cellCount; cellIndex += stride)
	{
		t_level.fluxBuffer[cellIndex] += t_level.avalancheBuffer[cellIndex];
		t_level.avalancheBuffer[cellIndex] = 0.0f;
	}
}

__global__ void downscaleMultigridLevelKernel(const MultigridLevel t_level, const MultigridLevel t_nextLevel)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 nextCell;
	
	for (nextCell.x = index.x; nextCell.x < t_nextLevel.gridSize.x; nextCell.x += stride.x)
	{
		for (nextCell.y = index.y; nextCell.y < t_nextLevel.gridSize.y; nextCell.y += stride.y)
		{
			const int2 cell{ 2 * nextCell };

			const int nextCellIndex{ getCellIndex(nextCell, t_nextLevel.gridSize) };
			float2 nextTerrain{ 0.0f, 0.0f };
			float nextFlux{ 0.0f };

			for (int x{ 0 }; x <= 1; ++x)
			{
				for (int y{ 0 }; y <= 1; ++y)
				{
					const int cellIndex{ getCellIndex(getWrappedCell(cell + make_int2(x, y), t_level.gridSize), t_level.gridSize) };
					
					nextTerrain += t_level.terrainBuffer[cellIndex];
					nextFlux += t_level.fluxBuffer[cellIndex];
				}
			}

			nextTerrain *= 0.25f;
			nextFlux *= 0.25f;

			nextFlux = fmaxf(nextTerrain.y + nextFlux, 0.0f) - nextTerrain.y;
			
			t_nextLevel.terrainBuffer[nextCellIndex] = nextTerrain;
			t_nextLevel.fluxBuffer[nextCellIndex] = nextFlux;
			t_nextLevel.avalancheBuffer[nextCellIndex] = 0.0f;
		}
	}
}

__global__ void upscaleMultigridLevelKernel(const MultigridLevel t_level, const MultigridLevel t_nextLevel)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 nextCell;

	for (nextCell.x = index.x; nextCell.x < t_nextLevel.gridSize.x; nextCell.x += stride.x)
	{
		for (nextCell.y = index.y; nextCell.y < t_nextLevel.gridSize.y; nextCell.y += stride.y)
		{
			const int2 cell{ nextCell / 2 };
			const int cellIndex{ getCellIndex(cell, t_level.gridSize) };

			const int nextCellIndex{ getCellIndex(nextCell, t_nextLevel.gridSize) };
			const float2 nextTerrain{ t_nextLevel.terrainBuffer[nextCellIndex] };
			float nextFlux{ t_level.fluxBuffer[cellIndex] };

			nextFlux = fmaxf(nextTerrain.y + nextFlux, 0.0f) - nextTerrain.y;

			t_nextLevel.fluxBuffer[nextCellIndex] = nextFlux;
			t_nextLevel.avalancheBuffer[nextCellIndex] = 0.0f;
		}
	}
}

__global__ void finishMultigridAvalanchingKernel(Array2D<float2> t_terrainArray, const MultigridLevel t_level)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			t_terrainArray.write(cell, t_terrainArray.read(cell) + make_float2(0.0f, t_level.fluxBuffer[cellIndex]));
		}
	}
}

void multigrid(const LaunchParameters& t_launchParameters)
{
	setupMultigridAvalanchingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.multigrid[0]);
	
	for (int i{ 0 }; i < t_launchParameters.multigridLevelCount - 1; ++i)
	{
		for (int j{ 0 }; j < t_launchParameters.multigridPresweepCount; ++j)
	    {
		    multigridAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.multigrid[i]);
		    applyMultigridAvalanchingKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.multigrid[i]);
	    }

		downscaleMultigridLevelKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.multigrid[i], t_launchParameters.multigrid[i + 1]);
	}
	
	const int iterations{ t_launchParameters.avalancheIterations / static_cast<int>(pow(2.0f, static_cast<float>(t_launchParameters.multigridLevelCount))) };

	for (int i{ t_launchParameters.multigridLevelCount - 1 }; i >= 1; --i)
	{
		for (int j{ 0 }; j < iterations * (i + 1); ++j)
	    {
		    multigridAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.multigrid[i]);
		    applyMultigridAvalanchingKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.multigrid[i]);
	    }

		upscaleMultigridLevelKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.multigrid[i], t_launchParameters.multigrid[i - 1]);
	}

	for (int i{ 0 }; i < iterations; ++i)
	{
		multigridAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.multigrid[0]);
		applyMultigridAvalanchingKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.multigrid[0]);
	}

	finishMultigridAvalanchingKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.multigrid[0]);
}

}