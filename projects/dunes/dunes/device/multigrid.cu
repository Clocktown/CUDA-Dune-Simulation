#include "multigrid.cuh"
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
namespace solver
{

__global__ void setup(const Array2D<float2> t_terrainArray, const MultigridLevel t_level)
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

__global__ void smooth(const MultigridLevel t_level, const Array2D<float4> t_resistanceArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell, t_level.gridSize))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell, t_level.gridSize) };
	const float2 terrain{ t_level.terrainBuffer[cellIndex] + make_float2(0.0f, t_level.fluxBuffer[cellIndex]) };
	const float height{ terrain.x + terrain.y };
	const float avalancheAngle{ lerp(c_parameters.avalancheAngle, c_parameters.vegetationAngle, fmaxf(t_resistanceArray.read(cell).y, 0.0f)) };

	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i], t_level.gridSize) };
		nextCellIndices[i] = getCellIndex(nextCell, t_level.gridSize);

		const float2 nextTerrain{ t_level.terrainBuffer[nextCellIndices[i]] };
		const float nextFlux{ t_level.fluxBuffer[nextCellIndices[i]] };
		const float nextHeight{ nextTerrain.x + nextTerrain.y + nextFlux };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - avalancheAngle * c_distances[i] * t_level.gridScale, 0.0f);
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

__global__ void apply(const MultigridLevel t_level)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < t_level.cellCount; cellIndex += stride)
	{
		t_level.fluxBuffer[cellIndex] += t_level.avalancheBuffer[cellIndex];
		t_level.avalancheBuffer[cellIndex] = 0.0f;
	}
}

__global__ void restrict(const MultigridLevel t_level, const MultigridLevel t_nextLevel)
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

			for (int x{ 0 }; x <= 1; ++x)
			{
				for (int y{ 0 }; y <= 1; ++y)
				{
					const int cellIndex{ getCellIndex(cell + make_int2(x, y), t_level.gridSize) };
					const float2 terrain{ t_level.terrainBuffer[cellIndex] };

					nextTerrain += terrain;
				}
			}

			nextTerrain *= 0.25f;

			t_nextLevel.terrainBuffer[nextCellIndex] = nextTerrain;
			t_nextLevel.fluxBuffer[nextCellIndex] = 0.0f;
			t_nextLevel.avalancheBuffer[nextCellIndex] = 0.0f;
		}
	}
}

__global__ void prolongate(const MultigridLevel t_level, const MultigridLevel t_nextLevel)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 nextCell;

	for (nextCell.x = index.x; nextCell.x < t_nextLevel.gridSize.x; nextCell.x += stride.x)
	{
		for (nextCell.y = index.y; nextCell.y < t_nextLevel.gridSize.y; nextCell.y += stride.y)
		{
			const int nextCellIndex{ getCellIndex(nextCell, t_nextLevel.gridSize) };

			const int2 cell{ nextCell / 2 };
			const int cellIndex{ getCellIndex(cell, t_level.gridSize) };

			t_nextLevel.terrainBuffer[nextCellIndex] += make_float2(0.0f, t_nextLevel.fluxBuffer[nextCellIndex]);
			t_nextLevel.fluxBuffer[nextCellIndex] = t_level.fluxBuffer[cellIndex];
			t_nextLevel.avalancheBuffer[nextCellIndex] = 0.0f;
		}
	}
}

__global__ void finish(Array2D<float2> t_terrainArray, const MultigridLevel t_level)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			const int cellIndex{ getCellIndex(cell) };

			t_terrainArray.write(cell, t_level.terrainBuffer[cellIndex] + make_float2(0.0f, t_level.fluxBuffer[cellIndex]));
		}
	}
}

}

void multigrid(const LaunchParameters& t_launchParameters)
{
	float divisor{ 1.0f };
	float scale{ 1.0f };

	solver::setup<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.multigrid.front());
	
	for (int i{ 0 }; i < t_launchParameters.multigridVCycleIterations; ++i)
	{
		for (int j{ 0 }; j < t_launchParameters.multigridLevelCount - 1; ++j)
		{
			for (int k{ 0 }; k < t_launchParameters.multigridSolverIterations / divisor; ++k)
			{
				solver::smooth<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.multigrid[j], t_launchParameters.resistanceArray);
				solver::apply<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.multigrid[j]);
			}

			solver::restrict<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.multigrid[j], t_launchParameters.multigrid[j + 1]);
			divisor *= scale;
		}

		for (int j{ 0 }; j < t_launchParameters.multigridSolverIterations / divisor; ++j)
		{
			solver::smooth<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.multigrid.back(), t_launchParameters.resistanceArray);
			solver::apply<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.multigrid.back());
		}

		for (int j{ t_launchParameters.multigridLevelCount - 1 }; j > 0; --j)
		{
			solver::prolongate<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.multigrid[j], t_launchParameters.multigrid[j - 1]);
			divisor /= scale;

			for (int k{ 0 }; k < t_launchParameters.multigridSolverIterations / divisor; ++k)
			{
				solver::smooth<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.multigrid[j - 1], t_launchParameters.resistanceArray);
				solver::apply<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.multigrid[j - 1]);
			}
		}
	}

	solver::finish<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.multigrid.front());
}

}