#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "multigrid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupBedrockAvalancheKernel(Array2D<float2> t_terrainArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	t_terrainBuffer[cellIndex] = t_terrainArray.read(cell);
}

template<BedrockAvalancheMode mode>
__global__ void bedrockAvalancheKernel(const Array2D<float4> t_resistanceArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	const float2 terrain{ t_terrainBuffer[cellIndex] };
	const float height{ terrain.x };
	
	int nextCellIndices[8];
	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i{ 0 }; i < 8; ++i)
	{
		nextCellIndices[i] = getCellIndex(getWrappedCell(cell + c_offsets[i]));
		const float2 nextTerrain{ t_terrainBuffer[nextCellIndices[i]] };
		const float nextHeight{ nextTerrain.x };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - c_parameters.bedrockAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float avalancheSize{ c_parameters.avalancheStrength * maxAvalanche /
								   (1.0f + maxAvalanche * rAvalancheSum) };


		const float scale{ avalancheSize * rAvalancheSum };

		for (int i{ 0 }; i < 8; ++i)
		{
			if (avalanches[i] > 0.0f)
			{
				if constexpr (mode == BedrockAvalancheMode::ToSand)
				{
					atomicAdd(&t_terrainBuffer[nextCellIndices[i]].y, scale * avalanches[i]);
				}
				else
				{
					atomicAdd(&t_terrainBuffer[nextCellIndices[i]].x, scale * avalanches[i]);
				}
			}
		}

		atomicAdd(&t_terrainBuffer[cellIndex].x, -avalancheSize);
	}
}

__global__ void finishBedrockAvalancheKernel(Array2D<float2> t_terrainArray, Buffer<float2> t_terrainBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };

	t_terrainArray.write(cell, t_terrainBuffer[cellIndex]);
}

void bedrockAvalanching(const LaunchParameters& t_launchParameters)
{
	if (t_launchParameters.bedrockAvalancheIterations <= 0)
	{
		return;
	}

	Buffer<float2> terrainBuffer{ reinterpret_cast<Buffer<float2>>(t_launchParameters.tmpBuffer) };
	setupBedrockAvalancheKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);

	if (t_launchParameters.bedrockAvalancheMode == BedrockAvalancheMode::ToSand)
	{
		for (int i = 0; i < t_launchParameters.bedrockAvalancheIterations; ++i)
		{
			bedrockAvalancheKernel<BedrockAvalancheMode::ToSand><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.resistanceArray, terrainBuffer);
		}
	}
	else
	{
		for (int i = 0; i < t_launchParameters.bedrockAvalancheIterations; ++i)
		{
			bedrockAvalancheKernel<BedrockAvalancheMode::ToBedrock><<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.resistanceArray, terrainBuffer);
		}
	}

	finishBedrockAvalancheKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, terrainBuffer);
}

}
