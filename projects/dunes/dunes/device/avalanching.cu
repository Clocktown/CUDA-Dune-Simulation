#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>
#include <stdio.h>

namespace dunes
{

__global__ void setupAvalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float4> t_avalancheBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	float2 terrain{ t_terrainArray.read(cell) };
	const float height{ terrain.x + terrain.y };

	float avalanches[8];
	float avalancheSum{ 0.0f };
	float maxAvalanche{ 0.0f };

	for (int i = 0; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		const float2 nextTerrain{ t_terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		avalanches[i] = fmaxf(heightDifference - c_parameters.avalancheAngle * c_distances[i] * c_parameters.gridScale, 0.0f);
		avalancheSum += avalanches[i];
		maxAvalanche = fmaxf(maxAvalanche, avalanches[i]);
	}

	if (avalancheSum > 0.0f)
	{
		const float rAvalancheSum{ 1.0f / avalancheSum };
		const float scale{ fminf(c_parameters.avalancheStrength * maxAvalanche /
								 (1.0f + maxAvalanche * rAvalancheSum), terrain.y) * rAvalancheSum };
		
		for (int i = 0; i < 8; ++i)
		{
			avalanches[i] *= scale;
		}
	}

	const int avalancheIndex{ 2 * getCellIndex(cell) };
	const float4* const avalancheGroups{ reinterpret_cast<float4*>(avalanches) };
	t_avalancheBuffer[avalancheIndex] = avalancheGroups[0];
	t_avalancheBuffer[avalancheIndex + 1] = avalancheGroups[1];
}

__global__ void avalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float4> t_avalancheBuffer)
{
	extern __shared__ float s_avalanches[];
	float4* const s_avalancheGroups{ reinterpret_cast<float4*>(s_avalanches) };

	const int2 globalOffset{ static_cast<int>(blockIdx.x * blockDim.x), static_cast<int>(blockIdx.y * blockDim.y) };
	const int2 blockSize{ getBlockSize2D() };

	const int2 localGridSize{ blockSize + 2 };
	const int2 localCell{ static_cast<int>(threadIdx.x) + 1, static_cast<int>(threadIdx.y) + 1 };
	
	for (int x{ localCell.x - 1 }; x < localGridSize.x; x += blockSize.x)
	{
		for (int y{ localCell.y - 1 }; y < localGridSize.y; y += blockSize.y)
		{
			const int2 cell{ getWrappedCell(int2{ x + globalOffset.x - 1, y + globalOffset.y - 1 }) };
			const int avalancheIndex{ 2 * getCellIndex(cell) };
			const int localAvalancheIndex{ 2 * getCellIndex(int2{ x, y }, localGridSize) };

			s_avalancheGroups[localAvalancheIndex] = t_avalancheBuffer[avalancheIndex];
			s_avalancheGroups[localAvalancheIndex + 1] = t_avalancheBuffer[avalancheIndex + 1];
		}
	}

	__syncthreads();

	const int2 cell{ localCell + globalOffset - 1 };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	float2 terrain{ t_terrainArray.read(cell) };

	const int localAvalancheIndex{ 8 * getCellIndex(localCell, localGridSize) };

	for (int i = 0; i < 8; ++i)
	{
		terrain.y -= s_avalanches[localAvalancheIndex + i];

		const int2 nextLocalCell{ localCell + c_offsets[i] };
		const int nextLocalAvalancheIndex{ 8 * getCellIndex(nextLocalCell, localGridSize) + (i + 4) % 8 };
		terrain.y += s_avalanches[nextLocalAvalancheIndex];
	}

	t_terrainArray.write(cell, terrain);
}

void avalanching(const LaunchParameters& t_launchParameters)
{
	const size_t sharedMemory{ static_cast<size_t>(8 * (t_launchParameters.blockSize2D.x + 2) * (t_launchParameters.blockSize2D.y + 2)) * sizeof(float) };
	Buffer<float4> avalancheBuffer{ reinterpret_cast<Buffer<float4>>(t_launchParameters.tmpBuffer) };

	for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	{
		setupAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, avalancheBuffer);
		avalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D, sharedMemory>>>(t_launchParameters.terrainArray, avalancheBuffer);
	}
}

}
