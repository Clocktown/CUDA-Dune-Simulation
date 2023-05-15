#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

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

	float4 avalanches[2];
	float* angles{ &avalanches[0].x };
	float angleSum{ 0.0f };
	float maxAngle{ 0.0f };

	for (int i = 0; i < 8; ++i)
	{
		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		const float2 nextTerrain{ t_terrainArray.read(nextCell) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };

		const float heightDifference{ height - nextHeight };
		angles[i] = fmaxf(heightDifference * c_rDistances[i] - c_parameters.avalancheAngle * c_parameters.gridScale, 0.0f);
		angleSum += angles[i];
		maxAngle = fmaxf(maxAngle, angles[i]);
	}

	if (angleSum > 0.0f)
	{
		const float rAngleSum{ 1.0f / angleSum };
		const float avalancheSize{ fminf(c_parameters.avalancheStrength * maxAngle / (1.0f + (maxAngle * rAngleSum)), terrain.y) };

		for (int i = 0; i < 8; ++i)
		{
			angles[i] *= rAngleSum * avalancheSize;
		}
	}

	const int cellIndex{ getCellIndex(cell) };
	const int avalancheIndex{ 2 * cellIndex };
	t_avalancheBuffer[avalancheIndex] = avalanches[0];
	t_avalancheBuffer[avalancheIndex + 1] = avalanches[1];
}

__global__ void avalanchingKernel(Array2D<float2> t_terrainArray, Buffer<float4> t_avalancheBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	float2 terrain{ t_terrainArray.read(cell) };

	const int avalancheIndex{ 2 * cellIndex };
	const float4 avalanches[2]{ t_avalancheBuffer[avalancheIndex], t_avalancheBuffer[avalancheIndex + 1] };
	Buffer<float> avalancheBuffer{ reinterpret_cast<Buffer<float>>(t_avalancheBuffer) };
	const float* avalanche{ &avalanches[0].x };
	 
	for (int i = 0; i < 8; ++i)
	{
		terrain.y -= *avalanche++;

		const int2 nextCell{ getWrappedCell(cell + c_offsets[i]) };
		terrain.y += avalancheBuffer[8 * getCellIndex(nextCell) + (i + 4) % 8];
	}

	t_terrainArray.write(cell, terrain);
}

void avalanching(const LaunchParameters& t_launchParameters)
{
	Buffer<float4> avalancheBuffer{ reinterpret_cast<Buffer<float4>>(t_launchParameters.tmpBuffer) };

	for (int i = 0; i < t_launchParameters.avalancheIterations; ++i)
	{
		setupAvalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, avalancheBuffer);
		avalanchingKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, avalancheBuffer);
	}
}

}
