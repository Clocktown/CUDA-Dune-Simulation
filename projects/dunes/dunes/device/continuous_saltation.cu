#include "continuous_saltation.cuh"
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

__global__ void setupContinuousSaltationKernel(Array2D<float2> t_terrainArray, const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer, Buffer<float> t_advectedSlabBuffer)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			float2 terrain{ t_terrainArray.read(cell) };

			const float2 windVelocity{ t_windArray.read(cell) };
			const float windSpeed{ length(windVelocity) };

			const float4 resistance{ t_resistanceArray.read(cell) };
			const float saltationResistance{ (1.0f - resistance.x) * (1.0f - resistance.y) };
			const float abrasionResistance{ saltationResistance * (1.0f - resistance.z) };

			const float scale{ windSpeed * c_parameters.deltaTime };
			const float abrasion{ fminf(terrain.y < c_parameters.abrasionThreshold ?
										c_parameters.abrasionStrength * abrasionResistance *
										(1.0f - terrain.y / c_parameters.abrasionThreshold) * scale : 0.0f, terrain.x) };

			const float saltation{ fminf(c_parameters.saltationStrength * saltationResistance * scale, terrain.y) };

			terrain.x -= abrasion;
			terrain.y -= saltation;
			t_terrainArray.write(cell, terrain);

			const int cellIndex{ getCellIndex(cell) };
			const float slab{ saltation + abrasion };

			t_slabBuffer[cellIndex] += slab;
			t_advectedSlabBuffer[cellIndex] = 0.0f;
		}
	}
}

__global__ void continuousSaltationKernel(const Array2D<float2> t_windArray, Buffer<float> t_slabBuffer, Buffer<float> t_advectedSlabBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const int cellIndex{ getCellIndex(cell) };
	const float slab{ t_slabBuffer[cellIndex] };

	const float2 windVelocity{t_windArray.read(cell) };

	const float2 position{ make_float2(cell) };

	if (slab > 0.0f)
	{
		const float2 nextPosition{ position + windVelocity * c_parameters.rGridScale * c_parameters.deltaTime };
		const int2 nextCell{ make_int2(nextPosition) };

		for (int x{ nextCell.x }; x <= nextCell.x + 1; ++x)
		{
			const float u{ 1.0f - abs(static_cast<float>(x) - nextPosition.x) };

			for (int y{ nextCell.y }; y <= nextCell.y + 1; ++y)
			{
				const float v{ 1.0f - abs(static_cast<float>(y) - nextPosition.y) };
				const float weight{ u * v };

				if (weight > 0.0f)
				{
					atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(int2{ x, y })), weight * slab);
				}
			}
		}
	}
}

__global__ void finishContinuousSaltationKernel(Array2D<float2> t_terrainArray, const Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer, const Buffer<float> t_advectedSlabBuffer)
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
			const float slab{ t_advectedSlabBuffer[cellIndex] };

			const float4 resistance{ t_resistanceArray.read(cell) };
			const float vegetationFactor = (terrain.y > 0.0f ? 0.4f : 0.6f);
			const float depositionProbability = fmaxf(resistance.x,
				(1.0f - vegetationFactor) + resistance.y * vegetationFactor);

			terrain.y += slab * depositionProbability;

			t_terrainArray.write(cell, terrain);
			t_slabBuffer[cellIndex] = slab * (1.f - depositionProbability);
			t_advectedSlabBuffer[cellIndex] = slab * (1.f - resistance.y);
		}
	}
}

void continuousSaltation(const LaunchParameters& t_launchParameters)
{
	setupContinuousSaltationKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
	continuousSaltationKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
	finishContinuousSaltationKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
}

}
