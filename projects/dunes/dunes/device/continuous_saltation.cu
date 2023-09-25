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
			const float saltationScale{ (1.0f - resistance.x) * (1.0f - resistance.y) * (resistance.w > 0.0f ? 0.5f : 1.0f) };

			//const float scale{ windSpeed * c_parameters.deltaTime };

			const float saltation{ fminf(c_parameters.saltationStrength * saltationScale + (resistance.w < 0.0f ? -resistance.w : 0.0f), terrain.y) };

			terrain.y -= saltation;
			t_terrainArray.write(cell, terrain);

			const int cellIndex{ getCellIndex(cell) };
			const float slab{ saltation };

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

	const float2 windVelocity{ t_windArray.read(cell) };

	const float2 position{ make_float2(cell) };

	if (slab > 0.0f)
	{
		const float2 nextPosition{ position + windVelocity * c_parameters.rGridScale * c_parameters.deltaTime };
		//const int2 nextCell{ getNearestCell(nextPosition) };
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
					atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(nextCell)), weight * slab);
				}
			}
		}
		
		//atomicAdd(t_advectedSlabBuffer + getCellIndex(getWrappedCell(nextCell)), slab);
	}
}

__global__ void finishContinuousSaltationKernel(Array2D<float2> t_terrainArray, const Array2D<float2> t_windArray, const Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer, Buffer<float> t_advectedSlabBuffer)
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

			const float windSpeed{ length(t_windArray.read(cell)) };
			
			const float4 resistance{ t_resistanceArray.read(cell) };
			const float saltationScale{ (1.0f - resistance.x) * (1.0f - resistance.y) };
			const float abrasionScale{ saltationScale * (1.0f - resistance.z) };
			const float vegetationFactor = (terrain.y > 0.0f ? 0.4f : 0.6f);
			const float depositionProbability = fminf(fmaxf(fmaxf(resistance.x,
				(1.0f - vegetationFactor) + resistance.y * vegetationFactor), resistance.w), resistance.w < 0.f ? 0.f : 1.f);


			const float new_slab = slab * (1.f - depositionProbability);
			//if (new_slab > 0.0f) {
				const float scale{ slab + c_parameters.windOnlyAbrasionAmount }; // 0.0001 is the amount of abrasion that happens with purely wind, no sand 
				float abrasion{ terrain.y < c_parameters.abrasionThreshold ?
				c_parameters.abrasionStrength * abrasionScale * windSpeed * c_parameters.deltaTime *
				clamp(1.0f - terrain.y / c_parameters.abrasionThreshold, 0.f, 1.f) * scale : 0.0f };
				//abrasion = fminf(abrasion, terrain.x);

				terrain.y += abrasion;
				terrain.x -= abrasion;
			//}
			terrain.y += slab * depositionProbability;
			t_terrainArray.write(cell, terrain);
			t_slabBuffer[cellIndex] = slab * (1.f - depositionProbability); // write updated advectedSlabBuffer back to slabBuffer (ping-pong)
			t_advectedSlabBuffer[cellIndex] = slab * (1.f - resistance.y); // Used in Reptation as slabBuffer
		}
	}
}

void continuousSaltation(const LaunchParameters& t_launchParameters)
{
	setupContinuousSaltationKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
	continuousSaltationKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
	finishContinuousSaltationKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.slabBuffer, t_launchParameters.tmpBuffer);
}

}
