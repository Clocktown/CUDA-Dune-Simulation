#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include "continuous_saltation.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupSaltationKernel(Buffer<float> t_slabBuffer)
{
	const int stride{ getGridStride1D() };

	for (int cellIndex{ getGlobalIndex1D() }; cellIndex < c_parameters.cellCount; cellIndex += stride)
	{
		t_slabBuffer[cellIndex] = 0.0f;
	}
}

__global__ void saltationKernel(Array2D<float2> t_terrainArray, const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray, Buffer<float> t_slabBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	float2 terrain{ t_terrainArray.read(cell) };

	const float2 windVelocity{ t_windArray.read(cell) };
	const float windSpeed{ length(windVelocity) };

	const float4 resistance{ t_resistanceArray.read(cell) };
	const float saltationScale{ (1.0f - resistance.x) * (1.0f - resistance.y) * (1.0f - resistance.w) };
	const float abrasionScale{ (1.0f - resistance.x) * (1.0f - resistance.y) * (1.0f - resistance.z) };
	
	const float scale{ windSpeed * c_parameters.deltaTime };
	const float abrasion{ fminf(terrain.y < c_parameters.abrasionThreshold ? 
								c_parameters.abrasionStrength * abrasionScale *  
								(1.0f - terrain.y / c_parameters.abrasionThreshold) * scale : 0.0f, terrain.x) };

	const float saltation{ fminf(c_parameters.saltationStrength * saltationScale, terrain.y) };
	const float slab{ saltation + abrasion };

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
					atomicAdd(t_slabBuffer + getCellIndex(getWrappedCell(int2{ x, y })), weight * slab);
				}
			}
		}

		terrain.x -= abrasion;
		terrain.y -= saltation;
		t_terrainArray.write(cell, terrain);
	}
}

__global__ void finishSaltationKernel(Array2D<float2> t_terrainArray, Buffer<float> t_slabBuffer)
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
			terrain.y += t_slabBuffer[getCellIndex(cell)];

			t_terrainArray.write(cell, terrain);
		}
	}
}

void saltation(const LaunchParameters& t_launchParameters)
{
	switch (t_launchParameters.saltationMode)
	{
	case SaltationMode::PerFrame:
	    setupSaltationKernel<<<t_launchParameters.optimalGridSize1D, t_launchParameters.optimalBlockSize1D>>>(t_launchParameters.tmpBuffer);
	    saltationKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
	
	    // Now handled by reptation
	    //finishSaltationKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.tmpBuffer);

		break;
	case SaltationMode::Continuous:
	    continuousSaltation(t_launchParameters); 
	    break;
	}
}

}
