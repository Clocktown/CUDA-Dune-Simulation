#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void venturiKernel(Array2D<float2> t_terrainArray, Array2D<float2> t_windArray)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{

			const float2 terrain{ t_terrainArray.read(cell) };
			const float height{ terrain.x + terrain.y };

			const float venturiScale{ (1.0f + c_parameters.venturiStrength * height) };
			float2 windVelocity{ venturiScale * c_parameters.windSpeed * c_parameters.windDirection };
			windVelocity *= sinf(height);
			t_windArray.write(cell, windVelocity);
		}
	}
}

void venturi(const LaunchParameters& t_launchParameters)
{
	venturiKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray);
}

}
