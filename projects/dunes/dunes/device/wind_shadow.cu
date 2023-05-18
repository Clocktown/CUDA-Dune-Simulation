#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void windShadowKernel(const Array2D<float2> t_terrainArray, const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const float2 terrain{ t_terrainArray.read(cell) };
	const float2 windDirection{ normalize(t_windArray.read(cell)) };
	float4 resistance{ t_resistanceArray.read(cell) };

	const float height{ terrain.x + terrain.y };
	float2 nextPosition{ make_float2(cell) + 0.5f };
	float maxAngle{ 0.0f };

	for (float distance = c_parameters.gridScale; distance <= c_parameters.windShadowDistance; distance += c_parameters.gridScale)
	{
		nextPosition -= windDirection;
		
		const float2 nextTerrain{ t_terrainArray.sample(nextPosition) };
		const float nextHeight{ nextTerrain.x + nextTerrain.y };
		const float heightDifference{ nextHeight - height };
		const float angle{ heightDifference / distance };
	
		maxAngle = fmaxf(maxAngle, angle);
	}

	resistance.x = clamp((maxAngle - c_parameters.minWindShadowAngle) /
					     (c_parameters.maxWindShadowAngle - c_parameters.minWindShadowAngle), 0.0f, 1.0f);

	t_resistanceArray.write(cell, resistance);
}

void windShadow(const LaunchParameters& t_launchParameters)
{
	windShadowKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray);
}

}
