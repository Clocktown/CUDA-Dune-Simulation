#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupStickyKernel(const Array2D<float2> t_terrainArray, const Array2D<float2> t_windArray, Buffer<float> t_cliffBuffer)
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

			const float2 windVelocity = t_windArray.read(cell);
			const float windSpeed = length(windVelocity);
			const float2 windDirection = windVelocity / (windSpeed + 1e-06f);

			float2 nextPosition{ make_float2(cell) - windDirection };
			int2 nextCell{ getWrappedCell(make_int2(nextPosition)) };
			const float2 nextTerrain{ t_terrainArray.sample(nextPosition) };
			const float nextHeight{ nextTerrain.x + nextTerrain.y };
			float cliffHeight{ height - nextHeight };
			const float angle{ cliffHeight / c_parameters.gridScale };

			const int cellIndex{ getCellIndex(cell) };

			if (angle >= c_parameters.stickyAngle)
			{
				t_cliffBuffer[cellIndex] = cliffHeight;
			}
			else
			{
				t_cliffBuffer[cellIndex] = 0.0f;
			}
		}
	}
}

__global__ void stickyKernel(const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray, Buffer<float> t_cliffBuffer)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const float2 windVelocity = t_windArray.read(cell);
	const float windSpeed = length(windVelocity);
	const float2 windDirection = windVelocity / (windSpeed + 1e-06f);

	float2 nextPosition{ make_float2(cell) };
	const float erosionResistance{ -c_parameters.stickyStrength };

	for (float distance = c_parameters.gridScale; distance <= c_parameters.maxStickyHeight; distance += c_parameters.gridScale)
	{
		nextPosition += windDirection;

		const int2 nextCell{ getWrappedCell(make_int2(nextPosition)) };
		const int nextCellIndex{ getCellIndex(nextCell) };
		const float cliffHeight{ t_cliffBuffer[nextCellIndex] };

		if (cliffHeight > 0.0f)
		{
			float4 resistance{ t_resistanceArray.read(nextCell) };
			const float maxDistance{ fminf(cliffHeight, c_parameters.maxStickyHeight) };
			const float erosionDistance{ c_parameters.stickyRange.x * maxDistance };
			const float stickyDistance{ c_parameters.stickyRange.y * maxDistance };
			
			if (distance <= erosionDistance)
			{
				resistance.w = erosionResistance;
				t_resistanceArray.write(cell, resistance);
			}
			else if (distance <= stickyDistance)
			{
				resistance.w = 1.0f - (distance - erosionDistance) / (stickyDistance - erosionDistance);
				t_resistanceArray.write(cell, resistance);
			}

			break;
		}
	}
}

void sticky(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	setupStickyKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.tmpBuffer);
	
	if (t_simulationParameters.stickyStrength > 0.0f)
	{
		stickyKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
	}
}

}
