#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

//__device__ __forceinline__ void atomicMax(const float* const addr, const float val)
//{
//	if (*addr >= val) return;
//
//	unsigned int* const uaddr = (unsigned int*)addr;
//	unsigned int old = *uaddr, assumed;
//
//	do
//	{
//		assumed = old;
//
//		if (__uint_as_float(assumed) >= val)
//		{
//			break;
//		}
//
//		old = atomicCAS(uaddr, assumed, __float_as_uint(val));
//	}
//	while (assumed != old);
//}

__global__ void setupStickyKernel(Array2D<float4> t_resistanceArray)
{
	const int2 index{ getGlobalIndex2D() };
	const int2 stride{ getGridStride2D() };

	int2 cell;

	for (cell.x = index.x; cell.x < c_parameters.gridSize.x; cell.x += stride.x)
	{
		for (cell.y = index.y; cell.y < c_parameters.gridSize.y; cell.y += stride.y)
		{
			float4 resistance{ t_resistanceArray.read(cell) };
			resistance.w = 0.0f;

			t_resistanceArray.write(cell, resistance);
		}
	}
}

__global__ void stickyKernel(const Array2D<float2> t_terrainArray, const Array2D<float2> t_windArray, Array2D<float4> t_resistanceArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

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

	if (angle >= c_parameters.stickyAngle)
	{
		const float maxDistance = fminf(cliffHeight, c_parameters.maxStickyHeight);
		const float erosionResistance{ -c_parameters.stickyStrength };

		int cellCount{ static_cast<int>(c_parameters.stickyRange.x * maxDistance / c_parameters.gridScale) };

		for (int i{ 0 }; i < cellCount; ++i)
		{
			float4 resistance{ t_resistanceArray.read(nextCell) };
			resistance.w = erosionResistance;
			t_resistanceArray.write(nextCell, resistance);

			nextPosition -= windDirection;
			nextCell = getWrappedCell(make_int2(nextPosition));
		}

		cellCount = static_cast<int>((c_parameters.stickyRange.y - c_parameters.stickyRange.x) * maxDistance / c_parameters.gridScale);
		const float step{ 1.0f / (c_parameters.stickyStrength * cellCount) };
		
		for (int i{ 0 }; i < cellCount; ++i)
		{
			float4 resistance{ t_resistanceArray.read(nextCell) };
			resistance.w = (1.0f - i * step);
			t_resistanceArray.write(nextCell, resistance);

			nextPosition -= windDirection;
			nextCell = getWrappedCell(make_int2(nextPosition));
		}
	}
}

void sticky(const LaunchParameters& t_launchParameters, const SimulationParameters& t_simulationParameters)
{
	setupStickyKernel<<<t_launchParameters.optimalGridSize2D, t_launchParameters.optimalBlockSize2D>>>(t_launchParameters.resistanceArray);
	
	if (t_simulationParameters.stickyStrength > 0.0f)
	{
		stickyKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray);
	}
}

}
