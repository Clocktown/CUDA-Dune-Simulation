#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void setupSaltationKernel(Buffer<float> t_slabBuffer)
{
	const int cellIndex{ getGlobalIndex1D() };

	if (isOutside(cellIndex))
	{
		return;
	}

	t_slabBuffer[cellIndex] = 0.0f;
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
	const float4 resistance{ t_resistanceArray.read(cell) };

	const float2 nextPosition{ make_float2(cell) + windVelocity * c_parameters.rGridScale * c_parameters.deltaTime };
	const int2 nextCell{ getWrappedCell(getNearestCell(nextPosition)) };

	const float saltationResistance{ (1.0f - resistance.x) * (1.0f - resistance.y) };
	const float slab{ fminf(saltationResistance * c_parameters.saltationSpeed * c_parameters.rGridScale * c_parameters.rGridScale * c_parameters.deltaTime, terrain.y) };
	terrain.y -= slab;

	t_terrainArray.write(cell, terrain);
	atomicAdd(t_slabBuffer + getCellIndex(nextCell), slab);
}

void saltation(const LaunchParameters& t_launchParameters)
{
	setupSaltationKernel<<<t_launchParameters.gridSize1D, t_launchParameters.blockSize1D>>>(t_launchParameters.tmpBuffer);
	saltationKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.terrainArray, t_launchParameters.windArray, t_launchParameters.resistanceArray, t_launchParameters.tmpBuffer);
}

}
