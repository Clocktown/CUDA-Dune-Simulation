#include "kernels.cuh"
#include "constants.cuh"
#include "grid.cuh"
#include <dunes/core/simulation_parameters.hpp>
#include <dunes/core/launch_parameters.hpp>
#include <sthe/device/vector_extension.cuh>

namespace dunes
{

__global__ void venturiKernel(Array2D<float2> t_windArray)
{
	const int2 cell{ getGlobalIndex2D() };

	if (isOutside(cell))
	{
		return;
	}

	const float2 windVelocity{ c_parameters.windSpeed * c_parameters.windDirection };
	t_windArray.write(cell, windVelocity);
}

void venturi(const LaunchParameters& t_launchParameters)
{
	venturiKernel<<<t_launchParameters.gridSize2D, t_launchParameters.blockSize2D>>>(t_launchParameters.windArray);
}

}
