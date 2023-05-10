#pragma once

#include <cuda_runtime.h>

namespace dunes
{

struct LaunchParameter
{
	dim3 gridSize2D;
	dim3 blockSize2D;
	unsigned int gridSize1D;
	unsigned int blockSize1D;
};

}
