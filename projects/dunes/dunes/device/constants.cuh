#pragma once

#include <dunes/core/simulation_parameters.hpp>
#include <cuda_runtime.h>

namespace dunes
{

extern __constant__ SimulationParameters c_parameters;
extern __constant__ int2 c_offsets[8];
extern __constant__ float c_distances[8];
extern __constant__ float c_rDistances[8];

}
