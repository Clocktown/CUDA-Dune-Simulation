#pragma once

#include "launch_parameter.cuh"
#include <cuda_runtime.h>

namespace dunes
{

void initialize(const LaunchParameter& t_launchParameter, const cudaSurfaceObject_t t_heightMap);

}
