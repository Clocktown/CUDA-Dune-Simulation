#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace dunes
{
namespace device
{

struct Array
{
	cudaTextureObject_t texture;
	cudaSurfaceObject_t surface;
};

struct Simulation
{
	int2 gridSize;
	float gridScale;
	float heightScale;

	float2 windDirection;
	float windStrength;
	float windCapacity;

	float avalancheAngle;
	float vegetationAngle;
	float minShadowAngle;
	float maxShadowAngle;
	 
	Array terrain;
	Array wind;
	Array resistance;

	float* slabs;
	curandState* curandStates;
};

}
}
