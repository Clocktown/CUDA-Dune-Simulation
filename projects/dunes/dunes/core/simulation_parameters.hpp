#pragma once

#include <sthe/device/buffer.cuh>
#include <sthe/device/array2d.cuh>
#include <cuda_runtime.h>
#include <cufft.h>

#define TAN10 0.1763f
#define TAN15 0.2679f
#define TAN33 0.6494f
#define TAN45 1.0f

namespace dunes
{

template<typename T>
using Array2D = sthe::device::Array2D<T>;

template<typename T>
using Buffer = sthe::device::Buffer<T>;

struct WindWarping
{
	int count{ 2 };
	float divisor{ 20.0f };
	float radii[4]{ 200.0f, 50.0f, 0.0f, 0.0f };
	float strengths[4]{ 0.8f, 0.2f, 0.0f, 0.0f };
	Buffer<cuComplex> gaussKernels[4];
	Buffer<cuComplex> smoothedHeights[4];
};

struct MultigridLevel
{
	int2 gridSize;
	float gridScale;
	int cellCount;
	Buffer<float2> terrainBuffer;
	Buffer<float> fluxBuffer;
	Buffer<float> avalancheBuffer;
};

struct SimulationParameters
{
	int2 gridSize{ 2048, 2048 };
	float gridScale{ 1.0f };
	float rGridScale{ 1.0f / gridScale };
	int cellCount{ gridSize.x * gridSize.y };

	float2 windDirection{ 1.0f, 0.0f };
	float windSpeed{ 10.0f };

	float venturiStrength{ 0.005f };

	float windShadowDistance{ 1.0f };
	float minWindShadowAngle{ TAN10 };
	float maxWindShadowAngle{ TAN15 };

	float abrasionStrength{ 0.0f };
	float abrasionThreshold{ 0.1f };
	float saltationStrength{ 0.05f };
	float reptationStrength{ 0.0f };

	float avalancheStrength{ 0.5f };
	float avalancheAngle{ TAN33 };
	float vegetationAngle{ TAN45 };

	float deltaTime{ 1.0f };
};

void upload(const SimulationParameters& t_simulationParameters);

}

#undef TAN10
#undef TAN15
#undef TAN33
#undef TAN45
