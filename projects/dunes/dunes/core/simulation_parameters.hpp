#pragma once

#include <sthe/device/buffer.cuh>
#include <sthe/device/array2d.cuh>
#include <cuda_runtime.h>
#include <cufft.h>

#define TAN10 0.1763f
#define TAN15 0.2679f
#define TAN33 0.6494f
#define TAN45 1.0f
#define TAN55 1.4281f
#define TAN68 2.5f

namespace dunes
{

template<typename T>
using Array2D = sthe::device::Array2D<T>;

template<typename T>
using Buffer = sthe::device::Buffer<T>;

struct WindWarping
{
	int count{ 2 };
	float i_divisor{ 1.f / 20.0f };
	float radii[2]{ 200.0f, 50.0f };
	float strengths[2]{ 0.8f, 0.2f };
	float gradientStrengths[2]{ 30.f, 5.f };
	Buffer<cuComplex> gaussKernels[2];
	Buffer<cuComplex> smoothedHeights[2];
};

struct SimulationParameters
{
	int2 gridSize{ 2048, 2048 };
	float gridScale{ 1.0f };
	float rGridScale{ 1.0f / gridScale };
	int cellCount{ gridSize.x * gridSize.y };

	int2 windGridSize {1024, 1024};
    float windGridScale {2.f};
    float rWindGridScale {1.f / windGridScale};
    int   windCellCount {windGridSize.x * windGridSize.y};

	float2 windDirection{ 1.0f, 0.0f };
	float windSpeed{ 10.0f };

	float venturiStrength{ 0.005f };

	float windShadowDistance{ 1.0f };
	float minWindShadowAngle{ TAN10 };
	float maxWindShadowAngle{ TAN15 };

	float stickyStrength{ 1.0f };
	float stickyAngle{ TAN55 };
	float2 stickyRange{ 0.4f, 2.0f };
	float maxStickyHeight{ 30.0f };

	float abrasionStrength{ 0.0f };
	float abrasionThreshold{ 0.1f };
	float saltationStrength{ 0.05f };
	float reptationStrength{ 0.0f };
	float reptationSmoothingStrength{ 0.0f };
	float reptationUseWindShadow{ 0.f };

	float avalancheStrength{ 0.5f };
	float avalancheAngle{ TAN33 };
	float bedrockAngle{ TAN68 };
	float vegetationAngle{ TAN45 };

	float deltaTime{ 1.0f };
	int timestep = 0;
};

void upload(const SimulationParameters& t_simulationParameters);

}

#undef TAN10
#undef TAN15
#undef TAN33
#undef TAN45
#undef TAN55
#undef TAN68
