#pragma once

#include <cuda_runtime.h>

namespace dunes
{

struct SimulationParameters
{
	int2 gridSize{ 2048, 2048 };
	float gridScale{ 1.0f };
	float rGridScale{ 1.0f / gridScale };
	int cellCount{ gridSize.x * gridSize.y };

	float2 windDirection{ 1.0f, 0.0f };
	float windSpeed{ 20.0f };
	
	float windShadowDistance{ 10.0f };
	float minWindShadowAngle{ 0.1763f }; // tan(10°)
	float maxWindShadowAngle{ 0.2679f }; // tan(15°)

	float saltationSpeed{ 1.0f };
	float reptationStrength{ 1.0f };

	float avalancheStrength{ 1.0f };
	float avalancheAngle{ 0.6494f }; // tan(33°)
	float vegetationAngle{ 1.0f }; // tan(45°)

	float deltaTime{ 1.0f };
};

void upload(const SimulationParameters& t_simulationParameters);

}
