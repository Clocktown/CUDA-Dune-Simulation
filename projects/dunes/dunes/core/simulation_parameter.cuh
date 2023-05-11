#pragma once

#include <cuda_runtime.h>

namespace dunes
{

struct SimulationParameter
{
	int2 gridSize{ 2048, 2048 };
	int gridLength{ gridSize.x * gridSize.y };
	float gridScale{ 1.0f };

	float2 windDirection{ 1.0f, 0.0f };
	float windStrength{ 1.0f };
	float windCapacity{ 1.0f };

	float avalancheAngle{ 0.6494f }; // tan(33°)
	float vegetationAngle{ 1.0f }; // tan(45°)
	float minShadowAngle{ 0.1763f }; // tan(10°)
	float maxShadowAngle{ 0.2679f }; // tan(15°)

	float reptationStrength{ 1.0f };
	float deltaTime {1.0f};
};

void upload(const SimulationParameter& t_simulationParameter);

}
