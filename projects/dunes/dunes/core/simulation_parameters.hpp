#pragma once

#include <cuda_runtime.h>

#define TAN10 0.1763f
#define TAN15 0.2679f
#define TAN33 0.6494f
#define TAN45 1.0f

namespace dunes
{

struct SimulationParameters
{
	int2 gridSize{ 2048, 2048 };
	float gridScale{ 1.0f };
	float rGridScale{ 1.0f / gridScale };
	int cellCount{ gridSize.x * gridSize.y };

	float2 windDirection{ 1.0f, 0.0f };
	float windSpeed{ 10.0f };

	float venturiStrength{ 0.005f };
	
	float windShadowDistance{ 10.0f };
	float minWindShadowAngle{ TAN10 };
	float maxWindShadowAngle{ TAN15 };

	float saltationSpeed{ 100.0f };
	float reptationStrength{ 1.0f };

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