#pragma once

#include <glm/glm.hpp>
#include <array>
#include <vector>

namespace dunes
{

struct RenderParameters
{
	glm::vec4 sandColor{ 0.9f, 0.8f, 0.6f, 1.0f };
	glm::vec4 bedrockColor{ 0.5f, 0.5f, 0.5f, 1.0f };
	glm::vec4 windShadowColor{ 1.0f, 0.25f, 0.25f, 1.0f };
	glm::vec4 vegetationColor{ 0.25f, 0.75f, 0.25f, 1.0f };
	glm::vec4 erosionColor{ 0.0f, 0.0, 1.f, 1.0f };
	glm::vec4 stickyColor{ 1.0f, 1.0f, 0.0f, 1.0f };
	glm::vec4 objectColor{ 0.75f, 0.25f, 0.75f, 1.0f };
};

}
