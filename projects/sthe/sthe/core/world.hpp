#pragma once

#include <glm/glm.hpp>

namespace sthe
{

enum class Space : unsigned char
{
    Local,
    World
};

namespace World
{
    inline constexpr glm::vec3 x{ 1.0f, 0.0f, 0.0f };
    inline constexpr glm::vec3 y{ 0.0f, 1.0f, 0.0f };
    inline constexpr glm::vec3 z{ 0.0f, 0.0f, 1.0f };
    inline constexpr glm::vec3 right{ 1.0f, 0.0f, 0.0f };
    inline constexpr glm::vec3 up{ 0.0f, 1.0f, 0.0f };
    inline constexpr glm::vec3 forward{ 0.0f, 0.0f, -1.0f };
}

}
