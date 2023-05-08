#include "io.hpp"
#include <filesystem>
#include <string>

namespace sthe
{

inline std::string getModelPath()
{
	return getResourcePath() + "models" + std::string{ std::filesystem::path::preferred_separator };
}

inline std::string getMeshPath()
{
	return getResourcePath() + "meshes" + std::string{ std::filesystem::path::preferred_separator };
}

inline std::string getShaderPath()
{
	return getResourcePath() + "shaders" + std::string{ std::filesystem::path::preferred_separator };
}

inline std::string getResourcePath()
{
	const std::filesystem::path path{ __FILE__ };
	const std::string separator{ std::filesystem::path::preferred_separator };

	return path.parent_path().parent_path().string() + separator + "resources" + separator;
}

}