#pragma once

#include "simulation_parameters.hpp"
#include "launch_parameters.hpp"
#include <sthe/sthe.hpp>
#include <bitset>

namespace dunes
{

class Simulator : public sthe::Component
{
public:
	// Constructors
	explicit Simulator();
	Simulator(const Simulator& t_simulator) = default;
	Simulator(Simulator&& t_simulator) = default;

	// Destructor
	~Simulator() = default;

	// Operators
	Simulator& operator=(const Simulator& t_simulator) = default;
	Simulator& operator=(Simulator&& t_simulator) = default;

	// Functionality
	void awake();
	void update();
	void resume();
	void pause();

	// Setters
	void setGridSize(const glm::ivec2& t_gridSize);
	void setGridScale(const float t_gridScale);
	void setWindAngle(const float t_windAngle);
	void setWindSpeed(const float t_windSpeed);
	void setWindShadowDistance(const float t_windShadowDistance);
	void setMinWindShadowAngle(const float t_minWindShadowAngle);
	void setMaxWindShadowAngle(const float t_maxWindShadowAngle);
	void setSaltationSpeed(const float t_saltationSpeed);
	void setReptationStrength(const float t_reptationStrength);
	void setAvalancheIterations(const int t_avalancheIterations);
	void setAvalancheAngle(const float t_avalancheAngle);
	void setVegetationAngle(const float t_vegetationAngle);
	void setDeltaTime(const float t_deltaTime);
private:
	// Functionality
	void map();
	void unmap();

	// Attributes
	SimulationParameters m_simulationParameters;
	LaunchParameters m_launchParameters;
	
	sthe::TerrainRenderer* m_terrainRenderer;
	std::shared_ptr<sthe::Terrain> m_terrain;
	std::shared_ptr<sthe::CustomMaterial> m_material;
	std::shared_ptr<sthe::gl::Program> m_program;
	std::shared_ptr<sthe::gl::Texture2D> m_terrainMap;
	std::shared_ptr<sthe::gl::Texture2D> m_resistanceMap;

	sthe::cu::Array2D m_terrainArray;
	sthe::cu::Array2D m_windArray;
	sthe::cu::Array2D m_resistanceArray;
	sthe::cu::Buffer m_slabBuffer;
	cudaTextureDesc m_textureDescriptor;

	std::bitset<2> m_hasChanged;
	bool m_isPaused;
};

}
