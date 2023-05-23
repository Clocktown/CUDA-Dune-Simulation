#pragma once

#include "simulation_parameters.hpp"
#include "launch_parameters.hpp"
#include <sthe/sthe.hpp>

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
	void reinitialize(const glm::ivec2& t_gridSize, const float t_gridScale);
	void awake();
	void update();
	void resume();
	void pause();

	// Setters
	void setWindAngle(const float t_windAngle);
	void setWindSpeed(const float t_windSpeed);
	void setVenturiStrength(const float t_venturiStrength);
	void setWindShadowDistance(const float t_windShadowDistance);
	void setMinWindShadowAngle(const float t_minWindShadowAngle);
	void setMaxWindShadowAngle(const float t_maxWindShadowAngle);
	void setSaltationStrength(const float t_saltationStrength);
	void setReptationStrength(const float t_reptationStrength);
	void setAvalancheMode(const AvalancheMode t_avalancheMode);
	void setAvalancheIterations(const int t_avalancheIterations);
	void setAvalancheFinalSoftIterations(const int t_avalancheFinalSoftIterations);
	void setAvalancheSoftIterationModulus(const int t_avalancheSoftIterationModulus);
	void setAvalancheStrength(const float t_avalancheStrength);
	void setAvalancheAngle(const float t_avalancheAngle);
	void setVegetationAngle(const float t_vegetationAngle);
	void setTimeMode(const TimeMode t_timeMode);
	void setTimeScale(const float t_timeScale);
	void setFixedDeltaTime(const float t_fixedDeltaTime);

	// Getters
	bool isPaused() const;
private:
	// Functionality
	void map();
	void unmap();

	// Attributes
	SimulationParameters m_simulationParameters;
	LaunchParameters m_launchParameters;
	float m_timeScale;
	float m_fixedDeltaTime;
	
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

	bool m_isAwake;
	bool m_isPaused;
};

}
