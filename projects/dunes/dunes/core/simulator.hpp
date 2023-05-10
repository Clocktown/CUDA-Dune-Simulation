#pragma once

#include "ui_parameter.hpp"
#include "simulation_parameter.cuh"
#include "launch_parameter.cuh"
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
	void awake();
	void update();
	void onGUI();
private:
	// Functionality
	void map();
	void unmap();

	// Attributes
	UIParameter m_uiParameter;
	SimulationParameter m_simulationParameter;
	LaunchParameter m_launchParameter;

	sthe::TerrainRenderer* m_terrainRenderer;
	std::shared_ptr<sthe::Terrain> m_terrain;
	std::shared_ptr<sthe::CustomMaterial> m_material;
	std::shared_ptr<sthe::gl::Program> m_program;
	std::shared_ptr<sthe::gl::Texture2D> m_heightMap;
	std::shared_ptr<sthe::gl::Texture2D> m_resistanceMap;

	sthe::cu::Array2D m_heightArray;
	sthe::cu::Array2D m_resistanceArray;
	sthe::cu::Array2D m_windArray;
	sthe::cu::Buffer m_slabBuffer;
	cudaTextureDesc m_textureDescriptor;
};

}
