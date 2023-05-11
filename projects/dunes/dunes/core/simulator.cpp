#include "simulator.hpp"
#include "ui_parameter.hpp"
#include "simulation_parameter.cuh"
#include "launch_parameter.cuh"
#include "constant.cuh"
#include "simulation.cuh"
#include <dunes/util/io.hpp>
#include <sthe/sthe.hpp>

namespace dunes
{

// Constructor
Simulator::Simulator() :
	m_terrain{ std::make_shared<sthe::Terrain>()},
	m_material{ std::make_shared<sthe::CustomMaterial>() },
	m_program{ std::make_shared<sthe::gl::Program>() },
	m_heightMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_resistanceMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_textureDescriptor{}
{
	m_textureDescriptor.addressMode[0] = cudaAddressModeWrap;
	m_textureDescriptor.addressMode[1] = cudaAddressModeWrap;
	m_textureDescriptor.filterMode = cudaFilterModeLinear;
}

// Functionality
void Simulator::awake()
{
	m_launchParameter.blockSize2D = dim3{ 16, 16 };
	m_launchParameter.blockSize1D = 256;
	m_launchParameter.gridSize2D.x = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameter.gridSize.x) / static_cast<float>(m_launchParameter.blockSize2D.x)));
	m_launchParameter.gridSize2D.y = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameter.gridSize.y) / static_cast<float>(m_launchParameter.blockSize2D.y)));
	m_launchParameter.gridSize1D = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameter.gridLength) / static_cast<float>(m_launchParameter.blockSize1D)));

	m_terrainRenderer = &getGameObject().addComponent<sthe::TerrainRenderer>();
	m_terrainRenderer->setTerrain(m_terrain);
	m_terrainRenderer->setMaterial(m_material);

	m_terrain->setGridSize(glm::ivec2 {m_simulationParameter.gridSize.x,
                                           m_simulationParameter.gridSize.y});
	m_terrain->setGridScale(m_simulationParameter.gridScale);
	m_terrain->setHeightMap(m_heightMap);
	m_terrain->addLayer(std::make_shared<sthe::TerrainLayer>(glm::vec3(194.0f, 178.0f, 128.0f) / 255.0f));

	m_material->setProgram(m_program);
	m_material->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0, m_resistanceMap);

	m_program->setPatchVertexCount(4);
	m_program->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, sthe::getShaderPath() + "terrain/phong.vert" });
	m_program->attachShader(sthe::gl::Shader{ GL_TESS_CONTROL_SHADER, sthe::getShaderPath() + "terrain/phong.tesc" });
	m_program->attachShader(sthe::gl::Shader{ GL_TESS_EVALUATION_SHADER, getShaderPath() + "terrain/phong.tese" });
	m_program->attachShader(sthe::gl::Shader{ GL_GEOMETRY_SHADER, sthe::getShaderPath() + "terrain/phong.geom" });
	m_program->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "terrain/phong.frag" });
	m_program->link();

	m_heightMap->reinitialize(m_simulationParameter.gridSize.x, m_simulationParameter.gridSize.y, GL_RG32F, false);
	m_resistanceMap->reinitialize(m_simulationParameter.gridSize.x, m_simulationParameter.gridSize.y, GL_RGBA32F, false);

	m_heightArray.reinitialize(*m_heightMap);
	m_resistanceArray.reinitialize(*m_resistanceMap);

	m_windArray.reinitialize(m_simulationParameter.gridSize.x, m_simulationParameter.gridSize.y, cudaCreateChannelDesc<float2>());
	m_windArray.recreateSurface();
	m_windArray.recreateTexture(m_textureDescriptor);
	m_slabBuffer.reinitialize(m_simulationParameter.gridLength, sizeof(float));

	map();

	initialize(m_launchParameter, m_heightArray.getSurface());

	unmap();
}

void Simulator::update()
{
	if (!m_uiParameter.isPaused)
	{
		map();



		unmap();
	}
}

void Simulator::map()
{
	GL_CHECK_ERROR(glFinish());
	
	upload(m_simulationParameter);

	m_heightArray.map();
	m_heightArray.recreateSurface();
	m_heightArray.recreateTexture(m_textureDescriptor);

	m_resistanceArray.map();
	m_resistanceArray.recreateSurface();
	m_resistanceArray.recreateTexture(m_textureDescriptor);
}

void Simulator::unmap()
{
	CU_CHECK_ERROR(cudaDeviceSynchronize());

	m_heightArray.unmap();
	m_resistanceArray.unmap();
}

}
