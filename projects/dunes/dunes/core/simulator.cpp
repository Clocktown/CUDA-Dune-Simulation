#include "simulator.hpp"
#include "simulation_parameters.hpp"
#include "launch_parameters.hpp"
#include <dunes/device/kernels.cuh>
#include <dunes/util/io.hpp>
#include <sthe/sthe.hpp>

#define DUNES_SIMULATOR_REUPLOAD_BIT 0
#define DUNES_SIMULATOR_RECREATE_BIT 1

namespace dunes
{

// Constructor
Simulator::Simulator() :
	m_terrain{ std::make_shared<sthe::Terrain>() },
	m_material{ std::make_shared<sthe::CustomMaterial>() },
	m_program{ std::make_shared<sthe::gl::Program>() },
	m_terrainMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_resistanceMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_textureDescriptor{},
	m_isPaused{ false }
{
	int device;
	int smCount;
	int smThreadCount;
	CU_CHECK_ERROR(cudaGetDevice(&device));
	CU_CHECK_ERROR(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
	CU_CHECK_ERROR(cudaDeviceGetAttribute(&smThreadCount, cudaDevAttrMaxThreadsPerMultiProcessor, device));
	const float threadCount{ static_cast<float>(smCount * smThreadCount) };
	
	m_launchParameters.blockSize1D = 256;
	m_launchParameters.blockSize2D = dim3{ 8, 8 };
	m_launchParameters.optimalGridSize1D = static_cast<unsigned int>(glm::ceil(threadCount / static_cast<float>(m_launchParameters.blockSize1D)));
	m_launchParameters.optimalGridSize2D.x = static_cast<unsigned int>(0.0625f * glm::ceil(glm::sqrt(threadCount / static_cast<float>(m_launchParameters.gridSize2D.x * m_launchParameters.gridSize2D.y))));
	m_launchParameters.optimalGridSize2D.y = m_launchParameters.optimalGridSize2D.x;

	m_terrain->setHeightMap(m_terrainMap);
	m_terrain->addLayer(std::make_shared<sthe::TerrainLayer>(glm::vec3(194.0f, 178.0f, 128.0f) / 255.0f));

	m_program->setPatchVertexCount(4);
	m_program->attachShader(sthe::gl::Shader{ GL_VERTEX_SHADER, sthe::getShaderPath() + "terrain/phong.vert" });
	m_program->attachShader(sthe::gl::Shader{ GL_TESS_CONTROL_SHADER, sthe::getShaderPath() + "terrain/phong.tesc" });
	m_program->attachShader(sthe::gl::Shader{ GL_TESS_EVALUATION_SHADER, getShaderPath() + "terrain/phong.tese" });
	m_program->attachShader(sthe::gl::Shader{ GL_FRAGMENT_SHADER, getShaderPath() + "terrain/phong.frag" });
	m_program->link();

	m_material->setProgram(m_program);
	m_material->setTexture(STHE_TEXTURE_UNIT_TERRAIN_CUSTOM0, m_resistanceMap);

	m_textureDescriptor.addressMode[0] = cudaAddressModeWrap;
	m_textureDescriptor.addressMode[1] = cudaAddressModeWrap;
	m_textureDescriptor.filterMode = cudaFilterModeLinear;
	m_textureDescriptor.normalizedCoords = 0;
}

// Functionality
void Simulator::awake()
{
	m_launchParameters.gridSize1D = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.cellCount) / static_cast<float>(m_launchParameters.blockSize1D)));
	m_launchParameters.gridSize2D.x = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.gridSize.x) / static_cast<float>(m_launchParameters.blockSize2D.x)));
	m_launchParameters.gridSize2D.y = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.gridSize.y) / static_cast<float>(m_launchParameters.blockSize2D.y)));
	
	m_terrainRenderer = &getGameObject().addComponent<sthe::TerrainRenderer>();
	m_terrainRenderer->setTerrain(m_terrain);
	m_terrainRenderer->setMaterial(m_material);

	m_terrain->setGridSize(glm::ivec2{ m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y });
	m_terrain->setGridScale(m_simulationParameters.gridScale);

	m_terrainMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RG32F, false);
	m_resistanceMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RGBA32F, false);

	m_windArray.reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, cudaCreateChannelDesc<float2>());
	m_launchParameters.windArray.surface = m_windArray.recreateSurface();
	m_launchParameters.windArray.texture = m_windArray.recreateTexture(m_textureDescriptor);

	m_terrainArray.reinitialize(*m_terrainMap);
	m_resistanceArray.reinitialize(*m_resistanceMap);

	m_slabBuffer.reinitialize(8 * m_simulationParameters.cellCount, sizeof(float));
	m_launchParameters.tmpBuffer = m_slabBuffer.getData<float>();

	map();

	initialization(m_launchParameters);

	unmap();
}

void Simulator::update()
{
	if (!m_isPaused)
	{
		map();

		venturi(m_launchParameters);
		windShadow(m_launchParameters);
		saltation(m_launchParameters);
		reptation(m_launchParameters);
		avalanching(m_launchParameters);

		unmap();
	}
}

void Simulator::resume()
{
	m_isPaused = false;
}

void Simulator::pause()
{
	m_isPaused = true;
}

void Simulator::map()
{
	if (m_hasChanged.test(DUNES_SIMULATOR_REUPLOAD_BIT))
	{
		upload(m_simulationParameters);
		m_hasChanged.reset(DUNES_SIMULATOR_REUPLOAD_BIT);
	}

	if (m_hasChanged.test(DUNES_SIMULATOR_RECREATE_BIT))
	{
		awake();
		m_hasChanged.reset(DUNES_SIMULATOR_RECREATE_BIT);
	}
	
	m_terrainArray.map();
	m_launchParameters.terrainArray.surface = m_terrainArray.recreateSurface();
	m_launchParameters.terrainArray.texture = m_terrainArray.recreateTexture(m_textureDescriptor);

	m_resistanceArray.map();
	m_launchParameters.resistanceArray.surface = m_resistanceArray.recreateSurface();
	m_launchParameters.resistanceArray.texture = m_resistanceArray.recreateTexture(m_textureDescriptor);
}

void Simulator::unmap()
{
	m_terrainArray.unmap();
	m_resistanceArray.unmap();
}

// Setters
void Simulator::setGridSize(const glm::ivec2& t_gridSize)
{
	m_simulationParameters.gridSize = int2{ t_gridSize.x, t_gridSize.y };
	m_simulationParameters.cellCount = t_gridSize.x * t_gridSize.y;
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
	m_hasChanged.set(DUNES_SIMULATOR_RECREATE_BIT);
}

void Simulator::setGridScale(const float t_gridScale)
{
	STHE_ASSERT(t_gridScale != 0.0f, "Grid scale cannot be 0");

	m_simulationParameters.gridScale = t_gridScale;
	m_simulationParameters.rGridScale = 1.0f / t_gridScale;
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setWindAngle(const float t_windAngle)
{
	const float windAngle{ glm::radians(t_windAngle) };
	m_simulationParameters.windDirection = float2{ glm::cos(windAngle), glm::sin(windAngle) };
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setWindSpeed(const float t_windSpeed)
{
	m_simulationParameters.windSpeed = t_windSpeed;
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setWindShadowDistance(const float t_windShadowDistance)
{
	m_simulationParameters.windShadowDistance = t_windShadowDistance;
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setMinWindShadowAngle(const float t_minWindShadowAngle)
{
	m_simulationParameters.minWindShadowAngle = glm::tan(glm::radians(t_minWindShadowAngle));
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setMaxWindShadowAngle(const float t_maxWindShadowAngle)
{
	m_simulationParameters.maxWindShadowAngle = glm::tan(glm::radians(t_maxWindShadowAngle));
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setSaltationSpeed(const float t_saltationSpeed)
{
	m_simulationParameters.saltationSpeed = t_saltationSpeed;
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setReptationStrength(const float t_reptationStrength)
{
	m_simulationParameters.reptationStrength = t_reptationStrength;
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setAvalancheIterations(const int t_avalancheIterations)
{
	m_launchParameters.avalancheIterations = t_avalancheIterations;
}

void Simulator::setAvalancheAngle(const float t_avalancheAngle)
{
	m_simulationParameters.avalancheAngle = glm::tan(glm::radians(t_avalancheAngle));
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setVegetationAngle(const float t_vegetationAngle)
{
	m_simulationParameters.vegetationAngle = glm::tan(glm::radians(t_vegetationAngle));
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

void Simulator::setDeltaTime(const float t_deltaTime)
{
	m_simulationParameters.deltaTime = t_deltaTime;
	m_hasChanged.set(DUNES_SIMULATOR_REUPLOAD_BIT);
}

}
