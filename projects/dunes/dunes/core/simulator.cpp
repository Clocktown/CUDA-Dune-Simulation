#include "simulator.hpp"
#include "simulation_parameters.hpp"
#include "launch_parameters.hpp"
#include <dunes/device/kernels.cuh>
#include <dunes/util/io.hpp>
#include <sthe/sthe.hpp>
#include <vector>

namespace dunes
{

// Constructor
Simulator::Simulator() :
	m_timeScale{ 1.0f },
	m_fixedDeltaTime{ 0.02f },
	m_terrain{ std::make_shared<sthe::Terrain>() },
	m_material{ std::make_shared<sthe::CustomMaterial>() },
	m_program{ std::make_shared<sthe::gl::Program>() },
	m_terrainMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_resistanceMap{ std::make_shared<sthe::gl::Texture2D>() },
	m_textureDescriptor{},
	m_isAwake{ false },
	m_isPaused{ false }
{
	int device;
	int smCount;
	int smThreadCount;
	CU_CHECK_ERROR(cudaGetDevice(&device));
	CU_CHECK_ERROR(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
	CU_CHECK_ERROR(cudaDeviceGetAttribute(&smThreadCount, cudaDevAttrMaxThreadsPerMultiProcessor, device));
	const float threadCount{ static_cast<float>(smCount * smThreadCount) };
	
	m_launchParameters.blockSize1D = 512;
	m_launchParameters.blockSize2D = dim3{ 8, 8 };

	m_launchParameters.optimalBlockSize1D = 256;
	m_launchParameters.optimalBlockSize2D = dim3{ 16, 16 };
	m_launchParameters.optimalGridSize1D = static_cast<unsigned int>(threadCount / static_cast<float>(m_launchParameters.blockSize1D));
	m_launchParameters.optimalGridSize2D.x = 2 * 5 * static_cast<unsigned int>(glm::sqrt(threadCount / static_cast<float>(m_launchParameters.optimalBlockSize2D.x * m_launchParameters.optimalBlockSize2D.y)));
	m_launchParameters.optimalGridSize2D.y = m_launchParameters.optimalGridSize2D.x;
	m_launchParameters.optimalGridSize2D.z = 1;

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
void Simulator::reinitialize(const glm::ivec2& t_gridSize, const float t_gridScale)
{
	STHE_ASSERT(t_gridScale != 0.0f, "Grid scale cannot be 0");

	m_simulationParameters.gridSize.x = t_gridSize.x;
	m_simulationParameters.gridSize.y = t_gridSize.y;
	m_simulationParameters.cellCount = t_gridSize.x * t_gridSize.y;
	m_simulationParameters.gridScale = t_gridScale;
	m_simulationParameters.rGridScale = 1.0f / t_gridScale;

	if (m_isAwake)
	{
		awake();
	}
}

void Simulator::awake()
{
	setupLaunchParameters();
	setupTerrain();
	setupArrays();
	setupBuffers();
	setupMultigrid();

	map();

	initialization(m_launchParameters);

	unmap();

	m_isAwake = true;
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

void Simulator::setupLaunchParameters()
{
	m_launchParameters.gridSize1D = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.cellCount) / static_cast<float>(m_launchParameters.blockSize1D)));
	m_launchParameters.gridSize2D.x = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.gridSize.x) / static_cast<float>(m_launchParameters.blockSize2D.x)));
	m_launchParameters.gridSize2D.y = static_cast<unsigned int>(glm::ceil(static_cast<float>(m_simulationParameters.gridSize.y) / static_cast<float>(m_launchParameters.blockSize2D.y)));
}

void Simulator::setupTerrain()
{
	m_terrainRenderer = &getGameObject().addComponent<sthe::TerrainRenderer>();
	m_terrainRenderer->setTerrain(m_terrain);
	m_terrainRenderer->setMaterial(m_material);

	m_terrain->setGridSize(glm::ivec2{ m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y });
	m_terrain->setGridScale(m_simulationParameters.gridScale);

	m_terrainMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RG32F, false);
	m_resistanceMap->reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, GL_RGBA32F, false);
}

void Simulator::setupArrays()
{
	m_windArray.reinitialize(m_simulationParameters.gridSize.x, m_simulationParameters.gridSize.y, cudaCreateChannelDesc<float2>());
	m_launchParameters.windArray.surface = m_windArray.recreateSurface();
	m_launchParameters.windArray.texture = m_windArray.recreateTexture(m_textureDescriptor);

	m_terrainArray.reinitialize(*m_terrainMap);
	m_resistanceArray.reinitialize(*m_resistanceMap);
}

void Simulator::setupBuffers()
{
	m_tmpBuffer.reinitialize(4 * m_simulationParameters.cellCount, sizeof(float));
	m_launchParameters.tmpBuffer = m_tmpBuffer.getData<float>();
}

void Simulator::setupMultigrid()
{
	int2 gridSize{ m_simulationParameters.gridSize };
	float gridScale{ m_simulationParameters.gridScale };
	int cellCount{ m_simulationParameters.cellCount };

	m_multigrid.resize(m_launchParameters.multigridLevelCount);
	m_launchParameters.multigrid.resize(m_launchParameters.multigridLevelCount);

	for (int i{ 0 }; i < m_launchParameters.multigridLevelCount; ++i)
	{
		sthe::cu::Buffer& buffer{ m_multigrid[i] };
		buffer.reinitialize(4 * cellCount, sizeof(float));

		MultigridLevel& level{ m_launchParameters.multigrid[i] };
		level.gridSize = gridSize;
		level.gridScale = gridScale;
		level.cellCount = cellCount;
		level.terrainBuffer = reinterpret_cast<Buffer<float2>>(buffer.getData<float>());
		level.fluxBuffer = buffer.getData<float>() + 2 * cellCount;
		level.avalancheBuffer = level.fluxBuffer + cellCount;

		gridSize /= 2;
		gridScale *= 2.0f;
		cellCount /= 4;
	}
}

void Simulator::map()
{
	m_simulationParameters.deltaTime = m_launchParameters.timeMode == TimeMode::DeltaTime ? sthe::getApplication().getDeltaTime() : m_fixedDeltaTime;
	m_simulationParameters.deltaTime *= m_timeScale;

	upload(m_simulationParameters);

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
void Simulator::setWindAngle(const float t_windAngle)
{
	const float windAngle{ glm::radians(t_windAngle) };
	m_simulationParameters.windDirection = float2{ glm::cos(windAngle), glm::sin(windAngle) };
}

void Simulator::setWindSpeed(const float t_windSpeed)
{
	m_simulationParameters.windSpeed = t_windSpeed;
}

void Simulator::setVenturiStrength(const float t_venturiStrength)
{
	m_simulationParameters.venturiStrength = t_venturiStrength;
}

void Simulator::setWindShadowDistance(const float t_windShadowDistance)
{
	m_simulationParameters.windShadowDistance = t_windShadowDistance;
}

void Simulator::setMinWindShadowAngle(const float t_minWindShadowAngle)
{
	m_simulationParameters.minWindShadowAngle = glm::tan(glm::radians(t_minWindShadowAngle));
}

void Simulator::setMaxWindShadowAngle(const float t_maxWindShadowAngle)
{
	m_simulationParameters.maxWindShadowAngle = glm::tan(glm::radians(t_maxWindShadowAngle));
}

void Simulator::setSaltationStrength(const float t_saltationStrength)
{
	m_simulationParameters.saltationStrength = t_saltationStrength;
}

void Simulator::setReptationStrength(const float t_reptationStrength)
{
	m_simulationParameters.reptationStrength = t_reptationStrength;
}

void Simulator::setAvalancheMode(const AvalancheMode t_avalancheMode)
{
	m_launchParameters.avalancheMode = t_avalancheMode;
}

void Simulator::setAvalancheIterations(const int t_avalancheIterations)
{
	m_launchParameters.avalancheIterations = t_avalancheIterations;
}

void Simulator::setAvalancheFinalSoftIterations(const int t_avalancheFinalSoftIterations)
{
	m_launchParameters.avalancheFinalSoftIterations = t_avalancheFinalSoftIterations;
}

void Simulator::setAvalancheSoftIterationModulus(const int t_avalancheSoftIterationModulus)
{
	m_launchParameters.avalancheSoftIterationModulus = t_avalancheSoftIterationModulus;
}

void Simulator::setAvalancheStrength(const float t_avalancheStrength)
{
	m_simulationParameters.avalancheStrength = t_avalancheStrength;
}

void Simulator::setAvalancheAngle(const float t_avalancheAngle)
{
	m_simulationParameters.avalancheAngle = glm::tan(glm::radians(t_avalancheAngle));
}

void Simulator::setVegetationAngle(const float t_vegetationAngle)
{
	m_simulationParameters.vegetationAngle = glm::tan(glm::radians(t_vegetationAngle));
}

void Simulator::setMultigridLevelCount(const int t_multigridLevelCount)
{
	STHE_ASSERT(t_multigridLevelCount >= 1, "Multigrid level count must be greater than or equal to 1");

	m_launchParameters.multigridLevelCount = t_multigridLevelCount;

	if (m_isAwake)
	{
		setupMultigrid();
	}
}

void Simulator::setMultigridPresweepCount(const int t_multigridPresweepCount)
{
	m_launchParameters.multigridPresweepCount = t_multigridPresweepCount;
}

void Simulator::setTimeMode(const TimeMode t_timeMode)
{
	m_launchParameters.timeMode = t_timeMode;
}

void Simulator::setTimeScale(const float t_timeScale)
{
	m_timeScale = t_timeScale;
}

void Simulator::setFixedDeltaTime(const float t_fixedDeltaTime)
{
	m_fixedDeltaTime = t_fixedDeltaTime;
}

// Getters
bool Simulator::isPaused() const
{
	return m_isPaused;
}

}
