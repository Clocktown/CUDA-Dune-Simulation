#pragma once

#include "simulation_parameters.hpp"
#include "launch_parameters.hpp"
#include "render_parameters.hpp"
#include <cufft.h>
#include <sthe/sthe.hpp>
#include <sthe/gl/buffer.hpp>
#include <vector>

namespace dunes
{

	class Simulator : public sthe::Component
	{
	public:
		// Constructors
		Simulator();
		Simulator(const Simulator& t_simulator) = delete;
		Simulator(Simulator&& t_simulator) = default;

		// Destructor
		~Simulator();

		// Operators
		Simulator& operator=(const Simulator& t_simulator) = delete;
		Simulator& operator=(Simulator&& t_simulator) = default;

		// Functionality
		void reinitialize(const glm::ivec2& t_gridSize, const float t_gridScale);
		void awake();
		void update();
		void resume();
		void pause();

		void setupCoverageCalculation();
		void calculateCoverage();
		void cleanupCoverageCalculation();

		// Setters
		void setWindAngle(const float t_windAngle);
		void setWindSpeed(const float t_windSpeed);
		void setVenturiStrength(const float t_venturiStrength);
		void setWindWarpingMode(const WindWarpingMode t_windWarpingMode);
		void setWindWarpingCount(const int t_windWarpingCount);
		void setWindWarpingDivisor(const int t_windWarpingDivisor);
		void setWindWarpingRadius(const int t_index, const float t_windWarpingRadius);
		void setWindWarpingStrength(const int t_index, const float t_windWarpingStrength);
		void setWindShadowMode(const WindShadowMode t_windShadowMode);
		void setWindShadowDistance(const float t_windShadowDistance);
		void setMinWindShadowAngle(const float t_minWindShadowAngle);
		void setMaxWindShadowAngle(const float t_maxWindShadowAngle);
		void setStickyAngle(const float t_stickyAngle);
		void setStickyStrength(const float t_stickyStrength);
		void setStickyRange(const float2 t_stickyRange);
		void setMaxStickyHeight(const float t_maxStickyHeight);
		void setAbrasionStrength(const float t_abrasionStrength);
		void setAbrasionThreshold(const float t_abrasionThreshold);
		void setSaltationMode(const SaltationMode t_saltationMode);
		void setSaltationStrength(const float t_saltationStrength);
		void setReptationStrength(const float t_reptationStrength);
		void setAvalancheMode(const AvalancheMode t_avalancheMode);
		void setBedrockAvalancheMode(const BedrockAvalancheMode t_bedrockAvalancheMode);
		void setAvalancheIterations(const int t_avalancheIterations);
		void setAvalancheFinalSoftIterations(const int t_avalancheFinalSoftIterations);
		void setAvalancheSoftIterationModulus(const int t_avalancheSoftIterationModulus);
		void setBedrockAvalancheIterations(const int t_bedrockAvalancheIterations);
		void setAvalancheStrength(const float t_avalancheStrength);
		void setAvalancheAngle(const float t_avalancheAngle);
		void setBedrockAngle(const float t_bedrockAngle);
		void setVegetationAngle(const float t_vegetationAngle);
		void setMultigridLevelCount(const int t_multigridLevelCount);
		void setMultigridPresweepCount(const int t_multigridPresweepCount);
		void setMultigridPostsweepCount(const int t_multigridPostsweepCount);
		void setTimeMode(const TimeMode t_timeMode);
		void setTimeScale(const float t_timeScale);
		void setFixedDeltaTime(const float t_fixedDeltaTime);
		void setInitializationParameters(const InitializationParameters& t_initializationParameters);
		void setRenderParameters(const RenderParameters& t_renderParameters);
		void setCoverageThreshold(const float t_threshold);
		float getCoverage();
		void setTargetCoverage(const float t_targetCoverage);
		void setCoverageSpawnAmount(const float t_amount);
		void setSpawnSteps(const int t_steps);
		void setConstantCoverage(const bool t_constantCoverage);
		void setConstantCoverageAllowRemove(const bool t_constantCoverageAllowRemove);


		void setSecondWindAngle(const float t_windAngle);
		void enableBidirectional(const bool t_enable);
		void setBidirectionalBaseTime(const float t_time);
		void setBidirectionalR(const float t_R);

		void setStopIterations(const int t_stopIterations);

		void updateWindShadow();
		void updateStickyCells();


		// Getters
		int getTimeStep() const;
		bool isPaused() const;
		std::shared_ptr<sthe::gl::Texture2D> getTerrainMap() const;
		std::shared_ptr<sthe::gl::Texture2D> getResistanceMap() const;
	private:
		// Functionality
		void setupLaunchParameters();
		void setupTerrain();
		void setupArrays();
		void setupBuffers();
		void setupWindWarping();
		void setupMultigrid();
		void map();
		void unmap();

		void setWindDirection(const float t_windAngle);

		// Attributes
		SimulationParameters m_simulationParameters;
		LaunchParameters m_launchParameters;
		InitializationParameters m_initializationParameters;
		RenderParameters m_renderParameters;
		float m_timeScale;
		float m_fixedDeltaTime;
		float m_coverage;
		float m_coverageThreshold;
		bool m_constantCoverage{ false };
		bool m_constantCoverageAllowRemove{ false };
		float m_targetCoverage{ 0.5f };
		float m_coverageSpawnAmount{ 0.01f };
		int m_spawnSteps{ 10 };
		int m_timeStep = 0;
		int m_stopIterations = 0;

		float m_time{ 0.f };
		float m_firstWindAngle{ 0.0f };
		float m_secondWindAngle{ 0.0f };
		float m_windBidirectionalR{ 1.f };
		float m_windBidirectionalBaseTime{ 1.f };
		bool m_enableBidirectional{ false };

		sthe::TerrainRenderer* m_terrainRenderer;
		std::shared_ptr<sthe::Terrain> m_terrain;
		std::shared_ptr<sthe::CustomMaterial> m_material;
		std::shared_ptr<sthe::gl::Program> m_program;
		std::shared_ptr<sthe::gl::Texture2D> m_terrainMap;
		std::shared_ptr<sthe::gl::Texture2D> m_windMap;
		std::shared_ptr<sthe::gl::Texture2D> m_resistanceMap;
		std::shared_ptr<sthe::gl::Buffer> m_renderParameterBuffer;

		sthe::cu::Array2D m_terrainArray;
		sthe::cu::Array2D m_windArray;
		sthe::cu::Array2D m_resistanceArray;
		sthe::cu::Buffer m_slabBuffer;
		sthe::cu::Buffer m_tmpBuffer;
		std::array<sthe::cu::Buffer, 4> m_windWarpingBuffers;
		std::vector<sthe::cu::Buffer> m_multigrid;
		std::unique_ptr<sthe::cu::Buffer> m_coverageMap;
		cudaTextureDesc m_textureDescriptor;

		bool m_isAwake;
		bool m_isPaused;
		bool m_reinitializeWindWarping;
	};

}
