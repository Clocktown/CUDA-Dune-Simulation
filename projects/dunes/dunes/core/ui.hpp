#pragma once

#include "simulator.hpp"
#include <sthe/sthe.hpp>

namespace dunes
{

class UI : public sthe::Component
{
public:
	// Constructors
	UI() = default;
	UI(const UI& t_ui) = default;
	UI(UI&& t_ui) = default;

	// Destructor
	~UI() = default;

	// Operators
	UI& operator=(const UI& t_ui) = default;
	UI& operator=(UI&& t_ui) = default;

	// Functionality
	void awake();
	void onGUI();
private:
	// Static
	static inline const char* saltationModes[2]{ "Per Frame", "Continuous" };
	static inline const char* windWarpingModes[2]{ "None", "Standard" };
	static inline const char* windShadowModes[2]{ "Linear", "Curved" };
	static inline const char* avalancheModes[5]{ "Atomic Buffered", "Atomic In-Place", "Shared In-Place", "Mixed In-Place", "Multigrid" };
	static inline const char* timeModes[2]{ "Delta Time", "Fixed Delta Time" };
	static inline const char* initializationTargets[NumNoiseGenerationTargets]{ "Bedrock", "Sand", "Vegetation", "Abrasion Resistance" };

	// Functionality
	void createApplicationNode();
	void createRenderingNode();
	void createSceneNode();
	void createSimulationNode();

	// Attributes
	Simulator* m_simulator{ nullptr };

	// Application
	bool m_vSync{ false };
	bool m_calcCoverage{false};
	float m_coverageThreshold{ 0.001f };
	int m_targetFrameRate{ 0 };

	// Simulation
	glm::ivec2 m_gridSize{ 2048, 2048 };
	float m_gridScale{ 1.0f };

	float m_windAngle{ 0.0f };
	float m_windSpeed{ 30.0f };

	float m_venturiStrength{ 0.005f };

	int m_windWarpingMode{ static_cast<int>(WindWarpingMode::None) };
	int m_windWarpingCount{ 2 };
	float m_windWarpingDivisor{ 20.0f };
	std::array<float, 4> m_windWarpingRadii{ 200.0f, 50.0f, 0.0f, 0.0f };
	std::array<float, 4> m_windWarpingStrengths{ 0.8f, 0.2f, 0.0f, 0.0f };

	int m_windShadowMode{ static_cast<int>(WindShadowMode::Linear) };
	float m_windShadowDistance{ 10.0f };
	float m_minWindShadowAngle{ 10.0f };
	float m_maxWindShadowAngle{ 15.0f };

	float m_stickyStrength{ 1.0f };
	float m_stickyAngle{ 55.0f };
	float2 m_stickyRange{ 0.4f, 2.0f };
	float m_maxStickyHeight{ 30.0f };

	float m_abrasionStrength{ 0.0f };
	float m_abrasionThreshold{ 0.1f };
	int m_saltationMode{ static_cast<int>(SaltationMode::PerFrame) };
	float m_saltationStrength{ 0.5f };
	float m_reptationStrength{ 0.0f };

	int m_avalancheMode{ static_cast<int>(AvalancheMode::AtomicInPlace) };
	int m_avalancheIterations{ 50 };
	int m_avalancheSoftIterationModulus{ 10 };
	int m_avalancheFinalSoftIterations{ 5 };
	float m_avalancheStrength{ 0.5f };
	float m_avalancheAngle{ 33.0f };
	float m_vegetationAngle{ 45.0f };
	int m_multigridLevelCount{ 3 };
	int m_multigridPresweepCount{ 0 };
	int m_multigridPostsweepCount{ 0 };

	int m_timeMode{ static_cast<int>(TimeMode::DeltaTime) };
	float m_timeScale{ 15.0f };
	float m_fixedDeltaTime{ 0.02f };

	InitializationParameters m_initializationParameters{};
	RenderParameters m_renderParameters{};
};

}
