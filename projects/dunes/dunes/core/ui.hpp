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
	static inline const char* avalancheModes[5]{ "Atomic Buffered", "Atomic In-Place", "Shared In-Place", "Mixed In-Place", "Multigrid" };
	static inline const char* timeModes[2]{ "Delta Time", "Fixed Delta Time" };

	// Functionality
	void createApplicationNode();
	void createSceneNode();
	void createSimulationNode();

	// Attributes
	Simulator* m_simulator{ nullptr };

	// Application
	bool m_vSync{ false };
	int m_targetFrameRate{ 0 };

	// Simulation
	glm::ivec2 m_gridSize{ 2048, 2048 };
	float m_gridScale{ 1.0f };

	float m_windAngle{ 0.0f };
	float m_windSpeed{ 30.0f };

	float m_venturiStrength{ 0.005f };

	float m_windShadowDistance{ 10.0f };
	float m_minWindShadowAngle{ 10.0f };
	float m_maxWindShadowAngle{ 15.0f };

	float m_abrasionStrength{ 0.0f };
	float m_abrasionThreshold{ 0.1f };
	float m_saltationStrength{ 0.1f };
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
	float m_timeScale{ 10.0f };
	float m_fixedDeltaTime{ 0.02f };
};

}
