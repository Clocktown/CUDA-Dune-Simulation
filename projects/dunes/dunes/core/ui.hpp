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
	// Functionality
	void createApplicationNode();
	void createSceneNode();
	void createSimulationNode();

	// Attributes
	Simulator* m_simulator{ nullptr };

	// Application
	bool m_vSync{ false };
	int m_targetFrameRate{ 60 };

	// Simulation
	glm::ivec2 m_gridSize{ 2048, 2048 };
	float m_gridScale{ 1.0f };

	float m_windAngle{ 0.0f };
	float m_windSpeed{ 10.0f };

	float m_venturiStrength{ 0.005f };

	float m_windShadowDistance{ 10.0f };
	float m_minWindShadowAngle{ 10.0f };
	float m_maxWindShadowAngle{ 15.0f };

	float m_saltationSpeed{ 1.0f };
	float m_reptationStrength{ 1.0f };

	int m_avalancheIterations{ 50 };
	float m_avalancheStrength{ 0.5f };
	float m_avalancheAngle{ 33.0f };
	float m_vegetationAngle{ 45.0f };

	float m_timeScale{ 50.0f };
};

}