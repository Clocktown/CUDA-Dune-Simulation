#include "ui.hpp"
#include "simulator.hpp"
#include <sthe/sthe.hpp>

namespace dunes
{

// Functionality
void UI::awake()
{
	m_simulator = getGameObject().getComponent<Simulator>();

	STHE_ASSERT(m_simulator != nullptr, "Simulator cannot be nullptr");

	sthe::Application& application{ sthe::getApplication() };
	application.setVSyncCount(m_vSync);
	application.setTargetFrameRate(m_targetFrameRate);

	m_simulator->pause();
	m_simulator->setWindAngle(m_windAngle);
	m_simulator->setWindSpeed(m_windSpeed);
	m_simulator->setVenturiStrength(m_venturiStrength);
	m_simulator->setWindShadowDistance(m_windShadowDistance);
	m_simulator->setMinWindShadowAngle(m_minWindShadowAngle);
	m_simulator->setMaxWindShadowAngle(m_maxWindShadowAngle);
	m_simulator->setSaltationStrength(m_saltationStrength);
	m_simulator->setReptationStrength(m_reptationStrength);
	m_simulator->setAvalancheMode(static_cast<AvalancheMode>(m_avalancheMode));
	m_simulator->setAvalancheIterations(m_avalancheIterations);
	m_simulator->setAvalancheStrength(m_avalancheStrength);
	m_simulator->setAvalancheAngle(m_avalancheAngle);
	m_simulator->setVegetationAngle(m_vegetationAngle);
	m_simulator->setTimeMode(static_cast<TimeMode>(m_timeMode));
	m_simulator->setTimeScale(m_timeScale);
	m_simulator->setFixedDeltaTime(m_fixedDeltaTime);
}

void UI::onGUI()
{
	ImGui::Begin("Settings");

	createApplicationNode();
	createSceneNode();
	createSimulationNode();

	ImGui::End();
}

void UI::createApplicationNode()
{
	sthe::Application& application{ sthe::getApplication() };

	if (ImGui::TreeNode("Application"))
	{
		if (ImGui::Checkbox("VSync", &m_vSync))
		{
			application.setVSyncCount(m_vSync);
		}

		if (ImGui::InputInt("Target Frame Rate", &m_targetFrameRate))
		{
			application.setTargetFrameRate(m_targetFrameRate);
		}

		ImGui::TreePop();
	}
}

void UI::createSceneNode()
{
	if (ImGui::TreeNode("Scene"))
	{
		if (ImGui::Button("Reset"))
		{
			m_simulator->reinitialize(m_gridSize, m_gridScale);
		}

		ImGui::InputInt2("Grid Size", &m_gridSize.x);
		ImGui::InputFloat("Grid Scale", &m_gridScale);

		ImGui::TreePop();
	}

}

void UI::createSimulationNode()
{
	if (ImGui::TreeNode("Simulation"))
	{
		if (m_simulator->isPaused())
		{
			if (ImGui::Button("Resume"))
			{
				m_simulator->resume();
			}
		}
		else
		{
			if (ImGui::Button("Pause"))
			{
				m_simulator->pause();
			}
		}

		if (ImGui::InputFloat("Wind Angle", &m_windAngle))
		{
			m_simulator->setWindAngle(m_windAngle);
		}

		if (ImGui::InputFloat("Wind Speed", &m_windSpeed))
		{
			m_simulator->setWindSpeed(m_windSpeed);
		}

		if (ImGui::InputFloat("Venturi Strength", &m_venturiStrength))
		{
			m_simulator->setVenturiStrength(m_venturiStrength);
		}

		if (ImGui::InputFloat("Wind Shadow Distance", &m_windShadowDistance))
		{
			m_simulator->setWindShadowDistance(m_windShadowDistance);
		}

		if (ImGui::InputFloat("Min. Wind Shadow Angle", &m_minWindShadowAngle))
		{
			m_simulator->setMinWindShadowAngle(m_minWindShadowAngle);
		}

		if (ImGui::InputFloat("Max. Wind Shadow Angle", &m_maxWindShadowAngle))
		{
			m_simulator->setMaxWindShadowAngle(m_maxWindShadowAngle);
		}

		if (ImGui::InputFloat("Saltation Strength", &m_saltationStrength))
		{
			m_simulator->setSaltationStrength(m_saltationStrength);
		}

		if (ImGui::InputFloat("Reptation Strength", &m_reptationStrength))
		{
			m_simulator->setReptationStrength(m_reptationStrength);
		}

		if (ImGui::Combo("Avalanche Mode", &m_avalancheMode, avalancheModes, IM_ARRAYSIZE(avalancheModes)))
		{
			m_simulator->setAvalancheMode(static_cast<AvalancheMode>(m_avalancheMode));
		}

		if (ImGui::InputInt("Avalanche Iterations", &m_avalancheIterations))
		{
			m_simulator->setAvalancheIterations(m_avalancheIterations);
		}

		if (ImGui::InputFloat("Avalanche Strength", &m_avalancheStrength))
		{
			m_simulator->setAvalancheStrength(m_avalancheStrength);
		}

		if (ImGui::InputFloat("Avalanche Angle", &m_avalancheAngle))
		{
			m_simulator->setAvalancheAngle(m_avalancheAngle);
		}

		if (ImGui::InputFloat("VegetationAngle", &m_vegetationAngle))
		{
			m_simulator->setVegetationAngle(m_vegetationAngle);
		}

		if (ImGui::Combo("Time Mode", &m_timeMode, timeModes, IM_ARRAYSIZE(timeModes)))
		{
			m_simulator->setTimeMode(static_cast<TimeMode>(m_timeMode));
		}

		if (ImGui::InputFloat("Time Scale", &m_timeScale))
		{
			m_simulator->setTimeScale(m_timeScale);
		}

		if (ImGui::InputFloat("Fixed Delta Time", &m_fixedDeltaTime))
		{
			m_simulator->setFixedDeltaTime(m_fixedDeltaTime);
		}

		ImGui::TreePop();
	}
}

}
