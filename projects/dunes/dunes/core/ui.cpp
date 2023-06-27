#include "ui.hpp"
#include "simulator.hpp"
#include <sthe/sthe.hpp>
#include <glm/gtc/type_ptr.hpp>

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
	m_simulator->setWindWarpingMode(static_cast<WindWarpingMode>(m_windWarpingMode));
	m_simulator->setWindWarpingCount(m_windWarpingCount);
	
	for (int i{ 0 }; i < 4; ++i)
	{
		m_simulator->setWindWarpingRadius(i, m_windWarpingRadii[i]);
	}

	for (int i{ 0 }; i < 4; ++i)
	{
		m_simulator->setWindWarpingStrength(i, m_windWarpingStrengths[i]);
	}
	
	m_simulator->setWindShadowDistance(m_windShadowDistance);
	m_simulator->setMinWindShadowAngle(m_minWindShadowAngle);
	m_simulator->setMaxWindShadowAngle(m_maxWindShadowAngle);
	m_simulator->setAbrasionStrength(m_abrasionStrength);
	m_simulator->setAbrasionThreshold(m_abrasionThreshold);
	m_simulator->setSaltationMode(static_cast<SaltationMode>(m_saltationMode));
	m_simulator->setSaltationStrength(m_saltationStrength);
	m_simulator->setReptationStrength(m_reptationStrength);
	m_simulator->setAvalancheMode(static_cast<AvalancheMode>(m_avalancheMode));
	m_simulator->setAvalancheFinalSoftIterations(m_avalancheFinalSoftIterations);
	m_simulator->setAvalancheSoftIterationModulus(m_avalancheSoftIterationModulus);
	m_simulator->setAvalancheIterations(m_avalancheIterations);
	m_simulator->setAvalancheStrength(m_avalancheStrength);
	m_simulator->setAvalancheAngle(m_avalancheAngle);
	m_simulator->setVegetationAngle(m_vegetationAngle);
	m_simulator->setMultigridLevelCount(m_multigridLevelCount);
	m_simulator->setMultigridPresweepCount(m_multigridPresweepCount);
	m_simulator->setMultigridPostsweepCount(m_multigridPostsweepCount);
	m_simulator->setTimeMode(static_cast<TimeMode>(m_timeMode));
	m_simulator->setTimeScale(m_timeScale);
	m_simulator->setFixedDeltaTime(m_fixedDeltaTime);
}

void UI::onGUI()
{
	ImGui::Begin("Settings");

	createApplicationNode();
	createRenderingNode();
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

void UI::createRenderingNode()
{
	if (ImGui::TreeNode("Rendering")) 
	{
		bool dirty = false;
		dirty |= ImGui::ColorEdit3("Sand Color", glm::value_ptr(m_renderParameters.sandColor));

		if (dirty) {
			m_simulator->setRenderParameters(m_renderParameters);
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
			m_simulator->setInitializationParameters(m_initializationParameters);
			m_simulator->reinitialize(m_gridSize, m_gridScale);
		}

		for (int i = 0; i < NumNoiseGenerationTargets; ++i) {
			ImGui::PushID(i);
			if (ImGui::TreeNode(initializationTargets[i])) {
				auto& params = m_initializationParameters.noiseGenerationParameters[i];
				ImGui::Checkbox("Flat (Bias only)", &params.flat);
				ImGui::Checkbox("Enable", &params.enabled);
				ImGui::DragInt("Noise Iterations", &params.iters, 0.1f, 0, 50);
				ImGui::DragFloat2("Noise Stretch", &params.stretch.x, 1.f, 0.f, 100.f);
				ImGui::DragFloat2("Noise Offset", &params.offset.x, 1.f);
				ImGui::DragFloat2("Seamless Border", &params.border.x, 0.01f, 0.f, 1.f);
				ImGui::DragFloat("Height Scale", &params.scale, 1.f, 0.f, 10000.f);
				ImGui::DragFloat("Height Bias", &params.bias, 0.1f);
				ImGui::TreePop();
			}
			ImGui::PopID();
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

		if (ImGui::Combo("Wind Warping Mode", &m_windWarpingMode, windWarpingModes, IM_ARRAYSIZE(windWarpingModes)))
		{
			m_simulator->setWindWarpingMode(static_cast<WindWarpingMode>(m_windWarpingMode));
		}

		if (ImGui::InputInt("Wind Warping Count", &m_windWarpingCount))
		{
			m_simulator->setWindWarpingCount(m_windWarpingCount);
		}

		if (ImGui::InputFloat4("Wind Warping Radii", m_windWarpingRadii.data()))
		{
			for (int i{ 0 }; i < 4; ++i)
			{
				m_simulator->setWindWarpingRadius(i, m_windWarpingRadii[i]);
			}
		}

		if (ImGui::InputFloat4("Wind Warping Strenghts", m_windWarpingStrengths.data()))
		{
			for (int i{ 0 }; i < 4; ++i)
			{
				m_simulator->setWindWarpingStrength(i, m_windWarpingStrengths[i]);
			}
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

		if (ImGui::InputFloat("Abrasion Strength", &m_abrasionStrength))
		{
			m_simulator->setAbrasionStrength(m_abrasionStrength);
		}

		if (ImGui::InputFloat("Abrasion Threshold", &m_abrasionThreshold))
		{
			m_simulator->setAbrasionThreshold(m_abrasionThreshold);
		}

		if (ImGui::Combo("Saltation Mode", &m_saltationMode, saltationModes, IM_ARRAYSIZE(saltationModes)))
		{
			m_simulator->setSaltationMode(static_cast<SaltationMode>(m_saltationMode));
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

		if (ImGui::InputInt("Avalanche Soft Iterations", &m_avalancheFinalSoftIterations))
		{
			m_simulator->setAvalancheFinalSoftIterations(m_avalancheFinalSoftIterations);
		}

		if (ImGui::InputInt("Avalanche Soft Iteration Modulus", &m_avalancheSoftIterationModulus))
		{
			m_simulator->setAvalancheSoftIterationModulus(m_avalancheSoftIterationModulus);
		}

		if (ImGui::InputFloat("Avalanche Strength", &m_avalancheStrength))
		{
			m_simulator->setAvalancheStrength(m_avalancheStrength);
		}

		if (ImGui::InputFloat("Avalanche Angle", &m_avalancheAngle))
		{
			m_simulator->setAvalancheAngle(m_avalancheAngle);
		}

		if (ImGui::InputFloat("Vegetation Angle", &m_vegetationAngle))
		{
			m_simulator->setVegetationAngle(m_vegetationAngle);
		}

		if (ImGui::InputInt("Multigrid Level Count", &m_multigridLevelCount))
		{
			m_simulator->setMultigridLevelCount(m_multigridLevelCount);
		}

		if (ImGui::InputInt("Multigrid Presweep Count", &m_multigridPresweepCount))
		{
			m_simulator->setMultigridPresweepCount(m_multigridPresweepCount);
		}

		if (ImGui::InputInt("Multigrid Postsweep Count", &m_multigridPostsweepCount))
		{
			m_simulator->setMultigridPostsweepCount(m_multigridPostsweepCount);
		}

		if (ImGui::InputInt("Avalanche Soft Iteration Modulus", &m_avalancheSoftIterationModulus))
		{
			m_simulator->setAvalancheSoftIterationModulus(m_avalancheSoftIterationModulus);
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
