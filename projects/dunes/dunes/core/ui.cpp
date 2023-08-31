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
		m_simulator->setWindWarpingDivisor(m_windWarpingDivisor);

		for (int i{ 0 }; i < 4; ++i)
		{
			m_simulator->setWindWarpingRadius(i, m_windWarpingRadii[i]);
		}

		for (int i{ 0 }; i < 4; ++i)
		{
			m_simulator->setWindWarpingStrength(i, m_windWarpingStrengths[i]);
		}

		m_simulator->setWindShadowMode(static_cast<WindShadowMode>(m_windShadowMode));
		m_simulator->setWindShadowDistance(m_windShadowDistance);
		m_simulator->setMinWindShadowAngle(m_minWindShadowAngle);
		m_simulator->setMaxWindShadowAngle(m_maxWindShadowAngle);
		m_simulator->setStickyStrength(m_stickyStrength);
		m_simulator->setStickyAngle(m_stickyAngle);
		m_simulator->setStickyRange(m_stickyRange);
		m_simulator->setMaxStickyHeight(m_maxStickyHeight);
		m_simulator->setAbrasionStrength(m_abrasionStrength);
		m_simulator->setAbrasionThreshold(m_abrasionThreshold);
		m_simulator->setSaltationMode(static_cast<SaltationMode>(m_saltationMode));
		m_simulator->setSaltationStrength(m_saltationStrength);
		m_simulator->setReptationStrength(m_reptationStrength);
		m_simulator->setAvalancheMode(static_cast<AvalancheMode>(m_avalancheMode));
		m_simulator->setBedrockAvalancheMode(static_cast<BedrockAvalancheMode>(m_bedrockAvalancheMode));
		m_simulator->setAvalancheFinalSoftIterations(m_avalancheFinalSoftIterations);
		m_simulator->setAvalancheSoftIterationModulus(m_avalancheSoftIterationModulus);
		m_simulator->setAvalancheIterations(m_avalancheIterations);
		m_simulator->setBedrockAvalancheIterations(m_bedrockAvalancheIterations);
		m_simulator->setAvalancheStrength(m_avalancheStrength);
		m_simulator->setAvalancheAngle(m_avalancheAngle);
		m_simulator->setBedrockAngle(m_bedrockAngle);
		m_simulator->setVegetationAngle(m_vegetationAngle);
		m_simulator->setMultigridLevelCount(m_multigridLevelCount);
		m_simulator->setMultigridPresweepCount(m_multigridPresweepCount);
		m_simulator->setMultigridPostsweepCount(m_multigridPostsweepCount);
		m_simulator->setTimeMode(static_cast<TimeMode>(m_timeMode));
		m_simulator->setTimeScale(m_timeScale);
		m_simulator->setFixedDeltaTime(m_fixedDeltaTime);

		m_simulator->setCoverageThreshold(m_coverageThreshold);
		m_simulator->setTargetCoverage(m_targetCoverage);
		m_simulator->setCoverageSpawnAmount(m_coverageSpawnAmount);
		m_simulator->setSpawnSteps(m_spawnSteps);
		m_simulator->setConstantCoverage(m_constantCoverage);
		m_simulator->setConstantCoverageAllowRemove(m_constantCoverageAllowRemove);

		m_simulator->setSecondWindAngle(m_secondWindAngle);
		m_simulator->enableBidirectional(m_enableBidirectional);
		m_simulator->setBidirectionalBaseTime(m_windBidirectionalBaseTime);
		m_simulator->setBidirectionalR(m_windBidirectionalR);
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

			if (ImGui::DragInt("Target Frame Rate", &m_targetFrameRate))
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
			dirty |= ImGui::ColorEdit3("Vegetation Color", glm::value_ptr(m_renderParameters.vegetationColor));


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
			ImGui::DragFloat("Grid Scale", &m_gridScale);

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

			if (ImGui::TreeNode("Coverage")) {
				if (ImGui::Checkbox("Calculate Coverage", &m_calcCoverage)) {
					if (m_calcCoverage) {
						m_simulator->setupCoverageCalculation();
					}
					else {
						m_simulator->cleanupCoverageCalculation();
					}
				}
				if (m_calcCoverage) {
					if (ImGui::Checkbox("Constant Coverage", &m_constantCoverage)) {
						m_simulator->setConstantCoverage(m_constantCoverage);
					}
					if (ImGui::Checkbox("Allow Removal", &m_constantCoverageAllowRemove)) {
						m_simulator->setConstantCoverageAllowRemove(m_constantCoverageAllowRemove);
					}
					if (ImGui::DragFloat("Target Coverage", &m_targetCoverage)) {
						m_simulator->setTargetCoverage(m_targetCoverage);
					}
					if (ImGui::DragFloat("Spawn Amount", &m_coverageSpawnAmount)) {
						m_simulator->setCoverageSpawnAmount(m_coverageSpawnAmount);
					}
					if (ImGui::DragInt("Spawn every n steps", &m_spawnSteps)) {
						m_simulator->setSpawnSteps(m_spawnSteps);
					}
				}
				ImGui::DragFloat("Threshold", &m_coverageThreshold, 0.0001f, 0.f, 1.f, "%.6f");
				m_simulator->setCoverageThreshold(m_coverageThreshold);
				ImGui::Text("Coverage: %f%", m_simulator->getCoverage() * 100.f);
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Wind")) {
				if (ImGui::DragFloat("Speed", &m_windSpeed))
				{
					m_simulator->setWindSpeed(m_windSpeed);
				}
				if (ImGui::DragFloat("Angle", &m_windAngle))
				{
					m_simulator->setWindAngle(m_windAngle);
				}

				if (ImGui::DragFloat("Venturi", &m_venturiStrength, 0.005f))
				{
					m_simulator->setVenturiStrength(m_venturiStrength);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Bidirectional Wind Scheme"))
			{
				if (ImGui::Checkbox("Enable", &m_enableBidirectional))
				{
					m_simulator->enableBidirectional(m_enableBidirectional);
				}
				if (ImGui::DragFloat("Second Angle", &m_secondWindAngle))
				{
					m_simulator->setSecondWindAngle(m_secondWindAngle);
				}
				if (ImGui::DragFloat("Ratio", &m_windBidirectionalR))
				{
					m_simulator->setBidirectionalR(m_windBidirectionalR);
				}
				if (ImGui::DragFloat("Period", &m_windBidirectionalBaseTime))
				{
					m_simulator->setBidirectionalBaseTime(m_windBidirectionalBaseTime);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Wind Warping"))
			{

				if (ImGui::Combo("Mode", &m_windWarpingMode, windWarpingModes, IM_ARRAYSIZE(windWarpingModes)))
				{
					m_simulator->setWindWarpingMode(static_cast<WindWarpingMode>(m_windWarpingMode));
				}

				if (ImGui::DragInt("Count", &m_windWarpingCount))
				{
					m_simulator->setWindWarpingCount(m_windWarpingCount);
				}

				if (ImGui::DragFloat("Divisor", &m_windWarpingDivisor))
				{
					m_simulator->setWindWarpingDivisor(m_windWarpingDivisor);
				}

				if (ImGui::DragFloat4("Strenghts", m_windWarpingStrengths.data(), 0.05f))
				{
					for (int i{ 0 }; i < 4; ++i)
					{
						m_simulator->setWindWarpingStrength(i, m_windWarpingStrengths[i]);
					}
				}

				if (ImGui::DragFloat4("Radii", m_windWarpingRadii.data()))
				{
					for (int i{ 0 }; i < 4; ++i)
					{
						m_simulator->setWindWarpingRadius(i, m_windWarpingRadii[i]);
					}
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Wind Shadow"))
			{
				if (ImGui::Combo("Mode", &m_windShadowMode, windShadowModes, IM_ARRAYSIZE(windShadowModes)))
				{
					m_simulator->setWindShadowMode(static_cast<WindShadowMode>(m_windShadowMode));
				}

				if (ImGui::DragFloat("Distance", &m_windShadowDistance))
				{
					m_simulator->setWindShadowDistance(m_windShadowDistance);
				}

				if (ImGui::DragFloat("Min. Angle", &m_minWindShadowAngle))
				{
					m_simulator->setMinWindShadowAngle(m_minWindShadowAngle);
				}

				if (ImGui::DragFloat("Max. Angle", &m_maxWindShadowAngle))
				{
					m_simulator->setMaxWindShadowAngle(m_maxWindShadowAngle);
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Echo Dunes"))
			{
				if (ImGui::DragFloat("Strength", &m_stickyStrength, 0.01f))
				{
					m_simulator->setStickyStrength(m_stickyStrength);
				}

				if (ImGui::DragFloat2("Range", &m_stickyRange.x, 0.05f))
				{
					m_simulator->setStickyRange(m_stickyRange);
				}

				if (ImGui::DragFloat("Max. Height", &m_maxStickyHeight, 0.05f))
				{
					m_simulator->setMaxStickyHeight(m_maxStickyHeight);
				}
				if (ImGui::DragFloat("Angle", &m_stickyAngle))
				{
					m_simulator->setStickyAngle(m_stickyAngle);
				}
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Saltation"))
			{
				if (ImGui::Combo("Mode", &m_saltationMode, saltationModes, IM_ARRAYSIZE(saltationModes)))
				{
					m_simulator->setSaltationMode(static_cast<SaltationMode>(m_saltationMode));
				}

				if (ImGui::DragFloat("Strength", &m_saltationStrength, 0.05f))
				{
					m_simulator->setSaltationStrength(m_saltationStrength);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Abrasion"))
			{
				if (ImGui::DragFloat("Strength", &m_abrasionStrength, 0.05f))
				{
					m_simulator->setAbrasionStrength(m_abrasionStrength);
				}

				if (ImGui::DragFloat("Threshold", &m_abrasionThreshold, 0.05f))
				{
					m_simulator->setAbrasionThreshold(m_abrasionThreshold);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Reptation"))
			{
				if (ImGui::DragFloat("Strength", &m_reptationStrength, 0.05f))
				{
					m_simulator->setReptationStrength(m_reptationStrength);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Avalanching"))
			{
				if (ImGui::Combo("Mode", &m_avalancheMode, avalancheModes, IM_ARRAYSIZE(avalancheModes)))
				{
					m_simulator->setAvalancheMode(static_cast<AvalancheMode>(m_avalancheMode));
				}

				if (ImGui::DragInt("Iterations", &m_avalancheIterations))
				{
					m_simulator->setAvalancheIterations(m_avalancheIterations);
				}

				if (ImGui::DragInt("Soft Iterations", &m_avalancheFinalSoftIterations))
				{
					m_simulator->setAvalancheFinalSoftIterations(m_avalancheFinalSoftIterations);
				}

				if (ImGui::DragInt("Soft Iteration Modulus", &m_avalancheSoftIterationModulus))
				{
					m_simulator->setAvalancheSoftIterationModulus(m_avalancheSoftIterationModulus);
				}

				if (ImGui::DragFloat("Strength", &m_avalancheStrength, 0.05f))
				{
					m_simulator->setAvalancheStrength(m_avalancheStrength);
				}

				if (ImGui::DragFloat("Sand Angle", &m_avalancheAngle))
				{
					m_simulator->setAvalancheAngle(m_avalancheAngle);
				}

				if (ImGui::DragFloat("Vegetation Angle", &m_vegetationAngle))
				{
					m_simulator->setVegetationAngle(m_vegetationAngle);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Multigrid Avalanching"))
			{
				if (ImGui::DragInt("Level Count", &m_multigridLevelCount))
				{
					m_simulator->setMultigridLevelCount(m_multigridLevelCount);
				}

				if (ImGui::DragInt("Presweep Count", &m_multigridPresweepCount))
				{
					m_simulator->setMultigridPresweepCount(m_multigridPresweepCount);
				}

				if (ImGui::DragInt("Postsweep Count", &m_multigridPostsweepCount))
				{
					m_simulator->setMultigridPostsweepCount(m_multigridPostsweepCount);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Bedrock Avalanching"))
			{
				if (ImGui::Combo("Mode", &m_bedrockAvalancheMode, bedrockAvalancheModes, IM_ARRAYSIZE(bedrockAvalancheModes)))
				{
					m_simulator->setBedrockAvalancheMode(static_cast<BedrockAvalancheMode>(m_bedrockAvalancheMode));
				}

				if (ImGui::DragInt("Iterations", &m_bedrockAvalancheIterations))
				{
					m_simulator->setBedrockAvalancheIterations(m_bedrockAvalancheIterations);
				}

				if (ImGui::DragFloat("Angle", &m_bedrockAngle))
				{
					m_simulator->setBedrockAngle(m_bedrockAngle);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Time"))
			{
				if (ImGui::Combo("Mode", &m_timeMode, timeModes, IM_ARRAYSIZE(timeModes)))
				{
					m_simulator->setTimeMode(static_cast<TimeMode>(m_timeMode));
				}

				if (ImGui::DragFloat("Scale", &m_timeScale))
				{
					m_simulator->setTimeScale(m_timeScale);
				}

				if (ImGui::DragFloat("Fixed Delta Time", &m_fixedDeltaTime, 0.05f))
				{
					m_simulator->setFixedDeltaTime(m_fixedDeltaTime);
				}

				ImGui::TreePop();
			}

			ImGui::TreePop();
		}
	}

}
