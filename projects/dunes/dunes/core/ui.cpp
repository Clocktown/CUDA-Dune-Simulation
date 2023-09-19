#include "ui.hpp"
#include "simulator.hpp"
#include <sthe/sthe.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tinyfiledialogs/tinyfiledialogs.h>
#include <tinyexr.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

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
		m_simulator->setStopIterations(m_stopIterations);
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

	bool SaveEXR(const float* rgba, int width, int height, const char* outfilename, int tinyexr_pixeltype) {

		EXRHeader header;
		InitEXRHeader(&header);

		EXRImage image;
		InitEXRImage(&image);

		image.num_channels = 4;

		std::vector<float> images[4];
		images[0].resize(width * height);
		images[1].resize(width * height);
		images[2].resize(width * height);
		images[3].resize(width * height);

		// Split RGBRGBRGB... into R, G, B and A layer
		for (int i = 0; i < width * height; i++) {
			images[0][i] = rgba[4 * i + 0];
			images[1][i] = rgba[4 * i + 1];
			images[2][i] = rgba[4 * i + 2];
			images[3][i] = rgba[4 * i + 3];
		}

		float* image_ptr[4];
		image_ptr[0] = &(images[3].at(0)); // A
		image_ptr[1] = &(images[2].at(0)); // B
		image_ptr[2] = &(images[1].at(0)); // G
		image_ptr[3] = &(images[0].at(0)); // R

		image.images = (unsigned char**)image_ptr;
		image.width = width;
		image.height = height;

		header.num_channels = 4;
		header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
		// Must be (A)BGR order, since most of EXR viewers expect this channel order.
		strncpy(header.channels[0].name, "A", 255); header.channels[0].name[strlen("A")] = '\0';
		strncpy(header.channels[1].name, "B", 255); header.channels[1].name[strlen("B")] = '\0';
		strncpy(header.channels[2].name, "G", 255); header.channels[2].name[strlen("G")] = '\0';
		strncpy(header.channels[3].name, "R", 255); header.channels[3].name[strlen("R")] = '\0';

		header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		for (int i = 0; i < header.num_channels; i++) {
			header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
			header.requested_pixel_types[i] = tinyexr_pixeltype; // pixel type of output image to be stored in .EXR
		}

		const char* err = nullptr; // or nullptr in C++11 or later.
		int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
		free(header.channels);
		free(header.pixel_types);
		free(header.requested_pixel_types);
		if (ret != TINYEXR_SUCCESS) {
			std::string errorMsg = std::string("Could not save file:\n") + outfilename + "\nReason: " + err;
			tinyfd_messageBox("Error", errorMsg.c_str(), "ok", "error", 1);
			FreeEXRErrorMessage(err); // free's buffer for an error message
			return ret;
		}
	}

	bool UI::loadEXR(std::shared_ptr<sthe::gl::Texture2D> map, const std::string& input) {
		float* out; // width * height * RGBA
		int width;
		int height;
		const char* err = nullptr; // or nullptr in C++11

		int ret = LoadEXR(&out, &width, &height, input.c_str(), &err);
		if (ret != TINYEXR_SUCCESS) {
			if (err) {
				std::string errorMsg = std::string("Could not load file:\n") + input + "\nReason: " + err;
				tinyfd_messageBox("Error", errorMsg.c_str(), "ok", "error", 1);
				FreeEXRErrorMessage(err); // release memory of error message.
			}
		}
		else {
			if (width != m_gridSize.x || height != m_gridSize.y) {
				m_gridSize = { width, height };
				m_simulator->setInitializationParameters(m_initializationParameters);
				m_simulator->reinitialize(m_gridSize, m_gridScale);
			}
			map->upload(out, width, height, GL_RGBA, GL_FLOAT);
			free(out); // release memory of image data
			return true;
		}
		return false;
	}

	bool UI::fromJson(const std::string& path) {

	}

	bool UI::toJson(const std::string& path) {
		nlohmann::json json;

		// Application
		json["vSync"] = m_vSync;
		json["calcCoverage"] = m_calcCoverage;
		json["coverageThreshold"] = m_coverageThreshold;
		json["targetFrameRate"] = m_targetFrameRate;
		json["constantCoverage"] = m_constantCoverage;
		json["constantCoverageAllowRemove"] = m_constantCoverageAllowRemove;
		json["targetCoverage"] = m_targetCoverage;
		json["coverageSpawnAmount"] = m_coverageSpawnAmount;
		json["spawnSteps"] = m_spawnSteps;
		json["stopIterations"] = m_stopIterations;

		// Simulation
		json["gridSize"] = { m_gridSize.x, m_gridSize.y };
		json["gridScale"] = m_gridScale;
		json["windAngle"] = m_windAngle;
		json["secondWindAngle"] = m_secondWindAngle;
		json["windBidirectionalR"] = m_windBidirectionalR;
		json["windBidirectionalBaseTime"] = m_windBidirectionalBaseTime;
		json["enableBidirectional"] = m_enableBidirectional;
		json["windSpeed"] = m_windSpeed;
		json["venturiStrength"] = m_venturiStrength;

		json["windWarpingMode"] = windWarpingModes[m_windWarpingMode];
		json["windWarpingCount"] = m_windWarpingCount;
		json["windWarpingDivisor"] = m_windWarpingDivisor;
		json["windWarpingRadii"] = m_windWarpingRadii;
		json["windWarpingStrengths"] = m_windWarpingStrengths;

		json["windShadowMode"] = windShadowModes[m_windShadowMode];
		json["windShadowDistance"] = m_windShadowDistance;
		json["minWindShadowAngle"] = m_minWindShadowAngle;
		json["maxWindShadowAngle"] = m_maxWindShadowAngle;

		json["stickyStrength"] = m_stickyStrength;
		json["stickyAngle"] = m_stickyAngle;
		json["stickyRange"] = { m_stickyRange.x, m_stickyRange.y };
		json["maxStickyHeight"] = m_maxStickyHeight;

		json["abrasionStrength"] = m_abrasionStrength;
		json["abrasionThreshold"] = m_abrasionThreshold;
		json["saltationMode"] = saltationModes[m_saltationMode];
		json["saltationStrength"] = m_saltationStrength;
		json["reptationStrength"] = m_reptationStrength;

		json["avalancheMode"] = avalancheModes[m_avalancheMode];
		json["bedrockAvalancheMode"] = bedrockAvalancheModes[m_bedrockAvalancheMode];
		json["avalancheIterations"] = m_avalancheIterations;
		json["bedrockAvalancheIterations"] = m_bedrockAvalancheIterations;
		json["avalancheSoftIterationModulus"] = m_avalancheSoftIterationModulus;
		json["avalancheFinalSoftIterations"] = m_avalancheFinalSoftIterations;
		json["avalancheStrength"] = m_avalancheStrength;
		json["avalancheAngle"] = m_avalancheAngle;
		json["bedrockAngle"] = m_bedrockAngle;
		json["vegetationAngle"] = m_vegetationAngle;
		json["multigridLevelCount"] = m_multigridLevelCount;
		json["multigridPresweepCount"] = m_multigridPresweepCount;
		json["multigridPostsweepCount"] = m_multigridPostsweepCount;

		int m_timeMode{ static_cast<int>(TimeMode::FixedDeltaTime) };
		float m_timeScale{ 15.0f };
		float m_fixedDeltaTime{ 0.02f };

		json["timeMode"] = timeModes[m_timeMode];
		json["timeScale"] = m_timeScale;
		json["fixedDeltaTime"] = m_fixedDeltaTime;

		json["initializationParameters"] = nlohmann::json::object();
		for (int i = 0; i < NumNoiseGenerationTargets; ++i) {
				auto& params = m_initializationParameters.noiseGenerationParameters[i];
				auto obj = nlohmann::json::object();
				obj["flat"] = params.flat;
				obj["enable"] = params.enabled;
				obj["iterations"] = params.iters;
				obj["stretch"] = { params.stretch.x, params.stretch.y };
				obj["offset"] = { params.offset.x, params.offset.y };
				obj["border"] = { params.border.x, params.border.y };
				obj["scale"] = params.scale;
				obj["bias"] = params.bias;
				json["initializationParameters"][initializationTargets[i]] = obj;
		}

		json["sandColor"] = { m_renderParameters.sandColor.x,
			m_renderParameters.sandColor.y,
			m_renderParameters.sandColor.z,
			m_renderParameters.sandColor.w
		};

		json["bedrockColor"] = { m_renderParameters.bedrockColor.x,
			m_renderParameters.bedrockColor.y,
			m_renderParameters.bedrockColor.z,
			m_renderParameters.bedrockColor.w
		};

		if (m_exportMaps) {
			std::string terrainMapPath = path + ".terrain.exr";
			std::string resistanceMapPath = path + ".resistance.exr";
			const int width = m_simulator->getTerrainMap()->getWidth();
			const int height = m_simulator->getTerrainMap()->getHeight();
			std::vector<float> data(width * height * 4);
			m_simulator->getTerrainMap()->download(data,
				width,
				height,
				GL_RGBA,
				GL_FLOAT,
				0);
			SaveEXR(data.data(), width, height, terrainMapPath.c_str(), TINYEXR_PIXELTYPE_FLOAT);
			m_simulator->getResistanceMap()->download(data,
				width,
				height,
				GL_RGBA,
				GL_FLOAT,
				0);
			SaveEXR(data.data(), width, height, resistanceMapPath.c_str(), TINYEXR_PIXELTYPE_HALF);
		}

		auto str = json.dump(1);
		std::ofstream o(path);
		o << str;
		o.close();
		return o.good();
	}

	void UI::createSceneNode()
	{
		if (ImGui::TreeNode("Scene"))
		{
			if (ImGui::Button("Reset"))
			{
				m_simulator->setInitializationParameters(m_initializationParameters);
				m_simulator->reinitialize(m_gridSize, m_gridScale);
				if (!m_heightMapPath.empty()) {
					loadEXR(m_simulator->getTerrainMap(), m_heightMapPath);
				}
				if (!m_resistanceMapPath.empty()) {
					loadEXR(m_simulator->getResistanceMap(), m_resistanceMapPath);
				}
			}

			char const* filterPatterns[1] = { "*.exr" };
			if (ImGui::Button("Load Heights from EXR")) {
				auto input = tinyfd_openFileDialog("Load Heightmap", "./", 1, filterPatterns, "OpenEXR (.exr)", 0);

				if (input != nullptr) {
					m_heightMapPath = input;
					if (!loadEXR(m_simulator->getTerrainMap(), m_heightMapPath)) {
						m_heightMapPath = "";
					}
				}
				else {
					m_heightMapPath = "";
				}
			}
			ImGui::LabelText("Selected File##height", m_heightMapPath.empty() ? "None" : m_heightMapPath.c_str());
			ImGui::SameLine();
			if (ImGui::Button("Clear##height")) {
				m_heightMapPath = "";
			}
			if (ImGui::Button("Load Resistances from EXR")) {
				auto input = tinyfd_openFileDialog("Load Resistancemap", "./", 1, filterPatterns, "OpenEXR (.exr)", 0);

				if (input != nullptr) {
					m_resistanceMapPath = input;
					if (!loadEXR(m_simulator->getResistanceMap(), m_resistanceMapPath)) {
						m_resistanceMapPath = "";
					}
				}
				else {
					m_resistanceMapPath = "";
				}
			}
			ImGui::LabelText("Selected File##resistance", m_resistanceMapPath.empty() ? "None" : m_resistanceMapPath.c_str());
			ImGui::SameLine();
			if (ImGui::Button("Clear##resistance")) {
				m_resistanceMapPath = "";
			}

			if (ImGui::Button("Save Heights to EXR")) {
				auto output = tinyfd_saveFileDialog("Save Heightmap", "./heights.exr", 1, filterPatterns, "OpenEXR (.exr)");
				if (output != nullptr) {
					const int width = m_simulator->getTerrainMap()->getWidth();
					const int height = m_simulator->getTerrainMap()->getHeight();
					std::vector<float> data(width * height * 4);
					m_simulator->getTerrainMap()->download(data,
						width,
						height,
						GL_RGBA,
						GL_FLOAT,
						0);
					SaveEXR(data.data(), width, height, output, TINYEXR_PIXELTYPE_FLOAT);
				}
			}
			if (ImGui::Button("Save Resistances to EXR")) {
				auto output = tinyfd_saveFileDialog("Save Resistancemap", "./resistances.exr", 1, filterPatterns, "OpenEXR (.exr)");
				if (output != nullptr) {
					const int width = m_simulator->getResistanceMap()->getWidth();
					const int height = m_simulator->getResistanceMap()->getHeight();
					std::vector<float> data(width * height * 4);
					m_simulator->getResistanceMap()->download(data,
						width,
						height,
						GL_RGBA,
						GL_FLOAT,
						0);
					SaveEXR(data.data(), width, height, output, TINYEXR_PIXELTYPE_HALF); // Half Precision should be enough here since all values are [0,1]
				}
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
		if (ImGui::TreeNode("Save/Load JSON")) {
			ImGui::Checkbox("Export Maps to EXR", &m_exportMaps);
			if (ImGui::Button("Save")) {
				char const* filterPatterns[1] = { "*.json" };
				auto output = tinyfd_saveFileDialog("Save JSON", "./scene.json", 1, filterPatterns, "JSON (.json)");
				if (output != nullptr) {
					toJson(output);
				}
			}
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

			if (ImGui::DragInt("Stop after", &m_stopIterations, 0.1f, 0, 10000)) {
				m_simulator->setStopIterations(m_stopIterations);
			}
			ImGui::Text("Iterations: %i", m_simulator->getTimeStep());

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
				if (m_simulator->isPaused() && ImGui::Button("Update##windshadow")) {
					m_simulator->updateWindShadow();
				}
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
				if (m_simulator->isPaused() && ImGui::Button("Update##sticky")) {
					m_simulator->updateStickyCells();
				}
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

				if (ImGui::DragFloat("Soft Iteration Strength", &m_avalancheStrength, 0.001f, 0.f, 1.f))
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
