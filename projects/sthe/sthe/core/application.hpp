#pragma once

#include "pipeline.hpp"
#include "scene.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <string>
#include <vector>

namespace sthe
{

class Application
{
public:
	// Constructors
	Application(const Application& t_application) = delete;
	Application(Application&& t_application) = delete;

	// Destructor
	~Application() = default;

	// Operators
	Application& operator=(const Application& t_application) = delete;
	Application& operator=(Application&& t_application) = delete;

	// Functionality
	void run();
	void exit();

	Scene& addScene(const std::string& t_name = std::string{ "Scene" });
	void removeScene(const int t_index);
	void removeScene(const std::string& t_name);
	void removeScene(Scene& t_scene);
	void loadScene(const int t_index);
	void loadScene(const std::string& t_name);
	void loadScene(Scene& t_scene);

	// Setters
	void setName(const std::string& t_name);
	void setVSyncCount(const int t_vSyncCount);
	void setTargetFrameRate(const int t_targetFrameRate);
	void setMaximumDeltaTime(const float t_maximumDeltaTime);
	void setTimeScale(const float t_timeScale);
	void setPipeline(const std::shared_ptr<Pipeline>& t_pipeline);
	void setStartScene(const int t_index);
	void setStartScene(const std::string& t_name);
	void setStartScene(Scene& t_scene);

	// Getters
	const std::string& getName() const;
	int getVSyncCount() const;
	int getTargetFrameRate() const;
	int getFrameCount() const;
	float getRealTime() const;
	float getUnscaledTime() const;
	float getUnscaledDeltaTime() const;
	float getTime() const;
	float getDeltaTime() const;
	float getMaximumDeltaTime() const;
	float getTimeScale() const;
	const std::shared_ptr<Pipeline>& getPipeline() const;
	const Scene& getScene(const int t_index) const;
	Scene& getScene(const int t_index);
	const Scene* getScene(const std::string& t_name) const;
	Scene* getScene(const std::string& t_name);
	const Scene* getActiveScene() const;
	Scene* getActiveScene();
	bool isRunning() const;
private:
	// Static
	static Application& getInstance(const std::string* const t_name);

	// Constructor
	explicit Application(const std::string* const t_name);

	// Attributes
	std::string m_name;
	int m_vSyncCount;
	int m_targetFrameRate;
	int m_frameCount;
	double m_unscaledTime;
	double m_unscaledDeltaTime;
	double m_time;
	double m_deltaTime;
	double m_maximumDeltaTime;
	double m_timeScale;
	bool m_isRunning;
	
	std::shared_ptr<Pipeline> m_pipeline;
	std::vector<std::unique_ptr<Scene>> m_scenes;
	Scene* m_startScene;
	Scene* m_activeScene;
	Scene* m_nextScene;

	// Friends
	friend Application& createApplication(const std::string& t_name, const int t_width, const int t_height);
	friend Application& getApplication();
};

Application& createApplication(const std::string& t_name, const int t_width, const int t_height);
Application& getApplication();

}
