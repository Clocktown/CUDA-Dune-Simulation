#include "application.hpp"
#include "window.hpp"
#include "gui.hpp"
#include "pipeline.hpp"
#include "scene.hpp"
#include "event.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/pipelines/forward_pipeline.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <entt/entt.hpp>
#include <memory>
#include <string>
#include <vector>

namespace sthe
{

Application& createApplication(const std::string& t_name, const int t_width, const int t_height)
{
	createWindow(t_name, t_width, t_height);
	getGUI();

	return Application::getInstance(&t_name);
}

Application& getApplication()
{
	return Application::getInstance(nullptr);
}

// Static
Application& Application::getInstance(const std::string* const t_name)
{
	static Application application{ t_name };
	return application;
}

// Constructor
Application::Application(const std::string* const t_name) :
	m_vSyncCount{ 0 },
	m_targetFrameRate{ 60 },
	m_frameCount{ 0 },
	m_unscaledTime{ 0.0 },
	m_unscaledDeltaTime{ 0.0 },
	m_time{ 0.0 },
	m_deltaTime{ 0.0 },
	m_maximumDeltaTime{ 1.0f / 3.0f },
	m_timeScale{ 1.0f },
	m_isRunning{ false },
	m_pipeline{ std::make_shared<ForwardPipeline>() },
	m_startScene{ nullptr },
	m_activeScene{ nullptr },
	m_nextScene{ nullptr }
{
	STHE_ASSERT(t_name != nullptr, "Application was not created");

	m_name = *t_name;
	m_pipeline->use();

	GLFW_CHECK_ERROR(glfwSwapInterval(m_vSyncCount));
}

// Functionality
void Application::run()
{
	STHE_ASSERT(!m_isRunning, "Application must not be running");
	STHE_ASSERT(m_startScene != nullptr, "Application must have scenes");

	GLFW_CHECK_ERROR(glfwSetTime(0.0));

	m_isRunning = true;
	m_activeScene = m_startScene;
	m_nextScene = m_startScene;

	entt::dispatcher* dispatcher{ &m_activeScene->m_dispatcher };
	dispatcher->trigger<Event::Awake>();
	dispatcher->trigger<Event::Start>();

	double loopTime{ glfwGetTime() };
	double loopDeltaTime{ 0.0 };

	Window& window{ getWindow() };
	GUI& gui{ getGUI() };

	while (m_isRunning && window.isOpen())
	{
		const double realTime{ glfwGetTime() };
		loopDeltaTime += realTime - loopTime;
	
		if (m_activeScene != m_nextScene)
		{
			m_activeScene = m_nextScene;

			dispatcher = &m_activeScene->m_dispatcher;
			dispatcher->trigger<Event::Awake>();
			dispatcher->trigger<Event::Start>();

			loopTime = glfwGetTime();
		}
		else
		{
			loopTime = realTime;
		}

		if (m_vSyncCount > 0 || m_targetFrameRate <= 0 || loopDeltaTime * static_cast<double>(m_targetFrameRate) >= 1.0)
		{
			m_unscaledDeltaTime = loopDeltaTime;
			m_unscaledTime += m_unscaledDeltaTime;
			m_deltaTime = std::min(m_timeScale * m_unscaledDeltaTime, m_maximumDeltaTime);
			m_time += m_deltaTime;

			gui.start();

			dispatcher->trigger<Event::Update>();
			dispatcher->trigger<Event::LateUpdate>();
			dispatcher->trigger<Event::OnPreRender>();

			m_pipeline->render(*m_activeScene);

			dispatcher->trigger<Event::OnRender>();
			dispatcher->trigger<Event::OnPostRender>();
			dispatcher->trigger<Event::OnGUI>();

			gui.render();

			window.update();

			const std::string fps{ std::to_string(1.0 / m_unscaledDeltaTime) + "fps" };
			const std::string ms{ std::to_string(1000.0 * m_unscaledDeltaTime) + "ms" };
			const std::string title{ m_name + " @ " + fps + " / " + ms };
			window.setTitle(title);

			++m_frameCount;
			loopDeltaTime = 0.0;
		}
	}

	m_frameCount = 0;
	m_unscaledTime = 0.0;
	m_unscaledDeltaTime = 0.0;
	m_time = 0.0;
	m_deltaTime = 0.0;
	m_isRunning = false;
	m_activeScene = nullptr;
}

void Application::exit()
{
	m_isRunning = false;
}

Scene& Application::addScene(const std::string& t_name)
{
	Scene& scene{ *m_scenes.emplace_back(std::make_unique<Scene>(t_name)) };

	if (m_startScene == nullptr)
	{
		m_startScene = &scene;
	}

	return scene;
}

void Application::removeScene(const int t_index)
{
	Scene* const scene{ &getScene(t_index) };

	STHE_ASSERT(m_activeScene != scene, "Active scene cannot be removed");

	m_scenes.erase(m_scenes.begin() + t_index);

	if (m_startScene == scene)
	{
		m_startScene = m_scenes[0].get();
	}
}

void Application::removeScene(const std::string& t_name)
{
	auto iterator{ m_scenes.begin() };

	while (iterator != m_scenes.end())
	{
		if ((*iterator)->getName() == t_name)
		{
			break;
		}

		++iterator;
	}

	if (iterator == m_scenes.end())
	{
		return;
	}

	Scene* const scene{ iterator->get() };

	STHE_ASSERT(m_activeScene != scene, "Active scene cannot be removed");

	m_scenes.erase(iterator);

	if (m_startScene == scene)
	{
		m_startScene = m_scenes[0].get();
	}
}

void Application::removeScene(Scene& t_scene)
{
	auto iterator{ m_scenes.begin() };
	Scene* const scene{ &t_scene };

	while (iterator != m_scenes.end())
	{
		if (iterator->get() == scene)
		{
			break;
		}

		++iterator;
	}

	if (iterator == m_scenes.end())
	{
		return;
	}

	STHE_ASSERT(m_activeScene != scene, "Active scene cannot be removed");

	m_scenes.erase(iterator);

	if (m_startScene == scene)
	{
		m_startScene = m_scenes[0].get();
	}
}

void Application::loadScene(const int t_index)
{
	m_nextScene = &getScene(t_index);
}

void Application::loadScene(const std::string& t_name)
{
	m_nextScene = getScene(t_name);

	STHE_ASSERT(m_nextScene != nullptr, "Name must refer to an existing scene");
}

void Application::loadScene(Scene& t_scene)
{
	m_nextScene = &t_scene;
}

// Setters
void Application::setName(const std::string& t_name)
{
	m_name = t_name;
}

void Application::setVSyncCount(const int t_vSyncCount)
{
	m_vSyncCount = t_vSyncCount;
	GLFW_CHECK_ERROR(glfwSwapInterval(t_vSyncCount));
}

void Application::setTargetFrameRate(const int t_targetFrameRate)
{
	m_targetFrameRate = t_targetFrameRate;
}

void Application::setMaximumDeltaTime(const float t_maximumDeltaTime)
{
	STHE_ASSERT(t_maximumDeltaTime >= 0.0f, "Maximum delta time must be greater than or equal to 0");

	m_maximumDeltaTime = t_maximumDeltaTime;
}

void Application::setTimeScale(const float t_timeScale)
{
	STHE_ASSERT(t_timeScale >= 0.0f, "Time scale must be greater than or equal to 0");

	m_timeScale = t_timeScale;
}

void Application::setPipeline(const std::shared_ptr<Pipeline>& t_pipeline)
{
	STHE_ASSERT(t_pipeline != nullptr, "Pipeline cannot be nullptr");

	m_pipeline->disuse();
	m_pipeline = t_pipeline;
	m_pipeline->use();
}

void Application::setStartScene(const int t_index)
{
	STHE_ASSERT(!m_isRunning, "Application cannot be running");

	m_startScene = &getScene(t_index);
}

void Application::setStartScene(const std::string& t_name)
{
	STHE_ASSERT(!m_isRunning, "Application cannot be running");

	m_startScene = getScene(t_name);

	STHE_ASSERT(m_activeScene != nullptr, "Name must refer to an existing scene");
}

void Application::setStartScene(Scene& t_scene)
{
	STHE_ASSERT(!m_isRunning, "Application cannot be running");

	m_startScene = &t_scene;
}

// Getters
const std::string& Application::getName() const
{
	return m_name;
}

int Application::getVSyncCount() const
{
	return m_vSyncCount;
}

int Application::getTargetFrameRate() const
{
	return m_targetFrameRate;
}

int Application::getFrameCount() const
{
	return m_frameCount;
}

float Application::getRealTime() const
{
	return static_cast<float>(glfwGetTime());
}

float Application::getUnscaledTime() const
{
	return static_cast<float>(m_unscaledTime);
}

float Application::getUnscaledDeltaTime() const
{
	return static_cast<float>(m_unscaledDeltaTime);
}

float Application::getTime() const
{
	return static_cast<float>(m_time);
}

float Application::getDeltaTime() const
{
	return static_cast<float>(m_deltaTime);
}

float Application::getMaximumDeltaTime() const
{
	return static_cast<float>(m_maximumDeltaTime);
}

float Application::getTimeScale() const
{
	return static_cast<float>(m_timeScale);
}

const std::shared_ptr<Pipeline>& Application::getPipeline() const
{
	return m_pipeline;
}

const Scene& Application::getScene(const int t_index) const
{
	STHE_ASSERT(t_index >= 0 && t_index < static_cast<int>(m_scenes.size()), "Index must refer to an existing scene");

	return *m_scenes[t_index];
}

Scene& Application::getScene(const int t_index)
{
	STHE_ASSERT(t_index >= 0 && t_index < static_cast<int>(m_scenes.size()), "Index must refer to an existing scene");

	return *m_scenes[t_index];
}

const Scene* Application::getScene(const std::string& t_name) const
{
	for (const std::unique_ptr<Scene>& scene : m_scenes)
	{
		if (scene->getName() == t_name)
		{
			return scene.get();
		}
	}
	
	return nullptr;
}

Scene* Application::getScene(const std::string& t_name)
{
	for (const std::unique_ptr<Scene>& scene : m_scenes)
	{
		if (scene->getName() == t_name)
		{
			return scene.get();
		}
	}

	return nullptr;
}

const Scene* Application::getActiveScene() const
{
	return m_activeScene;
}

Scene* Application::getActiveScene()
{
	return m_activeScene;
}

bool Application::isRunning() const
{
	return m_isRunning;
}

}
