#include <sthe/sthe.hpp>
#include <dunes/dunes.hpp>

int main()
{
	sthe::Application& application{ sthe::createApplication("Demo", 1200, 800) };
	application.setTargetFrameRate(0);

	sthe::Scene& scene{ application.addScene() };
	sthe::GameObject& light{ scene.addDirectionalLight() };
	sthe::GameObject& camera{ scene.addTrackball() };
	camera.getComponent<sthe::Camera>()->setClipPlane(glm::vec2{ 0.1f, 1000.0f });
	camera.getTransform().setLocalPosition(glm::vec3{ 0.0f, 15.0f, 15.0f });

	sthe::GameObject& desert{ scene.addGameObject() };
	dunes::Simulator& simulator{ desert.addComponent<dunes::Simulator>() };
	dunes::UI& ui{ desert.addComponent<dunes::UI>() };

	desert.getTransform().setLocalScale(1.0f / 128.0f);

	application.run();
}
