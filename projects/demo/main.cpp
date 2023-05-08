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

	const std::shared_ptr<sthe::Terrain> terrain{ std::make_shared<sthe::Terrain>(glm::vec3{ 20.0f, 1.0f, 20.0f }) };
	scene.addTerrain(terrain);

	//GL_CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
	
	application.run();
}
