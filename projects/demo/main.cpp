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

	const std::shared_ptr<sthe::Terrain> terrain{ std::make_shared<sthe::Terrain>(glm::ivec2{ 512 }) };
	terrain->addLayer(std::make_shared<sthe::TerrainLayer>(glm::vec3(194.0f, 178.0f, 128.0f) / 255.0f));

	sthe::GameObject& desert{ scene.addTerrain(terrain) };
	desert.getTransform().setLocalScale(1.0f / 32.0f);

	//GL_CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
	
	application.run();
}
