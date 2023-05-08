#include <sthe/sthe.hpp>
#include <dunes/dunes.hpp>

int main()
{
	sthe::Application& application{ sthe::createApplication("Demo", 1200, 800) };
	sthe::Scene& scene{ application.addScene() };
	sthe::GameObject& light{ scene.addDirectionalLight() };
	sthe::GameObject& camera{ scene.addTrackball() };

	application.run();
}
