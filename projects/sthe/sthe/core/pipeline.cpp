#include "pipeline.hpp"
#include "scene.hpp"

namespace sthe
{

// Functionality
void Pipeline::use()
{
	
}

void Pipeline::disuse()
{

}

void Pipeline::render(const Scene& t_scene)
{
	if (t_scene.hasMainCamera())
	{
		render(t_scene, *t_scene.getMainCamera());
	}
}

}
