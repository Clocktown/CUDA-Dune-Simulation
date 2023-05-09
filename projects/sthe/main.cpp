#include <sthe/sthe.hpp>

struct CustomComponent : public sthe::Component
{
	// The following methods will be registered (if they exist) and called automatically
	// These methods must NOT be overloaded
	// "getApplication()" and "getWindow()" can be used to access the global state
	// ImGui should be used for the input and the user interface
	// ImGui is not restricted to the "onGUI" function

	void awake()
	{
		// Gets called before any "start" function 
		// Gets called after a game object is added to a running scene
	}

	void start()
	{
		// Gets called before the first frame update 
	}

	void update()
	{
		// Gets called every frame
	}

	void lateUpdate()
	{
		// Gets called every frame after every "update" function
	}

	void onPreRender()
	{
		// Gets called before the render pipeline starts rendering
	}

	void onRender()
	{
		// Gets called after the render pipeline rendered the scene
		// Custom render functions should be placed here
		// The uniform buffer at location "STHE_UNIFORM_BUFFER_PIPELINE" holds informations about the global state (see "pipelines/forward_render.hpp")
	}

	void onPostRender()
	{
		// Gets called after all rendering is complete
		// Can be used for post processing
	}

	void onGUI()
	{
		// Gets called once a frame to generate a user interface
	};
};

int main()
{
	// Create the application
	sthe::Application& application{ sthe::createApplication("Demo", 1200, 800) };

	// Resources should be created as shared_ptr
	// There are several shaders and meshes in the resources folder (see "core/importer.hpp" and "util/io.hpp")
	// Materials without a program will use the pipelines default program (phong shading)
	// Some OpenGL bindings are reserved by the render pipeline (see "config/binding.hpp" or "STHE_XXX" defines)
	std::shared_ptr<sthe::Mesh> mesh{ std::make_shared<sthe::Mesh>(sthe::MeshGeometry::Cube) };
	std::shared_ptr <sthe::Material> material{ std::make_shared<sthe::Material>(glm::vec3{ 1.0f, 0.0f, 1.0f })};

	// Add a scene
	sthe::Scene& scene{ application.addScene() };

	// Add a game object
	// Must be created by using the scene's "add" function
	sthe::GameObject& customGameObject{ scene.addGameObject() };

	// Add a "mesh renderer" component and set the resources
	sthe::MeshRenderer& meshRenderer{ customGameObject.addComponent<sthe::MeshRenderer>() };
	meshRenderer.addMaterial(material);
	meshRenderer.setMesh(mesh);

	// Add a custom component
	// Must be created by using the game object's "add" function
	CustomComponent& customComponent{ customGameObject.addComponent<CustomComponent>() };

	// Add predefined game objects 
	sthe::GameObject& pointLight{ scene.addPointLight() };
	pointLight.getTransform().setLocalPosition(glm::vec3{ 0.0f, 2.0f, -0.25f });

	sthe::GameObject& camera{ scene.addTrackball() };
	camera.getTransform().setLocalPosition(glm::vec3{ 7.0f, 2.0f, -0.25f });

	// Add a model from a file
	// Uses Assimp internally
	sthe::GameObject& sponza{ scene.addModel(sthe::getModelPath() + "sponza.gltf") };

	// Run the application
	application.run();
}
