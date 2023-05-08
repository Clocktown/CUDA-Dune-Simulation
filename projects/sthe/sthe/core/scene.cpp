#include "scene.hpp"
#include "application.hpp"
#include "environment.hpp"
#include "game_object.hpp"
#include "component.hpp"
#include "mesh.hpp"
#include "material.hpp"
#include "world.hpp"
#include "importer.hpp"
#include <sthe/util/io.hpp>
#include <sthe/components/transform.hpp>
#include <sthe/components/camera.hpp>
#include <sthe/components/trackball.hpp>
#include <sthe/components/light.hpp>
#include <sthe/components/mesh_renderer.hpp>
#include <sthe/components/terrain_renderer.hpp>
#include <sthe/gl/program.hpp>
#include <entt/entt.hpp>
#include <memory>
#include <string>
#include <vector>

namespace sthe
{

// Constructor
Scene::Scene(const std::string& t_name) :
	m_name{ t_name },
	m_mainCamera{ nullptr }
{

}

// Functionality
GameObject& Scene::addGameObject(const std::string& t_name)
{
	GameObject& gameObject{ *m_gameObjects.emplace_back(std::make_unique<GameObject>(t_name)) };
	gameObject.m_handle = m_registry.create();
	gameObject.m_scene = this;
	gameObject.m_transform = &gameObject.addComponent<Transform>();

	return gameObject;
}

GameObject& Scene::addCamera(const std::string& t_name)
{
	GameObject& gameObject{ addGameObject(t_name) };
	Camera* const camera{ &gameObject.addComponent<Camera>() };

	if (m_mainCamera == nullptr)
	{
		setMainCamera(camera);
	}

	return gameObject;
}

GameObject& Scene::addTrackball(const std::string& t_name)
{
	GameObject& gameObject{ addCamera(t_name) };
	gameObject.addComponent<Trackball>();

	return gameObject;
}

GameObject& Scene::addPointLight(const std::string& t_name)
{
	GameObject& gameObject{ addGameObject(t_name) };
	gameObject.addComponent<Light>(LightType::Point);

	return gameObject;
}

GameObject& Scene::addSpotLight(const std::string& t_name)
{
	GameObject& gameObject{ addGameObject(t_name) };

	Transform& transform{ gameObject.getTransform() };
	transform.rotate(-90.0f, World::x, Space::World);

	gameObject.addComponent<Light>(LightType::Spot);

	return gameObject;
}

GameObject& Scene::addDirectionalLight(const std::string& t_name)
{
	GameObject& gameObject{ addGameObject(t_name) };

	Transform& transform{ gameObject.getTransform() };
	transform.setLocalEulerAngles(glm::vec3{ -50.0f, 30.0f, 0.0f });

	gameObject.addComponent<Light>(LightType::Directional);

	return gameObject;
}

GameObject& Scene::addMesh(const std::shared_ptr<Mesh>& t_mesh, const std::shared_ptr<Material>& t_material)
{
	return addMesh(std::string{ "Mesh" }, t_mesh, t_material);
}

GameObject& Scene::addMesh(const std::shared_ptr<Mesh>& t_mesh, const std::vector<std::shared_ptr<Material>>& t_materials)
{
	return addMesh(std::string{ "Mesh" }, t_mesh, t_materials);
}

GameObject& Scene::addMesh(const std::string& t_name, const std::shared_ptr<Mesh>& t_mesh, const std::shared_ptr<Material>& t_material)
{
	GameObject& gameObject{ addGameObject(t_name) };
	gameObject.addComponent<MeshRenderer>(t_mesh, t_material);

	return gameObject;
}

GameObject& Scene::addMesh(const std::string& t_name, const std::shared_ptr<Mesh>& t_mesh, const std::vector<std::shared_ptr<Material>>& t_materials)
{
	GameObject& gameObject{ addGameObject(t_name) };
	gameObject.addComponent<MeshRenderer>(t_mesh, t_materials);

	return gameObject;
}

GameObject& Scene::addQuad(const std::shared_ptr<Material>& t_material)
{
	return addMesh(std::string{ "Quad" }, std::make_shared<Mesh>(Geometry::Quad), t_material);
}

GameObject& Scene::addQuad(const std::string& t_name, const std::shared_ptr<Material>& t_material)
{
	return addMesh(t_name, std::make_shared<Mesh>(Geometry::Quad), t_material);
}

GameObject& Scene::addPlane(const std::shared_ptr<Material>& t_material)
{
	return addMesh(std::string{ "Plane" }, std::make_shared<Mesh>(Geometry::Plane), t_material);
}

GameObject& Scene::addPlane(const std::string& t_name, const std::shared_ptr<Material>& t_material)
{
	return addMesh(t_name, std::make_shared<Mesh>(Geometry::Plane), t_material);
}

GameObject& Scene::addCube(const std::shared_ptr<Material>& t_material)
{
	return addMesh(std::string{ "Cube" }, std::make_shared<Mesh>(Geometry::Cube), t_material);
}

GameObject& Scene::addCube(const std::string& t_name, const std::shared_ptr<Material>& t_material)
{
	return addMesh(t_name, std::make_shared<Mesh>(Geometry::Cube), t_material);
}

GameObject& Scene::addCylinder(const std::shared_ptr<Material>& t_material)
{
	return addMesh(std::string{ "Cylinder" }, std::make_shared<Mesh>(Geometry::Cylinder), t_material);
}

GameObject& Scene::addCylinder(const std::string& t_name, const std::shared_ptr<Material>& t_material)
{
	return addMesh(t_name, std::make_shared<Mesh>(Geometry::Cylinder), t_material);
}

GameObject& Scene::addCone(const std::shared_ptr<Material>& t_material)
{
	return addMesh(std::string{ "Cone" }, std::make_shared<Mesh>(Geometry::Cone), t_material);
}

GameObject& Scene::addCone(const std::string& t_name, const std::shared_ptr<Material>& t_material)
{
	return addMesh(t_name, std::make_shared<Mesh>(Geometry::Cone), t_material);
}

GameObject& Scene::addSphere(const std::shared_ptr<Material>& t_material)
{
	return addMesh(std::string{ "Sphere" }, std::make_shared<Mesh>(Geometry::Sphere), t_material);
}

GameObject& Scene::addSphere(const std::string& t_name, const std::shared_ptr<Material>& t_material)
{
	return addMesh(t_name, std::make_shared<Mesh>(Geometry::Sphere), t_material);
}

GameObject& Scene::addModel(const std::string& t_file, const std::shared_ptr<gl::Program>& t_program, const bool t_hasMipmap)
{
	Importer importer{ t_file };
	return importer.importModel(*this, t_program, t_hasMipmap);
}

GameObject& Scene::addModel(const std::string& t_name, const std::string& t_file, const std::shared_ptr<gl::Program>& t_program, const bool t_hasMipmap)
{
	Importer importer{ t_file };
	GameObject& model{ importer.importModel(*this, t_program, t_hasMipmap) };
	model.setName(t_name);

	return model;
}

GameObject& Scene::addTerrain(const std::shared_ptr<Terrain>& t_terrain, const std::shared_ptr<CustomMaterial>& t_material)
{
	return addTerrain(std::string{ "Terrain" }, t_terrain, t_material);
}

GameObject& Scene::addTerrain(const std::string& t_name, const std::shared_ptr<Terrain>& t_terrain, const std::shared_ptr<CustomMaterial>& t_material)
{
	GameObject& gameObject{ addGameObject(t_name) };
	gameObject.addComponent<TerrainRenderer>(t_terrain, t_material);

	return gameObject;
}

void Scene::removeGameObject(const int t_index)
{
	GameObject& gameObject{ getGameObject(t_index) };
	Transform& transform{ gameObject.getTransform() };

	for (int i{ 0 }; i < transform.getChildCount(); ++i)
	{
		removeGameObject(transform.getChild(i).getGameObject());
	}

	m_registry.destroy(gameObject.m_handle);
	m_gameObjects.erase(m_gameObjects.begin() + t_index);
}

void Scene::removeGameObject(const std::string& t_name)
{
	auto iterator{ m_gameObjects.begin() };

	while (iterator != m_gameObjects.end())
	{
		if ((*iterator)->getName() == t_name)
		{
			break;
		}

		++iterator;
	}

	GameObject& gameObject{ **iterator };
	Transform& transform{ gameObject.getTransform() };

	for (int i{ 0 }; i < transform.getChildCount(); ++i)
	{
		removeGameObject(transform.getChild(i).getGameObject());
	}

	m_registry.destroy(gameObject.m_handle);
	m_gameObjects.erase(iterator);
}

void Scene::removeGameObject(GameObject& t_gameObject)
{
	auto iterator{ m_gameObjects.begin() };
	GameObject* gameObject{ &t_gameObject };

	while (iterator != m_gameObjects.end())
	{
		if (iterator->get() == gameObject)
		{
			break;
		}

		++iterator;
	}

	Transform& transform{ gameObject->getTransform() };

	for (int i{ 0 }; i < transform.getChildCount(); ++i)
	{
		removeGameObject(transform.getChild(i).getGameObject());
	}

	m_registry.destroy(gameObject->m_handle);
	m_gameObjects.erase(iterator);
}

// Setters
void Scene::setName(const std::string& t_name)
{
	m_name = t_name;
}

void Scene::setMainCamera(Camera* const t_mainCamera)
{
	m_mainCamera = t_mainCamera;
}

// Getters
const std::string& Scene::getName() const
{
	return m_name;
}

const Environment& Scene::getEnvironment() const
{
	return m_environment;
}

Environment& Scene::getEnvironment()
{
	return m_environment;
}

const GameObject& Scene::getGameObject(const int t_index) const
{
	STHE_ASSERT(t_index >= 0 && t_index < getGameObjectCount(), "Index must refer to an existing game object");

	return *m_gameObjects[t_index];
}

GameObject& Scene::getGameObject(const int t_index)
{
	STHE_ASSERT(t_index >= 0 && t_index < getGameObjectCount(), "Index must refer to an existing game object");

	return *m_gameObjects[t_index];
}

const GameObject* Scene::getGameObject(const std::string& t_name) const
{
	for (const std::unique_ptr<GameObject>& gameObject : m_gameObjects)
	{
		if (gameObject->getName() == t_name)
		{
			return gameObject.get();
		}
	}

	return nullptr;
}

GameObject* Scene::getGameObject(const std::string& t_name)
{
	for (const std::unique_ptr<GameObject>& gameObject : m_gameObjects)
	{
		if (gameObject->getName() == t_name)
		{
			return gameObject.get();
		}
	}

	return nullptr;
}

int Scene::getGameObjectCount() const
{
	return static_cast<int>(m_gameObjects.size());
}

std::vector<const GameObject*> Scene::getRoots() const
{
	std::vector<const GameObject*> roots;

	for (const std::unique_ptr<GameObject>& gameObject : m_gameObjects)
	{
		if (!gameObject->getTransform().hasParent())
		{
			roots.emplace_back(gameObject.get());
		}
	}

	return roots;
}

std::vector<GameObject*> Scene::getRoots()
{
	std::vector<GameObject*> roots;

	for (const std::unique_ptr<GameObject>& gameObject : m_gameObjects)
	{
		if (!gameObject->getTransform().hasParent())
		{
			roots.emplace_back(gameObject.get());
		}
	}

	return roots;
}

const Camera* Scene::getMainCamera() const
{
	return m_mainCamera;
}

Camera* Scene::getMainCamera()
{
	return m_mainCamera;
}

bool Scene::hasMainCamera() const
{
	return m_mainCamera != nullptr;
}

bool Scene::isActive() const
{
	return this == getApplication().getActiveScene();
}

}
