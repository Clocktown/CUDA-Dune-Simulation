#pragma once

#include "environment.hpp"
#include "game_object.hpp"
#include "mesh.hpp"
#include "terrain.hpp"
#include "material.hpp"
#include "custom_material.hpp"
#include <sthe/components/camera.hpp>
#include <sthe/components/light.hpp>
#include <sthe/gl/program.hpp>
#include <entt/entt.hpp>
#include <memory>
#include <string>
#include <vector>

namespace sthe
{

class Scene
{
public:
	// Constructors
	explicit Scene(const std::string& t_name);
	Scene(const Scene& t_scene) = delete;
	Scene(Scene&& t_scene) = default;

	// Destructor
	~Scene() = default;

	// Operators
	Scene& operator=(const Scene& t_scene) = delete;
	Scene& operator=(Scene&& t_scene) = default;

	// Functionality
	GameObject& addGameObject(const std::string& t_name = std::string{ "Game Object"});
	GameObject& addCamera(const std::string& t_name = std::string{ "Camera" });
	GameObject& addTrackball(const std::string& t_name = std::string{ "Trackball" });
	GameObject& addPointLight(const std::string& t_name = std::string{ "Point Light" });
	GameObject& addSpotLight(const std::string& t_name = std::string{ "Spot Light" });
	GameObject& addDirectionalLight(const std::string& t_name = std::string{ "Directional Light" });
	GameObject& addMesh(const std::shared_ptr<Mesh>& t_mesh, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addMesh(const std::shared_ptr<Mesh>& t_mesh, const std::vector<std::shared_ptr<Material>>& t_materials);
	GameObject& addMesh(const std::string& t_name, const std::shared_ptr<Mesh>& t_mesh, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addMesh(const std::string& t_name, const std::shared_ptr<Mesh>& t_mesh, const std::vector<std::shared_ptr<Material>>& t_materials);
	GameObject& addQuad(const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addQuad(const std::string& t_name, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addPlane(const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addPlane(const std::string& t_name, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addCube(const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addCube(const std::string& t_name, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addCylinder(const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addCylinder(const std::string& t_name, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addCone(const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addCone(const std::string& t_name, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addSphere(const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addSphere(const std::string& t_name, const std::shared_ptr<Material>& t_material = std::make_shared<Material>());
	GameObject& addModel(const std::string& t_file, const std::shared_ptr<gl::Program>& t_program = nullptr, const bool t_hasMipmap = true);
	GameObject& addModel(const std::string& t_name, const std::string& t_file, const std::shared_ptr<gl::Program>& t_program = nullptr, const bool t_hasMipmap = true);
	GameObject& addTerrain(const std::shared_ptr<Terrain>& t_terrain, const std::shared_ptr<CustomMaterial>& t_material = std::make_shared<CustomMaterial>());
	GameObject& addTerrain(const std::string& t_name, const std::shared_ptr<Terrain>& t_terrain, const std::shared_ptr<CustomMaterial>& t_material = std::make_shared<CustomMaterial>());
	void removeGameObject(const int t_index);
	void removeGameObject(const std::string& t_name);
	void removeGameObject(GameObject& t_gameObject);

	// Setters
	void setName(const std::string& t_name);
	void setMainCamera(Camera* const t_mainCamera);
	
	// Getters
	const std::string& getName() const;
	const Environment& getEnvironment() const;
	Environment& getEnvironment();
	const GameObject& getGameObject(const int t_index) const;
	GameObject& getGameObject(const int t_index);
	const GameObject* getGameObject(const std::string& t_name) const;
	GameObject* getGameObject(const std::string& t_name);
	int getGameObjectCount() const;
	std::vector<const GameObject*> getRoots() const;
	std::vector<GameObject*> getRoots();
	const Camera* getMainCamera() const;
	Camera* getMainCamera();
	bool hasMainCamera() const;
	bool isActive() const;

	template<typename TComponent, typename... TOthers, typename... TExcludes>
	entt::view<entt::get_t<const TComponent, const TOthers...>, entt::exclude_t<TExcludes...>> getComponents(entt::exclude_t<TExcludes...> t_excludes = entt::exclude_t{}) const;

	template<typename TComponent, typename... TOthers, typename... TExcludes>
	entt::view<entt::get_t<TComponent, TOthers...>, entt::exclude_t<TExcludes...>> getComponents(entt::exclude_t<TExcludes...> t_excludes = entt::exclude_t{});
private:
	// Attributes
	std::string m_name;
	Environment m_environment;

	entt::registry m_registry;
	entt::dispatcher m_dispatcher;
	std::vector<std::unique_ptr<GameObject>> m_gameObjects;
	Camera* m_mainCamera;
	
	// Friends
	friend class Application;
	friend class GameObject;
};

}

#include "scene.inl"
#include "game_object.inl"
