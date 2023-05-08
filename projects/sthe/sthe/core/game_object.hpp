#pragma once

#include <entt/entt.hpp>
#include <string>
#include <vector>

namespace sthe
{

class Scene;
class Transform;

class GameObject
{
public:
	// Constructors
	explicit GameObject(const std::string& t_name);
	GameObject(const GameObject& t_gameObject) = delete;
	GameObject(GameObject&& t_gameObject) = default;

	//Destructor
	~GameObject() = default;

	// Operator
	GameObject& operator=(const GameObject& t_gameObject) = delete;
	GameObject& operator=(GameObject&& t_gameObject) = default;

	// Functionality
	template<typename TComponent, typename... TArgs>
	TComponent& addComponent(TArgs&&... t_args);

	template<typename TComponent>
	void removeComponent();

	// Setter
	void setName(const std::string& t_name);

	// Getters
	const std::string& getName() const;
	const Scene& getScene() const;
	Scene& getScene();
	const Transform& getTransform() const;
	Transform& getTransform();

	template<typename TComponent>
	const TComponent* getComponent() const;

	template<typename TComponent>
	TComponent* getComponent();

	template<typename TComponent>
	const TComponent* getComponentInParent() const;

	template<typename TComponent>
	TComponent* getComponentInParent();

	template<typename TComponent>
	std::vector<const TComponent*> getComponentsInParent() const;

	template<typename TComponent>
	std::vector<TComponent*> getComponentsInParent();

	template<typename TComponent>
	const TComponent* getComponentInChildren() const;

	template<typename TComponent>
	TComponent* getComponentInChildren();

	template<typename TComponent>
	std::vector<const TComponent*> getComponentsInChildren() const;

	template<typename TComponent>
	std::vector<TComponent*> getComponentsInChildren();

	template<typename TComponent>
	bool hasComponent() const;
private:
	// Functionality
	template<typename TComponent>
	void disconnectComponent(entt::registry& t_registry, const entt::entity t_entity);

	// Getters
	template<typename TComponent>
	void getComponentsInChildren(std::vector<const TComponent*>& t_components) const;

	template<typename TComponent>
	void getComponentsInChildren(std::vector<TComponent*>& t_components);

	// Attributes
	entt::entity m_handle;
	std::string m_name;
	Scene* m_scene;
	Transform* m_transform;

	// Friend
	friend class Scene;
};

}
