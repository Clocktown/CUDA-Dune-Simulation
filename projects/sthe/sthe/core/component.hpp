#pragma once

#include "game_object.hpp"

namespace sthe
{

class Scene;

class Component
{
public:
	// Static
	static constexpr bool in_place_delete = true; // Enables pointer stability in EnTT

	// Constructors
	Component();
	Component(const Component& t_component) = delete;
	Component(Component&& t_component) = default;

	// Destructor
	virtual ~Component() = 0;

	// Operators
	Component& operator=(const Component& t_component) = delete;
	Component& operator=(Component&& t_component) = default;

	// Getters
	const Scene& getScene() const;
	Scene& getScene();
	const GameObject& getGameObject() const;
	GameObject& getGameObject();
private:
	// Attribute
	GameObject* m_gameObject;

	// Friends
	friend class GameObject;
};

}
