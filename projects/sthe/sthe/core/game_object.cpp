#include "game_object.hpp"
#include "scene.hpp"
#include <sthe/config/debug.hpp>
#include <entt/entt.hpp>
#include <string>

namespace sthe
{

// Constructor
GameObject::GameObject(const std::string& t_name) :
	m_handle{ entt::null },
	m_name{ t_name },
	m_scene{ nullptr },
	m_transform{ nullptr }
{
	
}

// Setter
void GameObject::setName(const std::string& t_name)
{
	m_name = t_name;
}

// Getters
const std::string& GameObject::getName() const
{
	return m_name;
}

const Scene& GameObject::getScene() const
{
	STHE_ASSERT(m_scene != nullptr, "Game Object must be constructed through a scene");

	return *m_scene;
}

Scene& GameObject::getScene()
{
	STHE_ASSERT(m_scene != nullptr, "Game Object must be constructed through a scene");

	return *m_scene;
}

const Transform& GameObject::getTransform() const
{
	STHE_ASSERT(m_transform != nullptr, "Game Object must be constructed through a scene");

	return *m_transform;
}

Transform& GameObject::getTransform()
{
	STHE_ASSERT(m_transform != nullptr, "Game Object must be constructed through a scene");

	return *m_transform;
}

}
