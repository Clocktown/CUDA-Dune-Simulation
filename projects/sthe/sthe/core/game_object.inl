#include "game_object.hpp"
#include "scene.hpp"
#include "component.hpp"
#include "event.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/components/transform.hpp>
#include <entt/entt.hpp>
#include <cassert>
#include <type_traits>
#include <vector>

namespace sthe
{

// Functionality
template<typename TComponent, typename ...TArgs>
inline TComponent& GameObject::addComponent(TArgs&&... t_args)
{
	static_assert(std::derived_from<TComponent, Component>, "TComponent must derive from Component");
	STHE_ASSERT(m_scene != nullptr, "Game object was not created by a scene");

	entt::registry& registry{ m_scene->m_registry };
	entt::dispatcher& dispatcher{ m_scene->m_dispatcher };

	TComponent& component{ registry.emplace_or_replace<TComponent>(m_handle, std::forward<TArgs>(t_args)...) };
	component.m_gameObject = this;

	constexpr bool hasAwakeFunction{ requires(TComponent& t_component) { t_component.awake(); } };
	constexpr bool hasStartFunction{ requires(TComponent& t_component) { t_component.start(); } };
	constexpr bool hasUpdateFunction{ requires(TComponent& t_component) { t_component.update(); } };
	constexpr bool hasLateUpdateFunction{ requires(TComponent& t_component) { t_component.lateUpdate(); } };
	constexpr bool hasOnPreRenderFunction{ requires(TComponent& t_component) { t_component.onPreRender(); } };
	constexpr bool hasOnRenderFunction{ requires(TComponent& t_component) { t_component.onRender(); } };
	constexpr bool hasOnPostRenderFunction{ requires(TComponent& t_component) { t_component.onPostRender(); } };
	constexpr bool hasOnGUIFunction{ requires(TComponent & t_component) { t_component.onGUI(); } };
	constexpr bool isScript = hasAwakeFunction || hasStartFunction || hasUpdateFunction || hasLateUpdateFunction ||
						      hasOnPreRenderFunction || hasOnRenderFunction || hasOnPostRenderFunction || hasOnGUIFunction;

	if constexpr (hasAwakeFunction)
	{
		dispatcher.sink<Event::Awake>().connect<&TComponent::awake>(component);
		
		if (m_scene->isActive())
		{
			component.awake();
		}
	}

	if constexpr (hasStartFunction)
	{
		dispatcher.sink<Event::Start>().connect<&TComponent::start>(component);
	}
	
	if constexpr (hasUpdateFunction)
	{
		dispatcher.sink<Event::Update>().connect<&TComponent::update>(component);
	}

	if constexpr (hasLateUpdateFunction)
	{
		dispatcher.sink<Event::LateUpdate>().connect<&TComponent::lateUpdate>(component);
	}

	if constexpr (hasOnPreRenderFunction)
	{
		dispatcher.sink<Event::OnPreRender>().connect<&TComponent::onPreRender>(component);
	}

	if constexpr (hasOnRenderFunction)
	{
		dispatcher.sink<Event::OnRender>().connect<&TComponent::onRender>(component);
	}

	if constexpr (hasOnPostRenderFunction)
	{
		dispatcher.sink<Event::OnPostRender>().connect<&TComponent::onPostRender>(component);
	}

	if constexpr (hasOnGUIFunction)
	{
		dispatcher.sink<Event::OnGUI>().connect<&TComponent::onGUI>(component);
	}

	if constexpr (isScript)
	{
		registry.on_destroy<TComponent>().connect<&GameObject::disconnectComponent<TComponent>>(*this);
	}

	return component;
}

template<typename TComponent>
inline void GameObject::removeComponent()
{
	static_assert(!std::is_same<TComponent, Transform>::value, "Transform cannot be removed");
	STHE_ASSERT(m_scene != nullptr, "Game object was not created by a scene");

	entt::registry& registry{ m_scene->m_registry };
	registry.erase<TComponent>(m_handle);
}

template<typename TComponent>
inline void GameObject::disconnectComponent(entt::registry& t_registry, const entt::entity t_entity)
{
	TComponent* const component{ t_registry.try_get<TComponent>(t_entity) };

	if (component != nullptr)
	{
		entt::dispatcher& dispatcher{ m_scene->m_dispatcher };
		dispatcher.disconnect(*component);
	}
}

// Getters
template<typename TComponent>
inline const TComponent* GameObject::getComponent() const
{
	STHE_ASSERT(m_scene != nullptr, "Game object was not created by a scene");

	return m_scene->m_registry.try_get<TComponent>(m_handle);
}

template<typename TComponent>
inline TComponent* GameObject::getComponent()
{
	STHE_ASSERT(m_scene != nullptr, "Game object was not created by a scene");

	return m_scene->m_registry.try_get<TComponent>(m_handle);
}

template<typename TComponent>
inline const TComponent* GameObject::getComponentInParent() const
{
	const TComponent* const component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		return component;
	}

	const Transform& transform{ getTransform() };
	
	if (transform.hasParent())
	{
		return transform.getParent()->getGameObject().getComponentInParent<TComponent>();
	}

	return nullptr;
}

template<typename TComponent>
inline TComponent* GameObject::getComponentInParent()
{
	TComponent* const component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		return component;
	}

	Transform& transform{ getTransform() };

	if (transform.hasParent())
	{
		return transform.getParent()->getGameObject().getComponentInParent<TComponent>();
	}

	return nullptr;
}

template<typename TComponent>
inline std::vector<const TComponent*> GameObject::getComponentsInParent() const
{
	std::vector<const TComponent*> components;
	const TComponent* component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		components.emplace_back(component);
	}

	const Transform* parent{ getTransform().getParent() };

	while (parent != nullptr)
	{
		component = parent->getGameObject().getComponent<TComponent>();

		if (component != nullptr)
		{
			components.emplace_back(component);
		}

		parent = parent->getParent();
	}

	return components;
}

template<typename TComponent>
inline std::vector<TComponent*> GameObject::getComponentsInParent()
{
	std::vector<TComponent*> components;
	TComponent* component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		components.emplace_back(component);
	}

	Transform* parent{ getTransform().getParent() };

	while (parent != nullptr)
	{
	    component = parent->getGameObject().getComponent<TComponent>();

		if (component != nullptr)
		{
			components.emplace_back(component);
		}

		parent = parent->getParent();
	}

	return components;
}

template<typename TComponent>
inline const TComponent* GameObject::getComponentInChildren() const
{
	const TComponent* component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		return component;
	}

	const Transform& transform{ getTransform() };

	for (int i{ 0 }; i < transform.getChildCount(); ++i)
	{
		component = transform.getChild(i).getGameObject().getComponentInChildren<TComponent>();

		if (component != nullptr)
		{
			return component;
		}
	}

	return nullptr;
}

template<typename TComponent>
inline TComponent* GameObject::getComponentInChildren()
{
	TComponent* component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		return component;
	}

	Transform& transform{ getTransform() };

	for (int i{ 0 }; i < transform.getChildCount(); ++i)
	{
		component = transform.getChild(i).getGameObject().getComponentInChildren<TComponent>();

		if (component != nullptr)
		{
			return component;
		}
	}

	return nullptr;
}

template<typename TComponent>
inline std::vector<const TComponent*> GameObject::getComponentsInChildren() const
{
	std::vector<const TComponent*> components;
	getComponentsInChildren(components);

	return components;
}

template<typename TComponent>
inline std::vector<TComponent*> GameObject::getComponentsInChildren()
{
	std::vector<TComponent*> components;
	getComponentsInChildren(components);

	return components;
}

template<typename TComponent>
inline void GameObject::getComponentsInChildren(std::vector<const TComponent*>& t_components) const
{
	const TComponent* const component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		t_components.emplace_back(component);
	}

	const Transform& transform{ getTransform() };

	for (int i{ 0 }; i < transform.getChildCount(); ++i)
	{
		transform.getChild(i).getGameObject().getComponentsInChildren(t_components);
	}
}

template<typename TComponent>
inline void GameObject::getComponentsInChildren(std::vector<TComponent*>& t_components)
{
	TComponent* const component{ getComponent<TComponent>() };

	if (component != nullptr)
	{
		t_components.emplace_back(component);
	}

	Transform& transform{ getTransform() };

	for (int i{ 0 }; i < transform.getChildCount(); ++i)
	{
		transform.getChild(i).getGameObject().getComponentsInChildren(t_components);
	}
}

template<typename TComponent>
inline bool GameObject::hasComponent() const
{
	STHE_ASSERT(m_scene != nullptr, "Game object was not created by a scene");

	return m_scene->m_registry.any_of<TComponent>(m_handle);
}

}
