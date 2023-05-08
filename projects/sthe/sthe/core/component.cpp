#include "component.hpp"
#include "game_object.hpp"
#include <sthe/config/debug.hpp>

namespace sthe
{

// Constructor
Component::Component() :
    m_gameObject{ nullptr }
{

}

// Destructor
Component::~Component() 
{

}

// Getters
const Scene& Component::getScene() const
{
    return getGameObject().getScene();
}

Scene& Component::getScene()
{
    return getGameObject().getScene();
}

const GameObject& Component::getGameObject() const
{
    STHE_ASSERT(m_gameObject != nullptr, "Component was not created by a game object");

    return *m_gameObject;
}

GameObject& Component::getGameObject()
{
    STHE_ASSERT(m_gameObject != nullptr, "Component was not created by a game object");

    return *m_gameObject;
}

}
