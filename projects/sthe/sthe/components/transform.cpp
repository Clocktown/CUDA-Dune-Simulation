#include "transform.hpp"
#include <sthe/config/debug.hpp>
#include <sthe/core/world.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace sthe
{

// Constructor
Transform::Transform() :
	m_localPosition{ 0.0f },
	m_localRotation{ 1.0f, 0.0f, 0.0f, 0.0f },
	m_localScale{ 1.0f },
	m_rotation{ 1.0f, 0.0f, 0.0f, 0.0f },
	m_modelMatrix{ 1.0f },
	m_inverseModelMatrix{ 1.0f },
	m_hasChanged{ false },
	m_parent{ nullptr }
{

}

// Functionality
void Transform::translate(const glm::vec3& t_translation, const Space t_space)
{
	if (t_space == Space::Local)
	{
		setLocalPosition(m_localPosition + transformDirection(t_translation));
	}
	else
	{
		setLocalPosition(m_localPosition + t_translation);
	}
}

void Transform::rotate(const glm::vec3& t_eulerAngles, const Space t_space)
{
	const glm::quat rotation{ glm::radians(t_eulerAngles) };

	if (t_space == Space::Local)
	{
		update();
		const glm::quat rotationX{ glm::angleAxis(glm::radians(t_eulerAngles.x), reinterpret_cast<glm::vec3&>(m_modelMatrix[0])) };
		const glm::quat rotationY{ glm::angleAxis(glm::radians(t_eulerAngles.y), reinterpret_cast<glm::vec3&>(m_modelMatrix[1])) };
		const glm::quat rotationZ{ glm::angleAxis(glm::radians(t_eulerAngles.z), reinterpret_cast<glm::vec3&>(m_modelMatrix[2])) };
		setLocalRotation(rotationX * rotationY * rotationZ * m_localRotation);
	}
	else
	{
		setLocalRotation(rotation * m_localRotation);
	}
}

void Transform::rotate(const float t_angle, const glm::vec3& t_axis, const Space t_space)
{
	if (t_space == Space::Local)
	{
		setLocalRotation(glm::angleAxis(glm::radians(t_angle), transformDirection(t_axis)) * m_localRotation);
	}
	else
	{
		setLocalRotation(glm::angleAxis(glm::radians(t_angle), t_axis) * m_localRotation);
	}
}

void Transform::lookAt(const Transform& t_target, const glm::vec3& t_worldUp)
{
	lookAt(t_target.getPosition(), t_worldUp);
}

void Transform::lookAt(const glm::vec3& t_target, const glm::vec3& t_worldUp)
{
	const glm::vec3& position{ getPosition() };

	if (position != t_target)
	{
		setRotation(glm::quatLookAt(glm::normalize(t_target - position), t_worldUp));
	}
}
 
glm::vec3 Transform::transformPoint(const glm::vec3& t_point) const
{
	return getModelMatrix() * glm::vec4{ t_point, 1.0f };
}

glm::vec3 Transform::transformDirection(const glm::vec3& t_direction) const
{
	return getRotation() * t_direction;
}

glm::vec3 Transform::inverseTransformPoint(const glm::vec3& t_point) const
{
	return getInverseModelMatrix() * glm::vec4{ t_point, 1.0f };
}

glm::vec3 Transform::inverseTransformDirection(const glm::vec3& t_direction) const
{
	return glm::conjugate(getRotation()) * t_direction;
}

void Transform::update() const
{
	if (m_hasChanged)
	{
		m_modelMatrix = glm::scale(glm::mat4_cast(m_localRotation), m_localScale);
		m_modelMatrix[3] = glm::vec4{ m_localPosition, 1.0f };

		m_inverseModelMatrix = glm::translate(glm::mat4_cast(glm::conjugate(m_localRotation)), -m_localPosition);

		for (int i{ 0 }; i < 3; ++i)
		{
			m_inverseModelMatrix[i].x /= m_localScale.x;
			m_inverseModelMatrix[i].y /= m_localScale.y;
			m_inverseModelMatrix[i].z /= m_localScale.z;
		}
		
		if (hasParent())
		{
			m_rotation = m_parent->getRotation() * m_localRotation;
			m_modelMatrix = m_parent->getModelMatrix() * m_modelMatrix;
			m_inverseModelMatrix *= m_parent->getInverseModelMatrix();
		}
		else
		{
			m_rotation = m_localRotation;
		}

		m_rotation = glm::normalize(m_rotation);
		m_hasChanged = false;
	}
}

void Transform::propergate()
{
	m_hasChanged = true;

	for (Transform* const transform : m_children)
	{
		transform->propergate();
	}
}

void Transform::detachChildren()
{
	for (Transform* const transform : m_children)
	{
		transform->setParent(nullptr);
	}

	m_children.clear();
}

// Setters
void Transform::setLocalPosition(const glm::vec3& t_localPosition)
{
	m_localPosition = t_localPosition;
	propergate();
}

void Transform::setLocalEulerAngles(const glm::vec3& t_localEulerAngles)
{
	setLocalRotation(glm::quat{ glm::radians(t_localEulerAngles) });
}

void Transform::setLocalRotation(const glm::quat& t_localRotation)
{
	m_localRotation = t_localRotation;
	propergate();
}

void Transform::setLocalScale(const float t_localScale)
{
	setLocalScale(glm::vec3{ t_localScale });
}

void Transform::setLocalScale(const glm::vec3& t_localScale)
{
	STHE_ASSERT(t_localScale.x != 0.0f, "Local scale x cannot be equal to 0");
	STHE_ASSERT(t_localScale.y != 0.0f, "Local scale y cannot be equal to 0");
	STHE_ASSERT(t_localScale.z != 0.0f, "Local scale z cannot be equal to 0");

	m_localScale = t_localScale;
	propergate();
}

void Transform::setLocalPositionAndRotation(const glm::vec3& t_localPosition, const glm::quat& t_localRotation)
{
	m_localPosition = t_localPosition;
	m_localRotation = t_localRotation;
	propergate();
}

void Transform::setPosition(const glm::vec3& t_position)
{
	if (hasParent())
	{
		m_localPosition = m_parent->inverseTransformPoint(t_position);
	}
	else
	{
		m_localPosition = t_position;
	}

	propergate();
}

void Transform::setEulerAngles(const glm::vec3& t_eulerAngles)
{
	setRotation(glm::quat{ glm::radians(t_eulerAngles) });
}

void Transform::setRotation(const glm::quat& t_rotation)
{
	if (hasParent())
	{
		m_localRotation = t_rotation * glm::conjugate(m_parent->getRotation());
	}
	else
	{
		m_localRotation = t_rotation;
	}

	propergate();
}

void Transform::setPositionAndRotation(const glm::vec3& t_position, const glm::quat& t_rotation)
{
	if (hasParent())
	{
		m_localPosition = m_parent->inverseTransformPoint(t_position);
		m_localRotation = t_rotation * glm::conjugate(m_parent->getRotation());
	}
	else
	{
		m_localPosition = t_position;
		m_localRotation = t_rotation;
	}

	propergate();
}

void Transform::setParent(Transform* const t_parent, const bool t_keepTransform)
{
	STHE_ASSERT(t_parent != nullptr && (t_parent != this), "Parent cannot be a self-reference");
	STHE_ASSERT(t_parent != nullptr && &t_parent->getScene() == &getScene(), "Parent must be in the same scene as this");
	STHE_ASSERT(t_parent != nullptr && !t_parent->isChildOf(*this), "Parent cannot be a child of this");

	const glm::vec3 position{ getPosition() };
	const glm::quat rotation{ m_rotation };

	if (hasParent())
	{
		m_parent->m_children.erase(std::find(m_parent->m_children.begin(), m_parent->m_children.end(), this));
	}

	if (t_parent != nullptr)
	{
		t_parent->m_children.emplace_back(this);
	}

	m_parent = t_parent;

	if (t_keepTransform)
	{
		setPositionAndRotation(position, rotation);
	}
	else
	{
		propergate();
	}
}

// Getters
const glm::vec3& Transform::getLocalPosition() const
{
	return m_localPosition;
}

glm::vec3 Transform::getLocalEulerAngles() const
{
	return glm::degrees(glm::eulerAngles(m_localRotation));
}

const glm::quat& Transform::getLocalRotation() const
{
	return m_localRotation;
}

const glm::vec3& Transform::getLocalScale() const
{
	return m_localScale;
}

const glm::vec3& Transform::getPosition() const
{
	update();
	return reinterpret_cast<glm::vec3&>(m_modelMatrix[3]);
}

glm::vec3 Transform::getEulerAngles() const
{
	return glm::degrees(glm::eulerAngles(getRotation()));
}

const glm::quat& Transform::getRotation() const
{
	update();
	return m_rotation;
}

const glm::mat4& Transform::getModelMatrix() const
{
	update();
	return m_modelMatrix;
}

const glm::mat4& Transform::getInverseModelMatrix() const
{
	update();
	return m_inverseModelMatrix;
}

const glm::vec3& Transform::getRight() const
{
	update();
	return reinterpret_cast<glm::vec3&>(m_modelMatrix[0]);
}

const glm::vec3& Transform::getUp() const
{
	update();
	return reinterpret_cast<glm::vec3&>(m_modelMatrix[1]);
}

glm::vec3 Transform::getForward() const
{
	update();
	return -m_modelMatrix[2];
}

const Transform* Transform::getParent() const
{
	return m_parent;
}

Transform* Transform::getParent()
{
	return m_parent;
}

const Transform& Transform::getChild(const int t_index) const
{
	STHE_ASSERT(t_index >= 0 && t_index < getChildCount(), "Index must refer to an existing child");

	return *m_children[t_index];
}

Transform& Transform::getChild(const int t_index)
{
	STHE_ASSERT(t_index >= 0 && t_index < getChildCount(), "Index must refer to an existing child");

	return *m_children[t_index];
}

int Transform::getChildCount() const
{
	return static_cast<int>(m_children.size());
}

bool Transform::hasParent() const
{
	return m_parent != nullptr;
}

bool Transform::hasChildren() const
{
	return !m_children.empty();
}

bool Transform::isChildOf(const Transform& t_parent) const
{
	const Transform* parent{ this };

	do
	{
		if (parent == &t_parent)
		{
			return true;
		}

		parent = parent->getParent();
	} 
	while (parent != nullptr);

	return false;
}

}
