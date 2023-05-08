#pragma once

#include <sthe/core/component.hpp>
#include <sthe/core/world.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace sthe
{

class Transform : public Component
{
public:
	// Constructors
	Transform();
	Transform(const Transform& t_transform) = delete;
	Transform(Transform&& t_transform) = default;

	// Destructor
	~Transform() = default;

	// Operators
	Transform& operator=(const Transform& t_transform) = delete;
	Transform& operator=(Transform&& t_transform) = default;

	// Functionality
	void translate(const glm::vec3& t_translation, const Space t_space = Space::Local);
	void rotate(const glm::vec3& t_eulerAngles, const Space t_space = Space::Local);
	void rotate(const float t_angle, const glm::vec3& t_axis, const Space t_space = Space::Local);
	void lookAt(const Transform& t_target, const glm::vec3& t_worldUp = World::up);
	void lookAt(const glm::vec3& t_target, const glm::vec3& t_worldUp = World::up);
	glm::vec3 transformPoint(const glm::vec3& t_point) const;
	glm::vec3 transformDirection(const glm::vec3& t_direction) const;
	glm::vec3 inverseTransformPoint(const glm::vec3& t_point) const;
	glm::vec3 inverseTransformDirection(const glm::vec3& t_direction) const;
	void detachChildren();

	// Setters
	void setLocalPosition(const glm::vec3& t_localPosition);
	void setLocalEulerAngles(const glm::vec3& t_localEulerAngles);
	void setLocalRotation(const glm::quat& t_localRotation);
	void setLocalScale(const float t_localScale);
	void setLocalScale(const glm::vec3& t_localScale);
	void setLocalPositionAndRotation(const glm::vec3& t_localPosition, const glm::quat& t_localRotation);
	void setPosition(const glm::vec3& t_position);
	void setEulerAngles(const glm::vec3& t_eulerAngles);
	void setRotation(const glm::quat& t_rotation);
	void setPositionAndRotation(const glm::vec3& t_position, const glm::quat& t_rotation);
	void setParent(Transform* const t_parent, const bool t_keepTransform = true);

	// Getters
	const glm::vec3& getLocalPosition() const;
	glm::vec3 getLocalEulerAngles() const;
	const glm::quat& getLocalRotation() const;
	const glm::vec3& getLocalScale() const;
	const glm::vec3& getPosition() const;
	glm::vec3 getEulerAngles() const;
	const glm::quat& getRotation() const;
	const glm::mat4& getModelMatrix() const;
	const glm::mat4& getInverseModelMatrix() const;
	const glm::vec3& getRight() const;
	const glm::vec3& getUp() const;
	glm::vec3 getForward() const;
	const Transform* getParent() const;
	Transform* getParent();
	const Transform& getChild(const int t_index) const;
	Transform& getChild(const int t_index);
	int getChildCount() const;
	bool hasParent() const;
	bool hasChildren() const;
	bool isChildOf(const Transform& t_parent) const;
private:
	// Functionality
	void update() const;
	void propergate();

	// Attributes
	glm::vec3 m_localPosition;
	glm::quat m_localRotation;
	glm::vec3 m_localScale;
	mutable glm::quat m_rotation;
	mutable glm::mat4 m_modelMatrix;
	mutable glm::mat4 m_inverseModelMatrix;
	mutable bool m_hasChanged;

	Transform* m_parent;
	std::vector<Transform*> m_children;

	// Friend
	friend class Scene;
};

}
