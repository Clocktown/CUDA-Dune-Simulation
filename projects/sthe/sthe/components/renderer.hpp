#pragma once

#include <sthe/core/component.hpp>

namespace sthe
{

class Renderer : public Component
{
public:
	// Constructors
	Renderer();
	Renderer(const Renderer& t_renderer) = delete;
	Renderer(Renderer&& t_renderer) = default;

	// Destructor
	virtual ~Renderer() = 0;

	// Operators
	Renderer& operator=(const Renderer& t_renderer) = delete;
	Renderer& operator=(Renderer&& t_renderer) = default;

	// Setters
	void setBaseInstance(const int t_baseInstance);
	void setInstanceCount(const int t_instanceCount);

	// Getters
	int getBaseInstance() const;
	int getInstanceCount() const;
private:
	// Attributes
	int m_baseInstance;
	int m_instanceCount;
};

}
