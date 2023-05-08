#include "renderer.hpp"

namespace sthe
{

// Constructor
Renderer::Renderer() :
	m_baseInstance{ 0 },
	m_instanceCount{ 1 }
{

}

// Destructor
Renderer::~Renderer()
{

}

// Setters
void Renderer::setBaseInstance(const int t_baseInstance)
{
	m_baseInstance = t_baseInstance;
}

void Renderer::setInstanceCount(const int t_instanceCount)
{
	m_instanceCount = t_instanceCount;
}

// Getters
int Renderer::getBaseInstance() const
{
	return m_baseInstance;
}

int Renderer::getInstanceCount() const
{
	return m_instanceCount;
}

}
