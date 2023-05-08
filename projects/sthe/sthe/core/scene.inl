#include "scene.hpp"
#include <entt/entt.hpp>

namespace sthe
{

// Getters
template<typename TComponent, typename... TOthers, typename... TExcludes>
inline entt::view<entt::get_t<const TComponent, const TOthers...>, entt::exclude_t<TExcludes...>> Scene::getComponents(entt::exclude_t<TExcludes...> t_excludes) const
{
	return m_registry.view<TComponent, TOthers..., TExcludes...>(t_excludes);
}

template<typename TComponent, typename... TOthers, typename... TExcludes>
inline entt::view<entt::get_t<TComponent, TOthers...>, entt::exclude_t<TExcludes...>> Scene::getComponents(entt::exclude_t<TExcludes...> t_excludes)
{
	return m_registry.view<TComponent, TOthers..., TExcludes...>(t_excludes);
}

}
