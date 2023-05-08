#version 460 core

// Input
layout(location = 0) in vec4 t_position;
layout(location = 1) in vec4 t_normal;
layout(location = 2) in vec4 t_tangent;
layout(location = 3) in vec2 t_uv;

struct Light
{
	vec3 position;
	unsigned int type;
	vec3 color;
	float intensity;
	vec3 attenuation;
	float range;
	vec3 direction;
	float spotOuterCutOff;
	float spotInnerCutOff;
	int pad1, pad2, pad3;
};

struct Environment
{
	vec3 ambientColor;
	float ambientIntensity;
	vec3 fogColor;
	float fogDensity;
	unsigned int fogMode;
	float fogStart;
	float fogEnd;
	int lightCount;
	Light lights[16];
};

struct Material
{
	vec3 diffuseColor;
	float opacity;
	vec3 specularColor;
	float specularIntensity;
	float shininess;
	bool hasDiffuseMap;
	bool hasNormalMap;
	int pad;
};

layout(std140, binding = 0) uniform PipelineBuffer
{
	float t_time;
	float t_deltaTime;
	ivec2 t_resolution;
	mat4 t_projectionMatrix;
	mat4 t_viewMatrix;
	mat4 t_inverseViewMatrix;
	mat4 t_viewProjectionMatrix;
	Environment t_environment;
	mat4 t_modelMatrix;
	mat4 t_inverseModelMatrix;
	mat4 t_modelViewMatrix;
	mat4 t_inverseModelViewMatrix;
	Material t_material;
};

// Output
struct Fragment
{
	vec3 position;
	vec3 normal;
	vec2 uv;
};

out Fragment fragment;
flat out mat3 tbnMatrix;

// Functionality
void main()
{
	const mat3 normalMatrix = mat3(t_modelMatrix);

	fragment.position = (t_modelMatrix * t_position).xyz;
	fragment.normal = normalize(normalMatrix * t_normal.xyz);
	fragment.uv = t_uv;

	if (t_material.hasNormalMap)
	{
		const vec3 tangent = normalize(normalMatrix * t_tangent.xyz);
		const vec3 bitangent = cross(fragment.normal, tangent);

		tbnMatrix = mat3(tangent, bitangent, fragment.normal);
	}

	gl_Position = t_viewProjectionMatrix * vec4(fragment.position, 1.0f);
}
