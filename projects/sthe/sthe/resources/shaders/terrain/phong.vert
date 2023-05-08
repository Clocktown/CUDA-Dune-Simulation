#version 460 core

// Input
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

struct TerrainLayer
{
	vec3 diffuseColor;
    float specularIntensity;
	vec3 specularColor;
	float shininess;
	bool hasDiffuseMap;
	int pad1, pad2, pad3;
};

struct Terrain
{
    vec3 size;
	int subDivision;
	int detail;
	int layerCount;
	bool hasHeightMap;
	bool hasAlphaMap;
	TerrainLayer layers[4];
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
	Terrain t_terrain;
};

// Output
out vec2 tescUV;

void main()
{
    const vec2 position = vec2(gl_InstanceID / t_terrain.subDivision, gl_InstanceID % t_terrain.subDivision);
	const vec2 offset = vec2(gl_VertexID / 2, gl_VertexID % 2);
	const float scale = 1.0f / float(t_terrain.subDivision);

	tescUV = scale * (position + offset);
	gl_Position = vec4((tescUV.x - 0.5f) * t_terrain.size.x, 0.0f, (tescUV.y - 0.5f) * t_terrain.size.z, 1.0f);
}
