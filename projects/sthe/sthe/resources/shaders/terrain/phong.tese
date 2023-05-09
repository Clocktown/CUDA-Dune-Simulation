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
	ivec2 gridSize;
	float gridScale;
	float heightScale;
	int tesselationLevel;
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

layout(binding = 2) uniform sampler2D t_heightMap;

layout(quads, equal_spacing, ccw) in;
in vec2 teseUV[];

// Output
out vec3 geomPosition;
out vec2 geomUV;

void main()
{
	const vec2 uv00 = teseUV[0];
	const vec2 uv01 = teseUV[1];
	const vec2 uv11 = teseUV[3];
	const vec2 uv10 = teseUV[2];
	const vec2 uv0 = mix(uv00, uv01, gl_TessCoord.x);
	const vec2 uv1 = mix(uv10, uv11, gl_TessCoord.x);
	geomUV = mix(uv0, uv1, gl_TessCoord.y);

    const vec2 position00 = gl_in[0].gl_Position.xz;
	const vec2 position01 = gl_in[1].gl_Position.xz;
	const vec2 position11 = gl_in[3].gl_Position.xz;
	const vec2 position10 = gl_in[2].gl_Position.xz;
	const vec2 position0 = mix(position00, position01, gl_TessCoord.x);
	const vec2 position1 = mix(position10, position11, gl_TessCoord.x);
	geomPosition.xz = mix(position0, position1, gl_TessCoord.y);
    geomPosition.y = t_terrain.hasHeightMap ? texture(t_heightMap, geomUV).x * t_terrain.heightScale : 0.0f;

	const vec4 position = t_modelMatrix * vec4(geomPosition, 1.0f);
	geomPosition = position.xyz;

    gl_Position = t_viewProjectionMatrix * position;
}