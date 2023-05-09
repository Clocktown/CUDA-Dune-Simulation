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

in vec2 tescUV[];

// Output
layout(vertices = 4) out;
out vec2 teseUV[];

void main()
{
    if (gl_InvocationID == 0)
    {
        const float tesselationLevel = float(t_terrain.tesselationLevel);

        gl_TessLevelInner[0] = tesselationLevel;
        gl_TessLevelInner[1] = tesselationLevel;

        gl_TessLevelOuter[0] = tesselationLevel;
        gl_TessLevelOuter[1] = tesselationLevel;
        gl_TessLevelOuter[2] = tesselationLevel;
        gl_TessLevelOuter[3] = tesselationLevel;
    }

    teseUV[gl_InvocationID] = tescUV[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}