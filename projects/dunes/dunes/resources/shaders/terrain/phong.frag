#version 460 core

// Constants
const int FOG_MODE_NONE = 0;
const int FOG_MODE_LINEAR = 1;
const int FOG_MODE_EXPONENTIAL = 2;
const int FOG_MODE_EXPONENTIAL2 = 3;
const int LIGHT_TYPE_SPOT = 1;
const int LIGHT_TYPE_DIRECTIONAL = 2;
const float EPSILON = 1e-6f;

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

struct Fragment
{
	vec3 position;
	vec3 normal;
	vec2 uv;
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
layout(binding = 3) uniform sampler2D t_alphaMap;
layout(binding = 4) uniform sampler2D t_diffuseMaps[4];
layout(binding = 8) uniform sampler2D t_windMap;
layout(binding = 9) uniform sampler2D t_resistanceMap;

layout(early_fragment_tests) in;
in Fragment fragment;

// Output
layout(location = 0) out vec4 fragmentColor;

// Functionality
vec3 getAmbientColor()
{
	const vec3 ambientColor = t_environment.ambientIntensity * t_environment.ambientColor;
	return ambientColor;
}

float getFogIntensity(const float t_viewDistance)
{
	float fogIntensity;

	switch (t_environment.fogMode)
	{
	case FOG_MODE_LINEAR:
		fogIntensity = (t_environment.fogEnd - t_viewDistance) / (t_environment.fogEnd - t_environment.fogStart);
		break;
	case FOG_MODE_EXPONENTIAL:
		fogIntensity = exp(-t_environment.fogDensity * t_viewDistance);
		break;
	case FOG_MODE_EXPONENTIAL2:
		fogIntensity = exp(-pow(t_environment.fogDensity * t_viewDistance, 2.0));
		break;
	default:
		fogIntensity = 1.0f;
		break;
	}

	return clamp(fogIntensity, 0.0f, 1.0f);
}

vec3 getDiffuseColor(const int t_index)
{
	if (t_terrain.layers[t_index].hasDiffuseMap)
	{
		return t_terrain.layers[t_index].diffuseColor + texture(t_diffuseMaps[t_index], fragment.uv).rgb;
	}

	return t_terrain.layers[t_index].diffuseColor;
}

vec3 getSpecularColor(const int t_index)
{
	const vec3 specularColor = t_terrain.layers[t_index].specularIntensity * t_terrain.layers[t_index].specularColor;
	return specularColor;
}


void main()
{
	const vec3 viewVector = t_inverseViewMatrix[3].xyz - fragment.position;
	const float viewDistance = length(viewVector);
	const vec3 viewDirection = viewVector / (viewDistance + EPSILON);

	vec3 normal = fragment.normal;
	if(normal.y == 0.0f) {
		// TODO: This is bugged for non-square resolutions.
		const vec2 size = vec2(2.0f * t_terrain.gridScale,0.0f);
        const ivec3 offset = ivec3(-1, 0, 1);

		const vec2 terrain01 = textureOffset(t_heightMap, fragment.uv, offset.xy).xy;
        const vec2 terrain21 = textureOffset(t_heightMap, fragment.uv, offset.zy).xy;
        const vec2 terrain10 = textureOffset(t_heightMap, fragment.uv, offset.yx).xy;
        const vec2 terrain12 = textureOffset(t_heightMap, fragment.uv, offset.yz).xy;
		const vec3 edge1 = normalize(vec3(size.x, t_terrain.heightScale * ((terrain21.x + terrain21.y) - (terrain01.x + terrain01.y)), size.y));
		const vec3 edge2 = normalize(vec3(size.y, t_terrain.heightScale * ((terrain12.x + terrain12.y) - (terrain10.x + terrain10.y)), size.x));
		normal = cross(edge2, edge1);
	} 

	vec4 alphas;
	int startLayer;

	if (t_terrain.hasAlphaMap) 
	{
	    alphas =  texture(t_alphaMap, fragment.uv);
		startLayer = 0;
	}
	else 
	{
	    alphas = vec4(1.0f);
		startLayer = t_terrain.layerCount - 1;
	}

	const vec3 ambientColor = getAmbientColor();
	vec3 colors[4] = { vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(0.0f) };
	
	for (int i = 0; i < t_environment.lightCount; ++i)
	{
		vec3 lightDirection;
		float attenuation;

		if (t_environment.lights[i].type == LIGHT_TYPE_DIRECTIONAL)
		{
			lightDirection = -t_environment.lights[i].direction;
			attenuation = 1.0f;
		}
		else
		{
			const vec3 lightVector = t_environment.lights[i].position - fragment.position;
			const float lightDistance2 = dot(lightVector, lightVector);
			const float lightDistance = sqrt(lightDistance2);

			if (lightDistance >= t_environment.lights[i].range)
			{
				continue;
			}

			lightDirection = lightVector / (lightDistance + EPSILON);
			attenuation = clamp(1.0f / (t_environment.lights[i].attenuation.x +
				                        t_environment.lights[i].attenuation.y * lightDistance +
				                        t_environment.lights[i].attenuation.z * lightDistance2), 0.0f, 1.0f);

			if (t_environment.lights[i].type == LIGHT_TYPE_SPOT)
			{
			    const float cosTheta = dot(lightDirection, -t_environment.lights[i].direction);

				if (cosTheta < t_environment.lights[i].spotOuterCutOff)
				{
				    continue;
				}

				attenuation *= clamp((cosTheta - t_environment.lights[i].spotOuterCutOff) /
					                 (t_environment.lights[i].spotInnerCutOff - t_environment.lights[i].spotOuterCutOff), 0.0f, 1.0f);
			}
		}
		
		const vec3 reflection = reflect(-lightDirection, normal);
		const float cosPhi = max(dot(normal, lightDirection), 0.0f);
		const float cosPsi = max(dot(reflection, viewDirection), 0.0f);
		const vec3 lightColor = attenuation * t_environment.lights[i].intensity * t_environment.lights[i].color;

	    for (int j = startLayer; j < t_terrain.layerCount; ++j) 
	    {
	        const vec3 diffuseColor = mix(getDiffuseColor(j), vec3(0.6,0.1,0.1), 0.5*texture(t_resistanceMap, fragment.uv).x);
			
	        const vec3 specularColor = getSpecularColor(j);
			const float cosPsiN = t_terrain.layers[j].shininess > 0.0f ? pow(cosPsi, t_terrain.layers[j].shininess) : 0.0f;
		   
		    colors[j] += ambientColor * diffuseColor;
			colors[j] += lightColor * (cosPhi * diffuseColor + cosPsiN * specularColor);
	    }
	}

	fragmentColor = vec4(colors[startLayer], 1.0f);

	for (int i = startLayer + 1; i < t_terrain.layerCount; ++i) 
	{
	    fragmentColor.rgb = mix(colors[i], fragmentColor.rgb, alphas[i]);
	}

	fragmentColor.rgb = clamp(fragmentColor.rgb, 0.0f, 1.0f);
	fragmentColor.rgb = mix(t_environment.fogColor, fragmentColor.rgb, getFogIntensity(viewDistance));

	//fragmentColor.rgb = mix(fragmentColor.rgb, vec3(abs(normalize(texture2D(t_windMap, fragment.uv).rg)), 0.0f), 0.15f);

	
}
