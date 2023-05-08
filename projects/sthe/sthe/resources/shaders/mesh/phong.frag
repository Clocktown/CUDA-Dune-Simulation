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
	Material t_material;
};

layout(binding = 2) uniform sampler2D t_diffuseMap;
layout(binding = 3) uniform sampler2D t_normalMap;

layout(early_fragment_tests) in;
in Fragment fragment;
flat in mat3 tbnMatrix;

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

vec3 getDiffuseColor()
{
	if (t_material.hasDiffuseMap)
	{
		return t_material.diffuseColor * texture(t_diffuseMap, fragment.uv).rgb;
	}

	return t_material.diffuseColor;
}

vec3 getSpecularColor()
{
	const vec3 specularColor = t_material.specularIntensity * t_material.specularColor;
	return specularColor;
}

vec3 getNormal()
{
	if (t_material.hasNormalMap)
	{
		return normalize(tbnMatrix * (2.0f * texture(t_normalMap, fragment.uv).rgb - 1.0f));
	}

	return normalize(fragment.normal);
}

void main()
{
	const vec3 diffuseColor = getDiffuseColor();
	const vec3 specularColor = getSpecularColor();
	const vec3 normal = getNormal();

	const vec3 viewVector = t_inverseViewMatrix[3].xyz - fragment.position;
	const float viewDistance = length(viewVector);
	const vec3 viewDirection = viewVector / (viewDistance + EPSILON);

	fragmentColor.rgb = getAmbientColor() * diffuseColor;
	fragmentColor.a = t_material.opacity;

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
		const float cosPsiN = t_material.shininess > 0.0f ? pow(max(dot(reflection, viewDirection), 0.0f), t_material.shininess) : 0.0f;

		fragmentColor.rgb += attenuation * t_environment.lights[i].intensity * t_environment.lights[i].color * 
			                 (cosPhi * diffuseColor + cosPsiN * specularColor);
	}

	fragmentColor.rgb = clamp(fragmentColor.rgb, 0.0f, 1.0f);
	fragmentColor.rgb = mix(t_environment.fogColor, fragmentColor.rgb, getFogIntensity(viewDistance));
}
