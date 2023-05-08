#version 460 core

// Input
layout (triangles) in;
in vec3 geomPosition[];
in vec2 geomUV[];

// Output
struct Fragment
{
	vec3 position;
	vec3 normal;
	vec2 uv;
};

layout (triangle_strip, max_vertices = 3) out;
out Fragment fragment;

void main() 
{   
    const vec3 edge1 = geomPosition[1] - geomPosition[0];
    const vec3 edge2 = geomPosition[2] - geomPosition[0];
    const vec3 normal = normalize(cross(edge1, edge2));

	for (int i = 0; i < 3; ++i) 
	{
	    fragment.position = geomPosition[i];
        fragment.normal = normal;
        fragment.uv = geomUV[i];

        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
	}
    
    EndPrimitive();
} 
