#shader vertex
#version 330 core
layout (location = 0) in vec2 aField; // Position of the charge

// uniform
uniform uint xWidth;
uniform uint yWidth;
out vec2 strength;

vec2 pos;

void main()
{
    // Position should be based on the index and between -1 and 1
    pos.x = (gl_VertexID % int(xWidth)) / float(xWidth) * 2.0 - 1.0;
    pos.y = (gl_VertexID / int(xWidth)) / float(yWidth) * 2.0 - 1.0;


    strength = aField;
    gl_Position = vec4(pos, 0.0, 1.0);
}

#shader fragment
#version 330 core

out vec4 FragColor;

in vec2 strength;
float strengthValue;
void main()
{
    strengthValue = length(strength);

    FragColor = vec4(strengthValue/22., strengthValue/22., 0., 1.0); // Red color for point charges
}