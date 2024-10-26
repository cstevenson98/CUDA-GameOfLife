#shader vertex
#version 330 core
layout (location = 0) in float aField; // Position of the charge

// uniform
uniform uint xWidth;
uniform uint yWidth;
out float strength;

vec2 pos;

void main()
{
    // Position should be based on the index and between -1 and 1
    pos.x = (gl_VertexID % int(xWidth)) / float(xWidth) * 2.0 - 1.0;
    pos.y = (gl_VertexID / int(xWidth)) / float(yWidth) * 2.0 - 1.0;

    // offset by half a pixel to center the point
    pos += vec2(1.0 / xWidth, 1.0 / yWidth);

    strength = aField;
    gl_Position = vec4(pos, 0.0, 1.0);
}

#shader fragment
#version 330 core

out vec4 FragColor;

in float strength;

void main()
{
    FragColor = vec4(strength/5., 0., 0., 1.0); // Red color for point charges
}