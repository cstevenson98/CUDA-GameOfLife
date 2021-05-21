#shader vertex 
#version 410 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 color;
varying vec3 vColor;
void main(void) {
	gl_Position = position;
	vColor = color;
};

#shader fragment
#version 410 core

varying vec3 vColor;
void main(void) {
	gl_FragColor = vec4(vColor, 1.);
};