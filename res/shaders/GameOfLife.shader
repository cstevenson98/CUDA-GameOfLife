#shader vertex 
#version 410 core

layout(location = 0) in uint state;
varying vec4 vColor;

uniform vec4 u_OnColour;
uniform vec4 u_OffColour;
uniform uint fullCellWidth;
uniform vec4 windowXY;
uniform float rgbData[300];

uint xId;
uint yId;
float xmin; float xmax;
float ymin; float ymax;
float dx; float dy;

uint store;
void main(void) 
{
	store = uint(floor(gl_VertexID/fullCellWidth));
	xId = gl_VertexID - (fullCellWidth * store);
	yId = store;

	dx = (windowXY.y - windowXY.x) / float(fullCellWidth);
	dy = (windowXY.w - windowXY.z) / float(fullCellWidth);

	gl_Position = vec4(windowXY.x+dx/2 + xId * dx, windowXY.z+dy/2 + yId * dy, 0., 1.0);
	
	vColor = vec4(
				u_OnColour.x * float(state) + u_OffColour.x * (1 - float(state)),
				u_OnColour.y * float(state) + u_OffColour.y * (1 - float(state)),
				u_OnColour.z * float(state) + u_OffColour.z * (1 - float(state)), 
				1.);

};



#shader fragment
#version 410 core
varying vec4 vColor;
void main(void) 
{
	gl_FragColor = vColor;
};