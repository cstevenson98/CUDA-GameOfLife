#shader vertex 
#version 410 core

layout(location = 0) in float state;
varying vec4 vColor;

uniform uint fullCellWidth;
uniform vec4 windowXY;
uniform float rgbData[300];
uniform float t;

uint xId;
uint yId;
float xmin; float xmax;
float ymin; float ymax;
float dx; float dy;
float x; float y;
float r;
float z;
uint index;
uint store;

void main(void) 
{
	store = uint(floor(gl_VertexID/fullCellWidth));
	xId = gl_VertexID - (fullCellWidth * store);
	yId = store;

	dx = (windowXY.y - windowXY.x) / float(fullCellWidth);
	dy = (windowXY.w - windowXY.z) / float(fullCellWidth);

	x = windowXY.x+dx/2 + xId * dx;
	y = windowXY.z+dy/2 + yId * dy;

	r = sqrt(x * x + y * y);

	z = (0.5 * (cos(3.141*(3.0 * r - 5.0 * t)) + 1.0));

	index = uint(floor(z * 100));

	gl_Position = vec4(x, y, 0., 1.0);

	vColor = vec4(rgbData[3*index], rgbData[3*index + 1], rgbData[3*index + 2], 1.);
};



#shader fragment
#version 410 core
varying vec4 vColor;
void main(void) 
{
	gl_FragColor = vColor;
};