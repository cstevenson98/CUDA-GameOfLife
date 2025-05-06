#version 330 core
out vec4 FragColor;

uniform float time;
uniform vec3 color;
uniform float speed;

void main() {
  // Create a strobing effect using sine wave with adjustable speed
  float intensity = (sin(time * speed * 10.0) + 1.0) / 2.0;
  FragColor = vec4(color, 1.0) * intensity;
}