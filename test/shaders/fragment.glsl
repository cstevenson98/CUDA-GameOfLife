#version 330 core
out vec4 FragColor;

uniform float time;

void main() {
    // Create a strobing effect using sine wave
    float intensity = (sin(time * 10.0) + 1.0) / 2.0;
    FragColor = vec4(1.0, 1.0, 1.0, 1.0) * intensity;
} 