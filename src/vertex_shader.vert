#version 140

in vec3 position;
in vec3 color;

out vec3 v_color;

uniform mat4 model;
uniform mat4 perspective;

void main() {
  v_color = color;
  gl_Position = perspective * model * vec4(position, 1.0);
}
