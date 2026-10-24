#version 330

in vec3 position;

void main() {
    // las cuentas de la silueta se hacen en el geometry shader, en espacio de mundo
    gl_Position = vec4(position, 1.0);
}
