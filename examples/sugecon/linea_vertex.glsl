#version 330

in vec3 position;
in float alpha;

out float opacidad_vertice;

void main() {
    // el geometry shader proyecta los extremos y arma el segmento en pantalla
    opacidad_vertice = alpha;
    gl_Position = vec4(position, 1.0);
}
