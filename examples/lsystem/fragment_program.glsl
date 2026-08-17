#version 330

// Sombreado difuso de dos caras: el valor absoluto del producto punto ilumina
// igual la cara de adelante y la de atras. Se usa asi porque las hojas son
// cuadrilateros sin grosor y, segun desde donde se miren, su normal apunta en
// sentido contrario a la luz; sin el valor absoluto la mitad de las hojas de
// un arbol saldria negra.

in vec3 world_normal;
in vec3 vertex_color;

uniform vec3 light_direction;
uniform float ambient_strength;

out vec4 out_color;

void main()
{
    vec3 n = normalize(world_normal);
    float diffuse = abs(dot(n, normalize(light_direction)));
    float lighting = ambient_strength + (1.0 - ambient_strength) * diffuse;
    out_color = vec4(vertex_color * lighting, 1.0);
}
