#version 330

in vec3 world_position;
in vec3 world_normal;

uniform vec3 color;             // albedo plano de la instancia
uniform float es_piso;          // 1.0 => tablero procedural en vez de color plano

uniform vec3 light_position;
uniform vec3 light_color;
uniform vec3 view_position;
uniform float ambient_strength;
uniform float diffuse_strength;
uniform float specular_strength;
uniform float shininess;

uniform float escala_tablero;
uniform vec3 color_piso_a;
uniform vec3 color_piso_b;

out vec4 outColor;

void main()
{
    // mismo Phong y mismo tablero que el trazador por CPU, para comparar
    vec3 albedo = color;
    if (es_piso > 0.5f) {
        float celda_x = floor(world_position.x * escala_tablero);
        float celda_z = floor(world_position.z * escala_tablero);
        float par = mod(celda_x + celda_z, 2.0f);
        albedo = (par < 0.5f) ? color_piso_a : color_piso_b;
    }

    vec3 normal_unitaria = normalize(world_normal);
    vec3 hacia_luz = normalize(light_position - world_position);
    vec3 hacia_camara = normalize(view_position - world_position);
    vec3 media = normalize(hacia_luz + hacia_camara);

    vec3 resultado = ambient_strength * albedo;
    float difuso = max(dot(normal_unitaria, hacia_luz), 0.0f);
    resultado += diffuse_strength * difuso * albedo * light_color;
    float especular = pow(max(dot(normal_unitaria, media), 0.0f), shininess);
    resultado += specular_strength * especular * light_color;

    outColor = vec4(resultado, 1.0f);
}
