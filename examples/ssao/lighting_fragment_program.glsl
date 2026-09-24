#version 330

in vec3 view_position;
in vec3 view_normal;
in vec3 fragment_color;

out vec4 out_color;

uniform sampler2D occlusion_texture;
uniform vec2 resolution;
uniform vec3 light_view_position;
uniform float ambient_strength;
uniform int use_occlusion;

void main() {
    // La oclusión vive en una textura del tamaño de la ventana, así que el
    // fragmento la indexa con su propia posición en pantalla. gl_FragCoord
    // está en píxeles y `texture` espera coordenadas normalizadas.
    vec2 screen_texcoord = gl_FragCoord.xy / resolution;
    float occlusion = use_occlusion == 1
        ? texture(occlusion_texture, screen_texcoord).r
        : 1.0f;

    vec3 normal = normalize(view_normal);
    vec3 light_direction = normalize(light_view_position - view_position);
    vec3 view_direction = normalize(-view_position);
    vec3 halfway = normalize(light_direction + view_direction);

    vec3 ambient = ambient_strength * fragment_color * occlusion;
    vec3 diffuse = fragment_color * max(dot(normal, light_direction), 0.0f);
    vec3 specular = vec3(0.25f) * pow(max(dot(normal, halfway), 0.0f), 32.0f);

    out_color = vec4(min(ambient + diffuse + specular, vec3(1.0f)), 1.0f);
}
