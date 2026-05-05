#version 330
in vec2 position;
in vec2 uv;

uniform vec2 flare_center_ndc;
uniform float flare_size;     // fracción del alto de la pantalla
uniform float aspect_ratio;   // ancho / alto

out vec2 frag_uv;

void main()
{
    // tamaño en NDC: el alto es 2 * flare_size (porque NDC va de -1 a 1
    // y flare_size se expresa en fracción del alto), el ancho se compensa
    // por el aspect_ratio para que el halo se vea circular en pantalla
    vec2 size_ndc = vec2(2.0 * flare_size / aspect_ratio, 2.0 * flare_size);
    vec2 vertex_ndc = flare_center_ndc + position * size_ndc;
    gl_Position = vec4(vertex_ndc, 0.0, 1.0);
    frag_uv = uv;
}
