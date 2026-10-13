#version 330

in vec2 frag_texcoord;

uniform sampler2D diffuse;
uniform float alpha_test_threshold;

out vec4 out_color;

void main()
{
    vec4 sample_color = texture(diffuse, frag_texcoord);

    // Alpha test: descartar pixeles con alpha bajo el umbral.
    if (sample_color.a < alpha_test_threshold) {
        discard;
    }

    // El shader no multiplica por alpha en ningun modo. En modo premultiplicado
    // la textura ya viene multiplicada desde la CPU, y multiplicar aqui de
    // nuevo oscureceria el borde igual que el blending estandar.
    out_color = sample_color;
}
