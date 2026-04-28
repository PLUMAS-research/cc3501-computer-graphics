#version 330

in vec2 frag_texcoord;

uniform sampler2D diffuse;
uniform float alpha_test_threshold;
uniform bool premultiply;

out vec4 out_color;

void main()
{
    vec4 sample_color = texture(diffuse, frag_texcoord);

    // Alpha test: descartar pixeles con alpha bajo el umbral.
    if (sample_color.a < alpha_test_threshold) {
        discard;
    }

    // Cuando el modo de blending es premultiplicado, multiplicamos el RGB por
    // alpha en el shader. La funcion de blending opera con (ONE, 1-src_alpha)
    // y obtiene el resultado correcto en los bordes.
    if (premultiply) {
        out_color = vec4(sample_color.rgb * sample_color.a, sample_color.a);
    } else {
        out_color = sample_color;
    }
}
