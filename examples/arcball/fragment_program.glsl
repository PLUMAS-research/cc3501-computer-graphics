#version 330

in vec2 frag_texcoord;
out vec4 outColor;
uniform sampler2D sampler_tex;
uniform float alpha_cutoff;

void main()
{
    vec4 texel = texture(sampler_tex, frag_texcoord);
    if (texel.a < alpha_cutoff)
        discard;
    outColor = texel;
}