#version 330

in vec2 frag_uv;

uniform sampler2D flare_texture;
uniform vec3 flare_color;
uniform float flare_intensity;

out vec4 out_color;

void main()
{
    // la textura tiene la caída radial en el canal alfa y blanco en RGB.
    // multiplicamos color por alfa antes de salir, así con blending aditivo
    // (GL_ONE, GL_ONE) la suma se calcula directamente sobre el framebuffer
    float falloff = texture(flare_texture, frag_uv).a;
    out_color = vec4(flare_color * flare_intensity * falloff, falloff);
}
