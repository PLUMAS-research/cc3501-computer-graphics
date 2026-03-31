#version 330

uniform sampler2D raster_tex;
in vec2 frag_texcoord;
out vec4 out_color;

void main()
{
    out_color = texture(raster_tex, frag_texcoord);
}
