#version 330

// Los dos adjuntos del framebuffer de geometría. El número de `location`
// corresponde al GL_COLOR_ATTACHMENT que se declaró con glDrawBuffers.
layout(location = 0) out vec4 out_view_position;
layout(location = 1) out vec4 out_view_normal;

in vec3 view_position;
in vec3 view_normal;

void main() {
    out_view_position = vec4(view_position, 1.0f);
    // La normal se renormaliza porque llega interpolada entre tres vértices,
    // y la combinación de tres vectores unitarios no es unitaria.
    out_view_normal = vec4(normalize(view_normal), 1.0f);
}
