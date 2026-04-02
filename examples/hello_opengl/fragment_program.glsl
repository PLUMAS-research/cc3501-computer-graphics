#version 330

in vec3 fragColor;
in vec3 virtualNDC;
out vec4 outColor;

// usamos int en lugar de bool para evitar problemas con los bindings de Python
uniform int clip_enabled;

void main()
{
    // simulamos el recorte del volumen normalizado:
    // descartamos los fragmentos cuya posición (antes de viewport_scale)
    // queda fuera del cubo [-1, 1]^3
    if (clip_enabled != 0 && (
        virtualNDC.x < -1.0 || virtualNDC.x > 1.0 ||
        virtualNDC.y < -1.0 || virtualNDC.y > 1.0 ||
        virtualNDC.z < -1.0 || virtualNDC.z > 1.0
    )) {
        discard;
    }

    outColor = vec4(fragColor, 1.0);
}
