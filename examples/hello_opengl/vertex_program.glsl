#version 330
in vec3 position;

uniform float angle;
uniform float object_scale;
uniform float viewport_scale;

out vec3 fragColor;
out vec3 virtualNDC;

void main()
{
    // rotación en torno al eje Y
    float c = cos(angle);
    float s = sin(angle);
    vec3 rotated = vec3(
        c * position.x + s * position.z,
        position.y,
        -s * position.x + c * position.z
    );

    // escala del objeto dentro del volumen normalizado
    vec3 scaled = rotated * object_scale;

    // guardamos la posición antes de aplicar viewport_scale,
    // para que el fragment shader pueda verificar si está dentro de [-1, 1]
    virtualNDC = scaled;

    fragColor = vec3(1.0, 1.0, 1.0);

    // viewport_scale reduce el cubo [-1,1]^3 para que sea visible en pantalla
    gl_Position = vec4(scaled * viewport_scale, 1.0);
}
