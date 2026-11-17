#version 330

// posición del píxel en píxeles de la ventana, con el origen abajo a la izquierda
in vec2 pixel;
out vec4 outColor;

// resolucion la entrega el anfitrión; los demás vienen de uniformes()
uniform vec2 resolucion;
uniform vec2 jugador;
uniform float radio_jugador;
uniform float radio_roca;
uniform vec2 rocas[16];
uniform int cantidad_rocas;
uniform bool golpeado;

// 1 dentro del círculo, 0 fuera, con un borde suave de un píxel
float disco(vec2 punto, vec2 centro, float radio)
{
    return 1.0 - smoothstep(radio - 1.0, radio + 1.0, distance(punto, centro));
}

void main()
{
    // cielo: degradado vertical
    vec3 color = mix(vec3(0.98, 0.85, 0.65), vec3(0.35, 0.55, 0.85), pixel.y / resolucion.y);

    for (int indice = 0; indice < cantidad_rocas; indice++) {
        color = mix(color, vec3(0.30, 0.25, 0.22), disco(pixel, rocas[indice], radio_roca));
    }

    vec3 color_jugador = golpeado ? vec3(0.85, 0.15, 0.15) : vec3(0.81, 0.22, 0.54);
    color = mix(color, color_jugador, disco(pixel, jugador, radio_jugador));

    outColor = vec4(color, 1.0);
}
