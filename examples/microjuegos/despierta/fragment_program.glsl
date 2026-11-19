#version 330

// posición del píxel en píxeles de la ventana, con el origen abajo a la izquierda
in vec2 pixel;
out vec4 outColor;

// resolucion y tiempo los entrega el anfitrión; los demás vienen de uniformes()
uniform vec2 resolucion;
uniform float tiempo;
uniform float sueno;   // 0 con los ojos abiertos, 1 con los ojos cerrados
uniform float golpe;   // 1 justo después de una cachetada; decae a 0
uniform bool sonando;  // el despertador está sonando
uniform bool dormido;

// Convierte la distancia con signo al borde de una figura (negativa dentro) en
// cobertura: 1 dentro, 0 fuera, con un borde suave de un píxel.
float relleno(float distancia)
{
    return 1.0 - smoothstep(-1.0, 1.0, distancia);
}

float disco(vec2 p, vec2 centro, float radio)
{
    return relleno(distance(p, centro) - radio);
}

// rectángulo entre las esquinas minimo y maximo
float caja(vec2 p, vec2 minimo, vec2 maximo)
{
    vec2 q = abs(p - (minimo + maximo) / 2.0) - (maximo - minimo) / 2.0;
    return relleno(length(max(q, 0.0)) + min(max(q.x, q.y), 0.0));
}

// trazo recto de a a b
float segmento(vec2 p, vec2 a, vec2 b, float grosor)
{
    vec2 ab = b - a;
    float h = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return relleno(distance(p, a + h * ab) - grosor / 2.0);
}

// trazo de una circunferencia, solo en el sector que mira hacia direccion
float onda(vec2 p, vec2 centro, float radio, vec2 direccion)
{
    vec2 v = p - centro;
    float sector = smoothstep(0.6, 0.8, dot(v, direccion) / max(length(v), 0.001));
    return relleno(abs(length(v) - radio) - 1.5) * sector;
}

vec2 rotar(vec2 v, float angulo)
{
    float c = cos(angulo), s = sin(angulo);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

// letra Z de altura t, con la esquina inferior izquierda en el origen
float zeta(vec2 p, float t)
{
    float grosor = 0.18 * t;
    float letra = segmento(p, vec2(0.0, t), vec2(t, t), grosor);
    letra = max(letra, segmento(p, vec2(t, t), vec2(0.0), grosor));
    return max(letra, segmento(p, vec2(0.0), vec2(t, 0.0), grosor));
}

// despertador de dos campanas, con el centro del cuerpo en el origen
vec3 despertador(vec2 p, vec3 color)
{
    vec3 rojo = vec3(0.80, 0.15, 0.20);
    vec3 negro = vec3(0.12);

    color = mix(color, negro, segmento(p, vec2(-30.0, -40.0), vec2(-45.0, -70.0), 8.0));
    color = mix(color, negro, segmento(p, vec2(30.0, -40.0), vec2(45.0, -70.0), 8.0));

    // el martillo golpea una campana y la otra mientras suena
    vec2 martillo = vec2(sonando ? 14.0 * sin(tiempo * 90.0) : 0.0, 82.0);
    color = mix(color, negro, segmento(p, vec2(0.0, 50.0), martillo, 4.0));
    color = mix(color, negro, disco(p, martillo, 6.0));
    color = mix(color, 0.8 * rojo, disco(p, vec2(-42.0, 58.0), 24.0));
    color = mix(color, 0.8 * rojo, disco(p, vec2(42.0, 58.0), 24.0));

    color = mix(color, rojo, disco(p, vec2(0.0), 62.0));
    color = mix(color, vec3(0.97, 0.95, 0.88), disco(p, vec2(0.0), 50.0));
    for (int hora = 0; hora < 12; hora++) {
        vec2 direccion = rotar(vec2(0.0, 1.0), -float(hora) * 0.5236);
        color = mix(color, vec3(0.35), segmento(p, 40.0 * direccion, 46.0 * direccion, 3.0));
    }

    // las siete en punto: el horario gira 7/12 de vuelta en sentido horario
    color = mix(color, negro, segmento(p, vec2(0.0), rotar(vec2(0.0, 26.0), -7.0 * 0.5236), 5.0));
    color = mix(color, negro, segmento(p, vec2(0.0), vec2(0.0, 40.0), 3.0));
    return mix(color, rojo, disco(p, vec2(0.0), 4.0));
}

vec3 escena(vec2 p)
{
    vec3 color = mix(vec3(0.93, 0.84, 0.70), vec3(0.85, 0.72, 0.58), p.y / resolucion.y);

    // ventana con el cielo del amanecer
    vec2 v0 = vec2(90.0, 250.0), v1 = vec2(330.0, 520.0);
    vec2 sol = vec2(230.0, 290.0);
    vec3 cielo = mix(vec3(1.0, 0.62, 0.35), vec3(0.45, 0.70, 0.95), (p.y - v0.y) / (v1.y - v0.y));
    cielo += vec3(1.0, 0.8, 0.5) * 0.35 * exp(-distance(p, sol) / 60.0);
    cielo = mix(cielo, vec3(1.0, 0.95, 0.70), disco(p, sol, 34.0));
    color = mix(color, vec3(0.96), caja(p, v0 - 12.0, v1 + 12.0));
    color = mix(color, cielo, caja(p, v0, v1));
    color = mix(color, vec3(0.96), caja(p, vec2(204.0, v0.y), vec2(216.0, v1.y)));
    color = mix(color, vec3(0.96), caja(p, vec2(v0.x, 379.0), vec2(v1.x, 391.0)));

    // velador
    color = mix(color, vec3(0.45, 0.28, 0.16), caja(p, vec2(500.0, 0.0), vec2(780.0, 215.0)));
    color = mix(color, vec3(0.55, 0.35, 0.20), caja(p, vec2(490.0, 205.0), vec2(790.0, 230.0)));

    // el despertador tiembla y vibra mientras suena; las ondas salen de las campanas
    vec2 centro = vec2(645.0, 302.0);
    float giro = 0.0;
    if (sonando) {
        centro.x += 3.0 * sin(tiempo * 70.0);
        giro = 0.05 * sin(tiempo * 55.0);
        float radio = 34.0 + 8.0 * fract(tiempo * 4.0);
        color = mix(color, vec3(0.25), onda(p - centro, vec2(-42.0, 58.0), radio, vec2(-0.707, 0.707)));
        color = mix(color, vec3(0.25), onda(p - centro, vec2(42.0, 58.0), radio, vec2(0.707, 0.707)));
    }
    color = despertador(rotar(p - centro, giro), color);

    // frazada en primer plano, con rayas diagonales y el borde ondulado
    float borde = 120.0 + 14.0 * sin(p.x * 0.015) + 8.0 * sin(p.x * 0.041 + 1.0);
    float raya = smoothstep(-0.2, 0.2, sin((p.x / 60.0 + p.y / 90.0) * 6.2832));
    vec3 frazada = mix(vec3(0.25, 0.35, 0.65), vec3(0.32, 0.45, 0.78), raya);
    return mix(color, frazada, relleno(p.y - borde));
}

void main()
{
    vec2 centro = resolucion / 2.0;

    // la cachetada sacude la cabeza de lado a lado y el sueño la balancea
    vec2 p = pixel - centro;
    p.x += 25.0 * golpe * sin(tiempo * 45.0);
    p = rotar(p, 0.06 * sueno * sin(tiempo * 1.7)) + centro;

    // visión doble: la escena evaluada dos veces, desplazada a cada lado, y promediada
    vec2 desfase = vec2(20.0 * sueno * sueno, 0.0);
    vec3 color = 0.5 * (escena(p - desfase) + escena(p + desfase));

    // con sueño, los colores se apagan
    float gris = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(color, vec3(gris), 0.7 * sueno);
    color *= 1.0 - 0.35 * sueno;
    color = mix(color, vec3(1.0), 0.25 * golpe);

    // Párpados: solo se ve el interior de una elipse, cuya altura es la
    // apertura de los ojos. La distancia al borde se mide en vertical, desde
    // el píxel hasta la curva del párpado. Al cerrarse, los párpados se
    // traslapan en 3 píxeles, para que no quede una rendija.
    vec2 relativo = pixel - centro;
    float semieje_x = 0.75 * resolucion.x;
    float semieje_y = 0.75 * resolucion.y * (1.0 - sueno) - 3.0;
    float borde = semieje_y * sqrt(max(1.0 - relativo.x * relativo.x / (semieje_x * semieje_x), 0.0));
    float distancia = abs(relativo.y) - borde;
    // la luz que atraviesa el párpado lo tiñe de rojo; las pestañas oscurecen el borde
    vec3 parpado = mix(vec3(0.05, 0.02, 0.02), vec3(0.30, 0.08, 0.08), smoothstep(4.0, 14.0, distancia));
    color = mix(color, parpado, smoothstep(-2.0, 2.0, distancia));

    if (dormido) {
        for (int i = 0; i < 3; i++) {
            // cada Z sube, crece y se desvanece; las tres van desfasadas
            float vida = fract(0.5 * tiempo + float(i) / 3.0);
            vec2 esquina = vec2(420.0 + 120.0 * vida + 20.0 * sin(6.0 * vida + float(i)), 260.0 + 220.0 * vida);
            float opacidad = smoothstep(0.0, 0.15, vida) * (1.0 - vida);
            color = mix(color, vec3(0.95, 0.90, 1.0), opacidad * zeta(pixel - esquina, 30.0 + 50.0 * vida));
        }
    }

    outColor = vec4(color, 1.0);
}
