#version 330

// Pass de profundidad: solo escribimos al depth buffer. gl_FragCoord.z
// se calcula automáticamente. El color que sale por out_color no se lee
// porque el FBO del shadow pass no tiene color attachment.
out vec4 out_color;

void main() {
    out_color = vec4(1.0);
}
