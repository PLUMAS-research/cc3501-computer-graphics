#version 330

in vec2 fragment_patch_uv;

uniform sampler3D tam_volume;
uniform int tam_levels;

out vec4 fragment_color;

void main() {
    // El overlay muestra los niveles del TAM como un atlas 2 x 3:
    // dos columnas y tres filas, total 6 celdas. La fila inferior tiene los
    // niveles mas claros (papel/casi vacio) y la superior los mas oscuros.
    vec2 grid = vec2(2.0, 3.0);
    vec2 cell_size = 1.0 / grid;
    vec2 cell_index = floor(fragment_patch_uv * grid);
    cell_index = clamp(cell_index, vec2(0.0), grid - 1.0);
    float level = cell_index.y * grid.x + cell_index.x;
    level = clamp(level, 0.0, float(tam_levels - 1));

    vec2 cell_uv = fract(fragment_patch_uv * grid);

    // Tomamos el centro de cada slice (sin interpolacion vertical) para
    // mostrar el nivel real de cada celda. Si la celda tiene tono = level
    // entonces r = (level + 0.5) / tam_levels cae exactamente en el centro
    // de su slice.
    float r = (level + 0.5) / float(tam_levels);
    float ink = texture(tam_volume, vec3(cell_uv, r)).r;

    vec3 paper_color = vec3(0.97, 0.94, 0.84);
    vec3 ink_color = vec3(0.18, 0.12, 0.08);
    vec3 frame_color = vec3(0.20, 0.15, 0.10);
    vec3 color = mix(paper_color, ink_color, ink);

    // Borde delgado entre celdas
    vec2 cell_border = min(cell_uv, 1.0 - cell_uv);
    float cell_inside = smoothstep(0.0, 0.025,
                                   min(cell_border.x, cell_border.y));
    color = mix(frame_color, color, cell_inside);

    // Borde exterior del overlay completo
    vec2 outer_border = min(fragment_patch_uv, 1.0 - fragment_patch_uv);
    float outer_inside = smoothstep(0.0, 0.015,
                                    min(outer_border.x, outer_border.y));
    color = mix(frame_color, color, outer_inside);

    fragment_color = vec4(color, 1.0);
}
