#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D densityTexture;

void main()
{
    float density = texture(densityTexture, TexCoord).r;

    // Simple grayscale visualization
    vec3 color = vec3(density);

    FragColor = vec4(color, 1.0);
}
