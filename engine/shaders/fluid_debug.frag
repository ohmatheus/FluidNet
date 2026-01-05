#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D densityTexture;
uniform sampler2D emitterTexture;
uniform sampler2D colliderTexture;

void main()
{
    float density = texture(densityTexture, TexCoord).r;
    float emitter = texture(emitterTexture, TexCoord).r;
    float collider = texture(colliderTexture, TexCoord).r;

    vec3 color = vec3(density);

    color = mix(color, vec3(0.2, 1.0, 0.2), emitter * 0.6);

    color = mix(color, vec3(1.0, 0.2, 0.2), collider * 0.6);

    FragColor = vec4(color, 1.0);
}
