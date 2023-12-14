#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
  mat4 modelMatrix;
  mat4 viewMatrix;
  mat4 projMatrix;
  vec3 diffuseColor;
  vec3 specularColor;
  float shininess;
  vec3 lightPos;
  float lightIntensity;
  vec3 lightColor;
}
ubo;

void main() {
  vec3 cameraPos = vec3(0.0, 0.0, -5.0);
  vec3 L = normalize(ubo.lightPos - fragWorldPos.xyz);
  vec3 N = normalize(fragWorldNormal);
  vec3 V = normalize(cameraPos - fragWorldPos.xyz);
  vec3 H = normalize(L + V);

  float distance = length(ubo.lightPos - fragWorldPos.xyz);
  float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);

  float lambertian = max(dot(N, L), 0.0);
  vec3 diffuse = ubo.diffuseColor * lambertian * ubo.lightIntensity *
                 attenuation * ubo.lightColor;

  float blinnPhong = pow(max(dot(N, H), 0.0), ubo.shininess);
  vec3 specular = ubo.specularColor * blinnPhong * ubo.lightIntensity *
                  attenuation * ubo.lightColor;
  vec3 ambient = vec3(0.1);
  outColor = vec4(pow(diffuse + specular + ambient, vec3(2.2)), 1.0);
}
