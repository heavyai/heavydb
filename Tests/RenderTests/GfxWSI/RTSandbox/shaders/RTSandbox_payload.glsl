/*
RTSandbox_payload.glsl
Ray payload specific to RTSandbox's material options
*/
struct RayPayload {
  vec3 color;
  vec3 reflectivity;
  float roughness;
  vec3 position;        // hit point in world space
  vec3 normal;          // surface normal in world space
  vec3 face_normal;     // geometric normal in world space
};
