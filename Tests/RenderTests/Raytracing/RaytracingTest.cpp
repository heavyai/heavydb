/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxDriver/GfxDriverTestFixtures.h"

#include <memory>
#include <random>
#include <string>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Render/Camera.h"
#include "GfxDriver/Render/LightManager.h"
#include "GfxDriver/Render/PBRMaterialManager.h"
#include "GfxDriver/Resources/AccelerationStructure.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/Utils/NoiseUtils.h"
#include "Tests/RenderTests/Utils/GoldenImage.h"
#include "Tests/RenderTests/Utils/MeshUtils.h"
#include "Tests/RenderTests/Utils/ShaderUtils.h"
#include "Tests/TestHelpers.h"

#define PRINT_TRACE_TIMES false
#if PRINT_TRACE_TIMES
#include "Shared/measure.h"
#endif

namespace GfxDriverTests {

// Render dimensions for all tests (golden images 400x300)
static constexpr uint32_t render_width = 400;
static constexpr uint32_t render_height = 300;

// Flag to force writing of all golden image files
static constexpr bool regenerate_golden_images = false;

// File paths
namespace {
static std::string g_golden_images_path(std::string(RENDER_TESTS_PATH) +
                                        "Raytracing/golden_images/");
// static std::string_view kFlushName{"RaytracingTest"};
}  // namespace

//
// RaytracingTestShaderLibrary
//
class RaytracingTestShaderLibrary {
 public:
  static void init(Library& library) {
    library.addFromManifestFile("ShaderManifest.json",
                                std::string(RENDER_TESTS_PATH) + "Raytracing/shaders/");
    library.addFromManifestFile("ShaderManifest.json",
                                std::string(GFX_DRIVER_PATH) + "Render/shaders/");
  }
};

//
// RaytracingTestBase fixture template
//
template <typename GTEST_BASE,
          typename DRIVER_TYPE,
          typename SHADER_LIBRARY = RaytracingTestShaderLibrary>
class RaytracingTestBase
    : public DeviceContextTest<GTEST_BASE, DRIVER_TYPE, SHADER_LIBRARY> {
 protected:
  using Base = DeviceContextTest<GTEST_BASE, DRIVER_TYPE, SHADER_LIBRARY>;

  void SetUp() override {
    Base::SetUp();
    golden_image_ = std::make_unique<GoldenImage>(
        g_golden_images_path, this->driver_->getType(), regenerate_golden_images);
  }
  void TearDown() override { Base::TearDown(); }

  std::pair<DeviceContext*, resource_ptr<Texture>> getDeviceAndOutput(
      std::string_view test_name,
      uint32_t output_width,
      uint32_t output_height) const {
    for (auto const& device : this->device_contexts_) {
      if (any_bits_set(device->getCapabilityBits() & DeviceCapabilityBits::kRaytracing)) {
        auto output_texture = device->getResourceManager().createTexture(
            test_name,
            output_width,
            output_height,
            1,
            PixelFormat::kRGBA8,
            1,
            false,
            ImageUsageBits::kStorageBit,
            default_texture_sampler_state_nearest);
        output_texture->clearPixels();
        return {device.get(), std::move(output_texture)};
      }
    }
    return {nullptr, nullptr};
  }

  std::unique_ptr<GoldenImage> golden_image_;
};

using RaytracingTest = RaytracingTestBase<::testing::Test, VulkanDriver>;

//
// DrawOneSphere
// Sphere with triangle color derived from barycentric coordinates
//
TEST_F(RaytracingTest, DrawOneSphere) {
  static constexpr std::string_view kTestName = "DrawOneSphere";
  auto [device_ptr, output_texture] =
      this->getDeviceAndOutput(kTestName, render_width, render_height);
  if (!device_ptr) {
    GTEST_SKIP() << "Could not find raytracing capable device";
  }
  auto& device = *device_ptr;

  // Resources
  auto& resource_mgr = device.getResourceManager();

  // Generate shader spir-v
  auto const& shader_mgr = this->gfx_context_->getShaderManager();
  auto caches = shader_mgr.createCacheVectorFromTemplate(
      {{"RaytracingTest/DrawOneSphere.raygen"},
       {"RaytracingTest/DrawOneSphere.miss"},
       {"RaytracingTest/DrawOneSphere.closest"}});
  ASSERT_TRUE(validate_shader_caches(caches));

  // Material
  auto material = resource_mgr.createMaterial(kTestName, caches);

  // Geometry
  auto sphere = build_uv_sphere(32, 32, true);
  auto [mesh_layout, mesh_vbo, mesh_ibo] =
      sphere->buildResources(resource_mgr,
                             "sphere",
                             BufferUsageBits::kDeviceAddressBit |
                                 BufferUsageBits::kAccelerationStructureReadOnlyBit);
  auto vertex_stride = mesh_layout->getNumBytesPerItem();

  // Create acceleration structure builder
  auto accel_builder = resource_mgr.createAccelerationStructureBuilder();

  //
  // Bottom level acceleration structure
  //
  resource_ptr<AccelerationStructure> blas;
  EXPECT_THROW(blas = resource_mgr.createBottomLevelAccelerationStructure("Test BLAS",
                                                                          *accel_builder),
               RenderError)
      << "Expected throw due to no empty BLAS";

  ASSERT_NO_THROW(accel_builder->addTriangleData({mesh_vbo->getDeviceAddress(),
                                                  sphere->numVertices(),
                                                  vertex_stride,
                                                  mesh_ibo->getDeviceAddress(),
                                                  IndexBufferDataType::kUnsigned32,
                                                  sphere->numIndices(),
                                                  0,  // transform
                                                  sphere->numTriangles()}));

  ASSERT_NO_THROW(blas = resource_mgr.createBottomLevelAccelerationStructure(
                      "Test BLAS", *accel_builder));

  //
  // Top level acceleration structure
  //
  resource_ptr<AccelerationStructure> tlas;
  BufferWrapperUqPtr instances_buffer;
  EXPECT_THROW(tlas = resource_mgr.createTopLevelAccelerationStructure(
                   "Test TLAS", *accel_builder, *instances_buffer),
               RenderError)
      << "Expected throw due to no empty TLAS";

  ASSERT_NO_THROW(
      accel_builder->addInstanceData(glm::mat4x3{1}, blas->getDeviceAddress()));

  // Create required instances buffer
  // TODO(scb): resolve ownership over this buffer (retain in Builder?)
  instances_buffer = resource_mgr.createBuffer(
      "instance data",
      {BufferType::kUnspecified,
       accel_builder->getInstancesBufferSize(),
       BufferUsageBits::kStorageBufferBit | BufferUsageBits::kDeviceAddressBit |
           BufferUsageBits::kAccelerationStructureReadOnlyBit});

  ASSERT_NO_THROW(tlas = resource_mgr.createTopLevelAccelerationStructure(
                      "Test TLAS", *accel_builder, *instances_buffer));

  // Camera
  Camera camera(Camera::Projection::kPerspective, Camera::Type::kLookAt);
  camera.setPerspectiveParams(
      60.0f, (float)render_width / (float)render_height, 0.1f, 512.0f);
  camera.setPosition({0.0f, 0.0f, 2.5f});
  camera.setTarget(glm::vec3{0.0f});

  material->setUniformAttribute("viewInverse", glm::inverse(camera.getViewTM()));
  material->setUniformAttribute("projInverse", glm::inverse(camera.getProjectionTM()));
  ASSERT_NO_THROW(material->setAccelerationStructureAttribute("topLevelAS", *tlas));
  material->setImageLoadStoreAttribute("outputImage", *output_texture);
  material->updateDescriptorSets();

  auto pipeline = resource_mgr.createRaytracingPipeline("test", *material, {});
  pipeline->create(1);

  auto const& shader_groups = pipeline->getShaderGroups();
  ASSERT_GT(shader_groups.size(), 0u);

  auto sbt = resource_mgr.createShaderBindingTable(*pipeline);

#if PRINT_TRACE_TIMES
  auto start_time = timer_start();
#endif
  device.getCommandList()
      .traceRays(*pipeline, *sbt, render_width, render_height, 1)
      .flush("trace");
#if PRINT_TRACE_TIMES
  std::cout << "render time: " << timer_stop_microseconds(start_time) << std::endl;
#endif

  // Compare to golden image
  EXPECT_TRUE(
      this->golden_image_->compare(*output_texture, "rt_sphere_test", false, 3, 1));

  // Cleanup resources
  resource_mgr.destroyPipeline(std::move(pipeline));
  resource_mgr.destroyAccelerationStructure(std::move(blas));
  resource_mgr.destroyAccelerationStructure(std::move(tlas));
  resource_mgr.destroyBuffer(std::move(instances_buffer));
  resource_mgr.destroyBuffer(std::move(mesh_vbo));
  resource_mgr.destroyBuffer(std::move(mesh_ibo));
  resource_mgr.destroyTexture(std::move(output_texture));
}

//
// InstancingTest
//
// Complex shading with reflections and shadows
// - Test multiple instances in the TLAS
// - Reflection via raygen shader
// - Shadows via raycasts from closest hit shader
//
TEST_F(RaytracingTest, InstancingTest) {
  static constexpr std::string_view kTestName = "InstancingTest";
  auto [device_ptr, output_texture] =
      this->getDeviceAndOutput(kTestName, render_width, render_height);
  if (!device_ptr) {
    GTEST_SKIP() << "Could not find raytracing capable device";
  }
  auto& device = *device_ptr;

  // Resources
  auto& resource_mgr = device.getResourceManager();

  // Generate shader spir-v
  auto const& shader_mgr = this->gfx_context_->getShaderManager();

  auto builders = shader_mgr.createBuilderVector({{"RaytracingTest/Instancing.raygen"},
                                                  {"RaytracingTest/Instancing.miss"},
                                                  {"Shading/rtShadowsSimple.miss"},
                                                  {"RaytracingTest/Instancing.closest"}});
  builders[0]->setExternalUniformBuffers({"PROJECTION_UBO_TYPE"});
  auto caches = shader_mgr.createCacheVector(std::move(builders));
  ASSERT_TRUE(validate_shader_caches(caches));

  // Material
  auto material = resource_mgr.createMaterial(kTestName, caches, true);

  // Geometry
  auto sphere = build_uv_sphere(32, 32, true);
  auto [mesh_layout, mesh_vbo, mesh_ibo] =
      sphere->buildResources(resource_mgr,
                             "sphere",
                             BufferUsageBits::kDeviceAddressBit |
                                 BufferUsageBits::kAccelerationStructureReadOnlyBit |
                                 BufferUsageBits::kStorageBufferBit);

  // Bottom level acceleration structure
  auto accel_builder = resource_mgr.createAccelerationStructureBuilder();
  accel_builder->addTriangleData({mesh_vbo->getDeviceAddress(),
                                  sphere->numVertices(),
                                  mesh_layout->getNumBytesPerItem(),
                                  mesh_ibo->getDeviceAddress(),
                                  IndexBufferDataType::kUnsigned32,
                                  sphere->numIndices(),
                                  0,  // transform
                                  sphere->numTriangles()});
  auto blas =
      resource_mgr.createBottomLevelAccelerationStructure("Test BLAS", *accel_builder);

  //
  // Instances
  //

  // Add TLAS instance data to AccelerationStructure Builder
  // generate random distribution of spheres within a cube
  const uint32_t kNumInstances = 100;
  const float kSphereScale = 0.2f;
  std::mt19937 pos_gen(1);
  std::mt19937 d_gen(2);
  std::uniform_real_distribution<float> pos_dis(-1.0f, 1.0f);

  for (uint32_t i = 0; i < kNumInstances; ++i) {
    AccelerationStructure::InstanceMatrixType m{1};
    m[0][3] = pos_dis(pos_gen);
    m[1][3] = pos_dis(pos_gen);
    m[2][3] = pos_dis(pos_gen);
    m[0][0] = kSphereScale;
    m[1][1] = kSphereScale;
    m[2][2] = kSphereScale;
    accel_builder->addInstanceData(m, blas->getDeviceAddress());
  }

  // Create a buffer to hold the instance data
  auto instances_buffer = resource_mgr.createBuffer(
      "instance data",
      {BufferType::kUnspecified,
       accel_builder->getInstancesBufferSize(),
       BufferUsageBits::kStorageBufferBit | BufferUsageBits::kDeviceAddressBit |
           BufferUsageBits::kAccelerationStructureReadOnlyBit});

  // Create the TLAS, filling the buffer with the accumulated instance data in
  // the builder
  auto tlas = resource_mgr.createTopLevelAccelerationStructure(
      "Test TLAS", *accel_builder, *instances_buffer);

  // Pipeline
  auto pipeline = resource_mgr.createRaytracingPipeline(kTestName, *material, {});
  pipeline->create(2);  // stack depth of 2 (raygen (primary or reflect) + shadow)

  // Shader binding table
  auto sbt = resource_mgr.createShaderBindingTable(*pipeline);

  // Camera
  Camera camera(Camera::Projection::kPerspective, Camera::Type::kLookAt);
  camera.setPerspectiveParams(
      60.0f, (float)render_width / (float)render_height, 0.1f, 512.0f);
  camera.setPosition({0.0f, 0.0f, 2.5f});
  camera.setTarget(glm::vec3{0.0f});

  auto projection_ubo = resource_mgr.createBuffer("Projection UBO",
                                                  {BufferType::kUnspecified,
                                                   Camera::getBufferDataSize(),
                                                   BufferUsageBits::kUniformBufferBit,
                                                   BufferAccessType::kHostVisible});

  camera.updateBufferData(*projection_ubo);

  // Lights
  LightManager light_mgr;
  light_mgr.addParallelLight(
      glm::normalize(glm::vec3{0.5, -0.25, 0.5}), glm::vec3{1.5, 1.25, 1.0}, 1.0);
  light_mgr.addPointLight({-15, -35, 30}, {0.2, 0.2, 0.9}, 1.0, false);
  light_mgr.addPointLight({-5, 35, 5}, {0.8, 0.2, 0.0}, 2000.0, true);

  auto light_data_ssbo = light_mgr.createBuffer(resource_mgr);
  light_mgr.updateBufferData(*light_data_ssbo);
  light_mgr.bindToMaterial(*material, *light_data_ssbo, 0.001f);

  // Material properties
  PBRMaterialManager material_mgr;
  material_mgr.addCookTorranceMaterial(PBRMaterialDescriptor::ColorSource::kConstant,
                                       glm::vec4{0.4f, 0.6f, 0.2f, 1.0f},
                                       1.0f,
                                       0.2f,
                                       0.7f);
  auto material_data_ssbo = material_mgr.createBuffer(resource_mgr);
  material_mgr.updateBufferData(*material_data_ssbo);
  material_mgr.bindToMaterial(*material, *material_data_ssbo);

  material->setAccelerationStructureAttribute("topLevelAS", *tlas);
  material->setImageLoadStoreAttribute("outputImage", *output_texture);
  material->bindExternalUniformBufferToBlock("PROJECTION_UBO_TYPE", *projection_ubo);
  material->bindShaderStorageBufferToBlock("Vertices", *mesh_vbo);
  material->bindShaderStorageBufferToBlock("Indices", *mesh_ibo);

  material->updateDescriptorSets();

#if PRINT_TRACE_TIMES
  auto start_time = timer_start();
#endif
  device.getCommandList()
      .traceRays(*pipeline, *sbt, render_width, render_height, 1)
      .flush("trace");
#if PRINT_TRACE_TIMES
  std::cout << "render time: " << timer_stop_microseconds(start_time) << std::endl;
#endif

  // Compare to golden image
  // This test is sensitive to both driver version and RTX vs emulation so it needs
  // tolerance
  EXPECT_TRUE(
      this->golden_image_->compare(*output_texture, "rt_instancing_test", false, 7, 1));

  // Cleanup resources
  resource_mgr.destroyPipeline(std::move(pipeline));
  resource_mgr.destroyAccelerationStructure(std::move(blas));
  resource_mgr.destroyAccelerationStructure(std::move(tlas));
  resource_mgr.destroyBuffer(std::move(projection_ubo));
  resource_mgr.destroyBuffer(std::move(light_data_ssbo));
  resource_mgr.destroyBuffer(std::move(material_data_ssbo));
  resource_mgr.destroyBuffer(std::move(instances_buffer));
  resource_mgr.destroyBuffer(std::move(mesh_vbo));
  resource_mgr.destroyBuffer(std::move(mesh_ibo));
  resource_mgr.destroyTexture(std::move(output_texture));
}

//
// AABBTest
//
// - AABB intersection with intersection shader
// - multiple hit ShaderGroups (procedural vs mesh)
// - multiple BLASs (procedural vs mesh)
// - heightfield mesh builder
// - callable shader
// disabling during nVidia driver version bump
TEST_F(RaytracingTest, DISABLED_AABBTest) {
  static constexpr std::string_view kTestName = "AABBTest";

  auto [device_ptr, output_texture] =
      this->getDeviceAndOutput(kTestName, render_width, render_height);
  if (!device_ptr) {
    GTEST_SKIP() << "Could not find raytracing capable device";
  }
  auto& device = *device_ptr;

  // Resources
  auto& resource_mgr = device.getResourceManager();

  // Generate shader spir-v
  auto const& shader_mgr = this->gfx_context_->getShaderManager();

  auto builders =
      shader_mgr.createBuilderVector({{"RaytracingTest/AABB.raygen"},
                                      {"RaytracingTest/AABB.miss"},
                                      {"Shading/rtShadowsSimple.miss"},
                                      {"RaytracingTest/AABB.isect"},
                                      {"RaytracingTest/AABB_procedural.closest"},
                                      {"RaytracingTest/AABB.call"}});
  builders[0]->setExternalUniformBuffers({"PROJECTION_UBO_TYPE"});
  // Add a second intersection shader in a different hit-group
  const uint32_t kTriangleHitGroupIndex = 1;
  builders.push_back(shader_mgr.createBuilder("RaytracingTest/AABB_mesh.closest"));
  builders.back()->setRaytracingHitGroupIndex(kTriangleHitGroupIndex);

  auto caches = shader_mgr.createCacheVector(std::move(builders));
  ASSERT_TRUE(validate_shader_caches(caches));

  // Material
  auto material = resource_mgr.createMaterial(kTestName, caches, true);

  //
  // Geometry - procedural
  //
  // Sphere struct that will be passed to the shader in spheres_buffer
  const uint32_t kNumSpheres = 100;
  constexpr float kSphereRadius = 0.25f;

  struct Sphere {
    glm::vec3 center{0.0f};
    float radius{kSphereRadius};
  };

  std::mt19937 a_gen(2);
  std::uniform_real_distribution<float> a_dis(-1.0f, 1.0f);

  // Generate the spheres for use in the shader, and bounds for the BLAS
  std::vector<Sphere> spheres;
  std::vector<AccelerationStructure::AABB> bounds;
  for (uint32_t i = 0; i < kNumSpheres; ++i) {
    spheres.push_back(
        {glm::normalize(glm::vec3{a_dis(a_gen), a_dis(a_gen), a_dis(a_gen)}) * 1.5f,
         kSphereRadius});

    auto const& s = spheres.back();
    bounds.emplace_back(glm::vec3(-kSphereRadius) + s.center,
                        glm::vec3(kSphereRadius) + s.center);
  }

  // Create and update the spheres buffer
  auto spheres_buffer_size = spheres.size() * sizeof(Sphere);
  auto spheres_buffer = resource_mgr.createBuffer("Spheres",
                                                  {BufferType::kUnspecified,
                                                   spheres_buffer_size,
                                                   BufferUsageBits::kStorageBufferBit});
  spheres_buffer->updateSubData(spheres.data(), spheres_buffer_size, 0);

  // Create and update the bounds buffer
  auto buffer_size = bounds.size() * sizeof(AccelerationStructure::AABB);
  auto bounds_buffer = resource_mgr.createBuffer(
      "bounds",
      {BufferType::kUnspecified,
       buffer_size,
       BufferUsageBits::kStorageBufferBit | BufferUsageBits::kDeviceAddressBit |
           BufferUsageBits::kAccelerationStructureReadOnlyBit});
  bounds_buffer->updateSubData(bounds.data(), buffer_size, 0);

  //
  // Geometry - triangle heightfield
  //
  // Build a fractal noise driven heightfield mesh
  auto height_mesh = build_heightfield(2.0f, 2.0f, 1000, 1000, [](float u, float v) {
    auto f = fractal_noise(3, u, v, 1.f, 0.4f, 0.1f, 0.9f);
    return std::min(f, 0.6f) * 0.3f;
  });
  auto [mesh_layout, mesh_vbo, mesh_ibo] =
      height_mesh->buildResources(resource_mgr,
                                  "sphere",
                                  BufferUsageBits::kDeviceAddressBit |
                                      BufferUsageBits::kAccelerationStructureReadOnlyBit |
                                      BufferUsageBits::kStorageBufferBit);

  //
  // BLAS
  //

  // Create acceleration structure builder
  auto accel_builder = resource_mgr.createAccelerationStructureBuilder();

  // Build procedural object blas
  ASSERT_NO_THROW(accel_builder->addAABBData({*bounds_buffer,
                                              static_cast<uint32_t>(bounds.size()),
                                              sizeof(AccelerationStructure::AABB)}));

  resource_ptr<AccelerationStructure> procedural_blas;
  ASSERT_NO_THROW(procedural_blas = resource_mgr.createBottomLevelAccelerationStructure(
                      "Procedural BLAS", *accel_builder));

  // Build mesh object blas
  accel_builder->clearBlasData();
  accel_builder->addTriangleData({mesh_vbo->getDeviceAddress(),
                                  height_mesh->numVertices(),
                                  mesh_layout->getNumBytesPerItem(),
                                  mesh_ibo->getDeviceAddress(),
                                  IndexBufferDataType::kUnsigned32,
                                  height_mesh->numIndices(),
                                  0,  // transform
                                  height_mesh->numTriangles()});

#if PRINT_TRACE_TIMES
  auto blas_start_time = timer_start();
#endif
  resource_ptr<AccelerationStructure> mesh_blas;
  ASSERT_NO_THROW(mesh_blas = resource_mgr.createBottomLevelAccelerationStructure(
                      "Mesh BLAS", *accel_builder));
#if PRINT_TRACE_TIMES
  auto blas_time = timer_stop(blas_start_time);
  std::cout << "Mesh BLAS build time: " << blas_time << " ms" << std::endl;
#endif

  //
  // TLAS
  //

  // Procedural instances
  auto tm = glm::translate(glm::mat4{1.f}, {0.0f, -0.5f, 0.0f});
  tm = transpose(glm::scale(tm, glm::vec3{0.5f}));
  ASSERT_NO_THROW(accel_builder->addInstanceData(
      tm, procedural_blas->getDeviceAddress(), 0xff, 0, 0));

  // Mesh instances
  tm = glm::scale(glm::mat4{1}, glm::vec3{4.f});

  // Set the SBT offset to the triangle hit group index (==1), to select the second
  // hit-group in the SBT
  ASSERT_NO_THROW(accel_builder->addInstanceData(
      tm, mesh_blas->getDeviceAddress(), 0xff, 0, kTriangleHitGroupIndex));

  // Create required instances buffer
  auto instances_buffer = resource_mgr.createBuffer(
      "instance data",
      {BufferType::kUnspecified,
       accel_builder->getInstancesBufferSize(),
       BufferUsageBits::kStorageBufferBit | BufferUsageBits::kDeviceAddressBit |
           BufferUsageBits::kAccelerationStructureReadOnlyBit});

  // Top level acceleration structure
  resource_ptr<AccelerationStructure> tlas;
  ASSERT_NO_THROW(tlas = resource_mgr.createTopLevelAccelerationStructure(
                      "Test TLAS", *accel_builder, *instances_buffer));

  //
  // Pipeline and SBT
  //
  auto pipeline = resource_mgr.createRaytracingPipeline(kTestName, *material, {});
  pipeline->create(2);

  auto sbt = resource_mgr.createShaderBindingTable(*pipeline);

  //
  // Scene components
  //

  // Camera
  Camera camera(Camera::Projection::kPerspective, Camera::Type::kLookAt);
  camera.setPerspectiveParams(
      60.0f, (float)render_width / (float)render_height, 0.1f, 512.0f);
  camera.setPosition({0.0f, -1.f, 2.5f});
  camera.setTarget(glm::vec3{0.0f});

  auto projection_ubo = resource_mgr.createBuffer("Projection UBO",
                                                  {BufferType::kUnspecified,
                                                   Camera::getBufferDataSize(),
                                                   BufferUsageBits::kUniformBufferBit,
                                                   BufferAccessType::kHostVisible});

  camera.updateBufferData(*projection_ubo);

  // Lights
  LightManager light_mgr;
  light_mgr.addParallelLight(
      glm::normalize(glm::vec3{0.5, -0.35, 0.5}), glm::vec3{1.5, 1.25, 1.0}, 1.0);
  light_mgr.addPointLight({-15, -30, 0}, {0.3, 0.3, 0.8}, 0.4f, false);

  auto light_data_ssbo = light_mgr.createBuffer(resource_mgr);
  light_mgr.updateBufferData(*light_data_ssbo);
  light_mgr.bindToMaterial(*material, *light_data_ssbo, 0.001f);

  // PBRMaterial properties
  PBRMaterialManager material_mgr;
  material_mgr.addCookTorranceMaterial(PBRMaterialDescriptor::ColorSource::kConstant,
                                       glm::vec4{0.6f, 0.8f, 0.4f, 1.0f},
                                       1.0f,
                                       0.2f,
                                       0.5f);
  material_mgr.addCookTorranceMaterial(PBRMaterialDescriptor::ColorSource::kConstant,
                                       glm::vec4{0.5f, 0.5f, 0.5f, 1.0f},
                                       1.0f,
                                       0.2f,
                                       0.5f);
  auto material_data_ssbo = material_mgr.createBuffer(resource_mgr);
  material_mgr.updateBufferData(*material_data_ssbo);
  material_mgr.bindToMaterial(*material, *material_data_ssbo);

  // Update Material descriptors
  material->setAccelerationStructureAttribute("topLevelAS", *tlas);
  material->setImageLoadStoreAttribute("outputImage", *output_texture);
  material->bindExternalUniformBufferToBlock("PROJECTION_UBO_TYPE", *projection_ubo);
  material->bindShaderStorageBufferToBlock("Spheres", *spheres_buffer);
  material->bindShaderStorageBufferToBlock("Vertices", *mesh_vbo);
  material->bindShaderStorageBufferToBlock("Indices", *mesh_ibo);

  material->updateDescriptorSets();

  //
  // Render
  //
#if PRINT_TRACE_TIMES
  auto start_time = timer_start();
#endif
  device.getCommandList()
      .traceRays(*pipeline, *sbt, render_width, render_height, 1)
      .flush("trace");
#if PRINT_TRACE_TIMES
  std::cout << "render time: " << timer_stop_microseconds(start_time) << " us"
            << std::endl;
#endif

  // Compare to golden image
  // Golden image generated on 2080ti with 535 drivers
  // Currently only RTX cards support ray query
  if (any_bits_set(device.getCapabilityBits() & DeviceCapabilityBits::kRayQuery)) {
    EXPECT_TRUE(this->golden_image_->compare(*output_texture, "rt_aabb_test"));
  } else {
    // Compute ray tracing does not match
    LOG(ERROR) << "compute raytracing detected, using tolerance";
    EXPECT_TRUE(
        this->golden_image_->compare(*output_texture, "rt_aabb_test", false, 7, 1));
  }

  // Cleanup resources
  resource_mgr.destroyPipeline(std::move(pipeline));
  resource_mgr.destroyAccelerationStructure(std::move(procedural_blas));
  resource_mgr.destroyAccelerationStructure(std::move(mesh_blas));
  resource_mgr.destroyAccelerationStructure(std::move(tlas));
  resource_mgr.destroyBuffer(std::move(mesh_vbo));
  resource_mgr.destroyBuffer(std::move(mesh_ibo));
  resource_mgr.destroyBuffer(std::move(instances_buffer));
  resource_mgr.destroyBuffer(std::move(bounds_buffer));
  resource_mgr.destroyBuffer(std::move(spheres_buffer));
  resource_mgr.destroyBuffer(std::move(projection_ubo));
  resource_mgr.destroyBuffer(std::move(light_data_ssbo));
  resource_mgr.destroyBuffer(std::move(material_data_ssbo));
  resource_mgr.destroyTexture(std::move(output_texture));
}

//
// BLASMultiMeshTest
//
TEST_F(RaytracingTest, BLASMultiMeshTest) {
  static constexpr std::string_view kTestName = "BLASMultiMeshTest";

  auto [device_ptr, output_texture] =
      this->getDeviceAndOutput(kTestName, render_width, render_height);
  if (!device_ptr) {
    GTEST_SKIP() << "Could not find raytracing capable device";
  }
  auto& device = *device_ptr;

  auto& resource_mgr = device.getResourceManager();
  // Generate shader spir-v
  auto const& shader_mgr = this->gfx_context_->getShaderManager();

  auto builders =
      shader_mgr.createBuilderVector({{"RaytracingTest/BlasMultiMesh.raygen"},
                                      {"RaytracingTest/BlasMultiMesh.miss"},
                                      {"RaytracingTest/BlasMultiMesh.closest"},
                                      {"Shading/rtShadowsSimple.miss"}});
  builders[0]->setExternalUniformBuffers({"PROJECTION_UBO_TYPE"});

  auto caches = shader_mgr.createCacheVector(std::move(builders));
  ASSERT_TRUE(validate_shader_caches(caches));

  // Material
  auto material = resource_mgr.createMaterial(kTestName, caches, true);

  //
  // Geometry - meshes
  //

  constexpr auto kAccelInputBits = BufferUsageBits::kDeviceAddressBit |
                                   BufferUsageBits::kAccelerationStructureReadOnlyBit |
                                   BufferUsageBits::kStorageBufferBit;

  // Build a fractal noise driven heightfield mesh
  auto height_mesh =
      build_heightfield(2.f, 2.f, 5, 5, [](float u, float v) { return 0.0f; });
  auto sphere_mesh = build_uv_sphere(40, 40, true);

  // Create a cube with its own vbo and ibo
  auto box_mesh = build_cube(true);
  auto [box_mesh_layout, box_mesh_vbo, box_mesh_ibo] =
      box_mesh->buildResources(resource_mgr, "box", kAccelInputBits);

  //
  // BLAS
  //

  // Create acceleration structure builder
  auto accel_builder = resource_mgr.createAccelerationStructureBuilder();
  std::vector<MeshObjectDesc> obj_descs;

  // We'll populate 2 separate mesh assemblies and add them both to the
  // same BLAS. Additional meshes could be added without using an assembly
  // but we'd have to manage the secondary transform buffers
  std::vector<BLASMeshAssembly> mesh_assembly(2);
  std::vector<glm::vec3> mesh_colors;

  //
  // Assembly 1
  //

  // Use undisplaced heightfield as a ground plane
  mesh_assembly[0].addMesh(*height_mesh, glm::mat4{1});
  mesh_colors.emplace_back(0.8f);

  // Sphere
  auto mesh_tm = glm::translate(glm::mat4{1}, glm::vec3(-0.6f, -0.4f, 0.2f));
  mesh_tm = glm::scale(mesh_tm, glm::vec3{0.4f});
  mesh_tm = glm::rotate(mesh_tm, 0.0f, glm::vec3{0.0f, 1.0f, 0.0f});
  mesh_assembly[0].addMesh(*sphere_mesh, mesh_tm);
  mesh_colors.emplace_back(0.8f, 0.5f, 0.4f);

  // Boxes
  // Add mesh with pre-existing vbo and ibo
  auto box_vbo_address = box_mesh_vbo->getDeviceAddress();
  auto box_ibo_address = box_mesh_ibo->getDeviceAddress();

  mesh_tm = glm::translate(glm::mat4{1}, glm::vec3{0.6f, -0.2f, 0.2f});
  mesh_tm = glm::rotate(mesh_tm, -0.6f, glm::vec3{0.0f, 1.0f, 0.0f});
  mesh_tm = glm::scale(mesh_tm, glm::vec3{0.1f});
  mesh_assembly[0].addMesh(*box_mesh, box_vbo_address, box_ibo_address, mesh_tm);
  mesh_colors.emplace_back(0.2f, 0.7f, 0.3f);

  // Add a second one (instancing into the BLAS)
  mesh_tm = glm::translate(glm::mat4{1}, glm::vec3{0.2f, -0.175f, -0.2f});
  mesh_tm = glm::rotate(mesh_tm, 0.5f, glm::vec3{0.0f, 1.0f, 0.0f});
  mesh_tm = glm::scale(mesh_tm, glm::vec3{0.15f});
  mesh_assembly[0].addMesh(*box_mesh, box_vbo_address, box_ibo_address, mesh_tm);
  mesh_colors.emplace_back(0.5f, 0.7f, 0.2f);

  // Build the mesh assembly buffers, which will also add the objects to the
  // MeshObjectDesc vector
  std::vector<BLASMeshAssembly::BuildResult> mesh_assembly_result(2);
  mesh_assembly_result[0] =
      mesh_assembly[0].build(resource_mgr, *accel_builder, obj_descs);

  //
  // Assembly 2
  //

  // Sphere
  mesh_tm = glm::translate(glm::mat4{1}, glm::vec3{0.3f, -0.6f, 0.0f});
  mesh_tm = glm::scale(mesh_tm, glm::vec3{0.15f});
  mesh_assembly[1].addMesh(*sphere_mesh, mesh_tm);
  mesh_colors.emplace_back(0.2f, 0.2f, 0.5f);

  // Box
  mesh_tm = glm::translate(glm::mat4{1}, glm::vec3{0.8f, -0.6f, 0.0f});
  mesh_tm = glm::rotate(mesh_tm, -0.4f, glm::vec3{1.0f, 0.0f, 0.0f});
  mesh_tm = glm::scale(mesh_tm, glm::vec3{0.075f});
  mesh_assembly[1].addMesh(*box_mesh, box_vbo_address, box_ibo_address, mesh_tm);
  mesh_colors.emplace_back(0.7f, 0.7f, 0.1f);

  mesh_assembly_result[1] =
      mesh_assembly[1].build(resource_mgr, *accel_builder, obj_descs);

  // Make sure we have a color for each mesh!
  EXPECT_EQ(obj_descs.size(), mesh_colors.size());

  // Build MeshObjectDesc buffer, which stores the device addresses of the
  // mesh components
  auto obj_desc_buffer =
      resource_mgr.createBuffer("Obj info",
                                {BufferType::kUnspecified,
                                 sizeof(MeshObjectDesc) * obj_descs.size(),
                                 BufferUsageBits::kDeviceAddressBit});
  obj_desc_buffer->updateSubData(
      obj_descs.data(), sizeof(MeshObjectDesc) * obj_descs.size(), 0);

  auto colors_buffer =
      resource_mgr.createBuffer("colors",
                                {BufferType::kUnspecified,
                                 sizeof(MeshObjectDesc) * obj_descs.size(),
                                 BufferUsageBits::kDeviceAddressBit});
  colors_buffer->updateSubData(
      mesh_colors.data(), sizeof(glm::vec3) * mesh_colors.size(), 0);

  resource_ptr<AccelerationStructure> mesh_blas;
  ASSERT_NO_THROW(mesh_blas = resource_mgr.createBottomLevelAccelerationStructure(
                      "Mesh BLAS", *accel_builder));

  //
  // TLAS
  //
  // Mesh instances
  auto tm = glm::scale(glm::mat4{1}, glm::vec3{1.25f});

  ASSERT_NO_THROW(
      accel_builder->addInstanceData(tm, mesh_blas->getDeviceAddress(), 0xff, 0, 0));

  // Create required instances buffer
  auto instances_buffer =
      resource_mgr.createBuffer("instance data",
                                {BufferType::kUnspecified,
                                 accel_builder->getInstancesBufferSize(),
                                 kAccelInputBits});

  // Top level acceleration structure
  resource_ptr<AccelerationStructure> tlas;
  ASSERT_NO_THROW(tlas = resource_mgr.createTopLevelAccelerationStructure(
                      "Test TLAS", *accel_builder, *instances_buffer));

  //
  // Pipeline and SBT
  //
  struct {
    DeviceAddress obj_descs;
    DeviceAddress colors;
  } push_constants;
  push_constants.obj_descs = obj_desc_buffer->getDeviceAddress();
  push_constants.colors = colors_buffer->getDeviceAddress();

  auto pipeline = resource_mgr.createRaytracingPipeline(
      kTestName,
      *material,
      {PushConstantRange{ShaderStageBits::kClosestHit, 0u, sizeof(push_constants)}});
  pipeline->create(2);

  auto sbt = resource_mgr.createShaderBindingTable(*pipeline);

  //
  // Scene components
  //

  // Camera
  Camera camera(Camera::Projection::kPerspective, Camera::Type::kLookAt);
  camera.setPerspectiveParams(
      60.0f, (float)render_width / (float)render_height, 0.1f, 512.0f);
  camera.setPosition({0.1f, -1.f, 2.5f});
  camera.setTarget(glm::vec3{0.0f});

  auto projection_ubo = resource_mgr.createBuffer("Projection UBO",
                                                  {BufferType::kUnspecified,
                                                   Camera::getBufferDataSize(),
                                                   BufferUsageBits::kUniformBufferBit,
                                                   BufferAccessType::kHostVisible});

  camera.updateBufferData(*projection_ubo);

  // Lights
  LightManager light_mgr;
  light_mgr.addPointLight({-20, -30, 20}, {0.8f, 0.75f, 0.7f}, 1.0f, false);
  light_mgr.addPointLight({5, -2, 1}, {0.3, 0.5, 0.7}, 1.0f, false);

  auto light_data_ssbo = light_mgr.createBuffer(resource_mgr);
  light_mgr.updateBufferData(*light_data_ssbo);
  light_mgr.bindToMaterial(*material, *light_data_ssbo, 0.01f);

  // PBRMaterial properties
  PBRMaterialManager material_mgr;
  material_mgr.addCookTorranceMaterial(
      PBRMaterialDescriptor::ColorSource::kConstant, glm::vec4{0.f}, 1.0f, 0.25f, 0.6f);
  auto material_data_ssbo = material_mgr.createBuffer(resource_mgr);
  material_mgr.updateBufferData(*material_data_ssbo);
  material_mgr.bindToMaterial(*material, *material_data_ssbo);

  // Update Material descriptors
  material->setAccelerationStructureAttribute("topLevelAS", *tlas);
  material->setImageLoadStoreAttribute("outputImage", *output_texture);
  material->bindExternalUniformBufferToBlock("PROJECTION_UBO_TYPE", *projection_ubo);

  material->updateDescriptorSets();

  //
  // Render
  //
  device.getCommandList()
      .setPushConstants(*pipeline,
                        "push constants",
                        ShaderStageBits::kClosestHit,
                        &push_constants,
                        sizeof(push_constants))
      .traceRays(*pipeline, *sbt, render_width, render_height, 1)
      .flush("trace");

  // Golden image generated with 520 drivers
  if (any_bits_set(device.getCapabilityBits() & DeviceCapabilityBits::kRayQuery)) {
    EXPECT_TRUE(
        this->golden_image_->compare(*output_texture, "rt_blas_multi_mesh", false, 6, 1));
  } else {
    // Running with compute emulation (1080s)
    LOG(ERROR) << "compute raytracing detected, using tolerance";
    EXPECT_TRUE(this->golden_image_->compare(
        *output_texture, "rt_blas_multi_mesh", false, 20355, 21));
  }

  resource_mgr.destroyPipeline(std::move(pipeline));
  resource_mgr.destroyAccelerationStructure(std::move(mesh_blas));
  resource_mgr.destroyAccelerationStructure(std::move(tlas));
  resource_mgr.destroyBuffer(std::move(box_mesh_vbo));
  resource_mgr.destroyBuffer(std::move(box_mesh_ibo));
  mesh_assembly_result[0].destroyBuffers(resource_mgr);
  mesh_assembly_result[1].destroyBuffers(resource_mgr);
  resource_mgr.destroyBuffer(std::move(obj_desc_buffer));
  resource_mgr.destroyBuffer(std::move(colors_buffer));
  resource_mgr.destroyBuffer(std::move(instances_buffer));
  resource_mgr.destroyBuffer(std::move(projection_ubo));
  resource_mgr.destroyBuffer(std::move(light_data_ssbo));
  resource_mgr.destroyBuffer(std::move(material_data_ssbo));
  resource_mgr.destroyTexture(std::move(output_texture));
}

}  // namespace GfxDriverTests

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
