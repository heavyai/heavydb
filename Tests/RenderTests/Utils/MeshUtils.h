/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/VertexBuffer.h"

namespace gfx {

//
// mat4x4
//
// DEPRECATED: use glm::mat4x4 instead
//
struct mat4x4 {
  float m[4][4];
  // Generate a perspective matrix intended for reversed depth testing (kGreaterOrEqual)
  // w and h are viewport width and height to compute aspect ratio
  // fov is in degrees
  // near and far are the world z near and far clipping distances (positive values)
  // camera is looking down -z axis
  void perspective(uint32_t w, uint32_t h, float fov, float near, float far) {
    constexpr double pi_over_180 = 3.14159265358979323846 / 180.0;
    zero();
    float s = static_cast<float>(w) / static_cast<float>(h);
    float g = 1.0f / std::tan(fov * 0.5 * pi_over_180);
    float A = near / (far - near);
    float B = far * near / (far - near);

    m[0][0] = g / s;
    m[1][1] = g;
    m[2][2] = A;
    m[3][2] = B;
    m[2][3] = -1.0f;
  }
  void identity() {
    zero();
    m[0][0] = 1.0f;
    m[1][1] = 1.0f;
    m[2][2] = 1.0f;
    m[3][3] = 1.0f;
  }
  void zero() { std::memset(m, 0, 16 * sizeof(float)); }
};

//
// Mesh class interface
// Able to build vertex and index buffers with arbitrary number of float attributes
//
// TODO:
//  - Support for different attribute types and BufferLayout
//  - Dynamic attribute count
class Mesh {
 public:
  using Attribute = std::pair<std::string, BufferAttrType>;
  using VertexResources = std::tuple<InterleavedBufferLayoutShPtr, BufferWrapperUqPtr>;
  using IndexedResources =
      std::tuple<InterleavedBufferLayoutShPtr, BufferWrapperUqPtr, BufferWrapperUqPtr>;

  virtual ~Mesh() = default;

  virtual const uint32_t getNumAttrs() const = 0;

  // Reserve storage for vertices (resizes storage but does not zero)
  virtual void reserveVertices(uint32_t num_vertices) = 0;
  // Get raw vertex data, used to update the vertex buffer
  virtual float* vertexData() const = 0;
  // Get the size of the vertex data in bytes
  virtual const size_t vertexDataSize() const = 0;

  virtual const std::vector<uint32_t>& getIndices() const = 0;
  virtual const std::vector<Attribute>& getAttributes() const = 0;

  // Clear all mesh data (vertices, indices, attributes)
  virtual void clear() = 0;

  // Get stats
  virtual uint32_t numVertices() const = 0;
  virtual uint32_t numIndices() const = 0;
  virtual uint32_t numTriangles() const = 0;

  // Add a single vertex
  virtual void addVertex(const std::vector<float>& v) = 0;
  // Add an array of vertices
  // returns {offset of first vertex, number of vertices added}
  virtual std::pair<uint32_t, uint32_t> addVertices(
      const std::vector<float>& vertices) = 0;

  // Reserve storage for indices (resizes storage but does not zero)
  virtual void reserveIndices(uint32_t num_indices) = 0;
  // Get raw index data pointer
  virtual uint32_t* indexData() const = 0;
  // Get size of index data in bytes
  virtual size_t indexDataSize() const = 0;

  // Add a single index
  virtual void addIndex(uint32_t i) = 0;
  // Replace existing indices with the passed in vector
  virtual void setIndices(std::vector<uint32_t> indices) = 0;

  // Add a vector of attributes to the attributes vector
  virtual void addAttributes(const std::vector<Attribute>& attrs) = 0;

  // Add a triangle to the indices vector
  virtual void addTriangle(uint32_t a, uint32_t b, uint32_t c) = 0;
  // Add a quad (2 triangles) to the indices vector
  virtual void addQuad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) = 0;

  // Append another Mesh with the same attribute count to this Mesh
  // The appended mesh can be offset and uniformly scaled during merging
  // position_offsets are applied to the first N attributes, with N being
  // the number of position attributes. For example a 3D mesh should have
  // the x, y, and z position attributes first, and position_offsets will
  // be of length 3.
  virtual void appendMesh(const Mesh& src,
                          const std::vector<float>& position_offsets,
                          const float scale) = 0;

  virtual InterleavedBufferLayoutShPtr buildVertexBufferLayout() const = 0;

  // Build the vertex buffer only
  virtual VertexResources buildVertexResources(
      ResourceManager& resource_mgr,
      const std::string& name,
      BufferUsageBits extra_usage_bits = BufferUsageBits::kNone) const = 0;

  // Build vertex buffer and index buffer
  virtual IndexedResources buildResources(
      ResourceManager& resource_mgr,
      const std::string& name,
      BufferUsageBits extra_usage_bits = BufferUsageBits::kNone) const = 0;
};

template <uint32_t NUM_ATTRS>
std::unique_ptr<Mesh> make_mesh();

// Data required to draw the Mesh given a compatible Pipeline
struct MeshDrawData {
  gfx::InterleavedBufferLayoutShPtr layout;

  // Buffer resources (must be explicitly destroyed)
  gfx::BufferWrapperUqPtr vbo_wrapper;
  gfx::BufferWrapperUqPtr ibo_wrapper;
  // Convenience pointers
  gfx::VertexBuffer* vbo{nullptr};
  gfx::IndexBuffer* ibo{nullptr};

  gfx::PrimitiveAssemblyUqPtr pa;
  uint32_t num_indices{0};

  void destroyResources(ResourceManager& resource_mgr) {
    if (vbo_wrapper) {
      resource_mgr.destroyBuffer(std::move(vbo_wrapper));
    }
    if (ibo_wrapper) {
      resource_mgr.destroyBuffer(std::move(ibo_wrapper));
    }
  }
};

// Generate all resources required to render mesh using resource_mgr
// name will serve as a base for Resource names
// material_attr_names pairs vertex attributes with Mesh attributes in PrimitiveAssembly
// reference_material MUST consume the named vertex attributes
// Caller takes ownership of resources and MUST destroy vbo_wrapper and ibo_wrapper
MeshDrawData build_mesh_draw_data(ResourceManager& resource_mgr,
                                  const Mesh& mesh,
                                  const std::string& name,
                                  const std::vector<std::string>& material_attr_names,
                                  Material& reference_material);

// UV style sphere mesh builder
std::unique_ptr<Mesh> build_uv_sphere(uint32_t meridians,
                                      uint32_t parallels,
                                      bool build_normals = false);

// Cube mesh builder
std::unique_ptr<Mesh> build_cube(bool build_normals = false);

// Heightfield builder
using HeightFieldCB = std::function<float(float u, float v)>;
std::unique_ptr<Mesh> build_heightfield(
    float u_scale,
    float v_scale,
    uint32_t u_segments,
    uint32_t v_segments,
    HeightFieldCB compute_height_callback);  // always builds normals

//
// MeshObjectDesc
//
struct MeshObjectDesc {
  DeviceAddress vbo_address;
  DeviceAddress ibo_address;
  DeviceAddress model_transform_address;
  DeviceAddress normal_transform_address;

  MeshObjectDesc(DeviceAddress vbo_address,
                 DeviceAddress ibo_address,
                 DeviceAddress model_transform_address,
                 DeviceAddress normal_transform_address)
      : vbo_address{vbo_address}
      , ibo_address{ibo_address}
      , model_transform_address{model_transform_address}
      , normal_transform_address{normal_transform_address} {}
};

//
// BLASMeshAssembly
//
// Helper class for building a single BLAS with multiple meshes
// Multiple BLASMeshAssemblies can be added to the same BLAS
// Meshes must have the same attribute structure as vertex stride is fixed
// Meshes can provide a vertex and index buffer
//   Meshes without vertex and index buffers will be added to shared buffers
// Buffers will be created for:
//   - BLAS transform (3x4 row major)
//   - Model transform (4x4 column major version of BLAS transform)
//   - Normal transform (3x3 inverse transpose of model)
class BLASMeshAssembly {
 public:
  void addMesh(const Mesh& mesh, const glm::mat4& transform);
  void addMesh(const Mesh& mesh,
               DeviceAddress vbo_address,
               DeviceAddress ibo_address,
               const glm::mat4& model_transform);

  struct BuildResult {
    BufferWrapperUqPtr shared_vbo;
    BufferWrapperUqPtr shared_ibo;
    BufferWrapperUqPtr blas_transform_buffer;
    BufferWrapperUqPtr model_transform_buffer;
    BufferWrapperUqPtr normal_transform_buffer;

    void destroyBuffers(ResourceManager& resource_mgr);
  };

  BuildResult build(ResourceManager& resource_mgr,
                    AccelerationStructure::Builder& accel_builder,
                    std::vector<MeshObjectDesc>& obj_descs) const;
  void clear();

 private:
  struct MeshInfo {
    const Mesh* mesh{nullptr};
    DeviceAddress vbo_address{0};
    DeviceAddress ibo_address{0};
    MeshInfo(const Mesh* mesh, DeviceAddress vbo_address, DeviceAddress ibo_address)
        : mesh{mesh}, vbo_address{vbo_address}, ibo_address{ibo_address} {}
  };
  std::vector<MeshInfo> mesh_infos_;
  std::vector<glm::mat3x4> blas_transforms_;  // row major (vulkan)
  std::vector<glm::mat4> model_transforms_;   // column major (glsl)
  std::vector<glm::mat3> normal_transforms_;

  void addTransform(const glm::mat4& transform);
};

}  // namespace gfx
