/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/Utils/MeshUtils.h"

#include <cmath>
#include <iostream>

#include <tbb/parallel_for.h>

#include "Shared/measure.h"

namespace gfx {

// MeshImpl
template <uint32_t NUM_ATTRS>
class MeshImpl : public Mesh {
 public:
  using Vertex = std::array<float, NUM_ATTRS>;
  using Attribute = std::pair<std::string, BufferAttrType>;
  using VertexResources = std::tuple<InterleavedBufferLayoutShPtr, BufferWrapperUqPtr>;
  using IndexedResources =
      std::tuple<InterleavedBufferLayoutShPtr, BufferWrapperUqPtr, BufferWrapperUqPtr>;

  ~MeshImpl() override = default;

  void reserveVertices(uint32_t num_vertices) override {
    vertices_.resize(vertices_.size() + num_vertices);
  }
  float* vertexData() const override {
    return const_cast<float*>(reinterpret_cast<const float*>(vertices_.data()));
  }
  const size_t vertexDataSize() const override {
    return vertices_.size() * NUM_ATTRS * sizeof(float);
  }
  const std::vector<uint32_t>& getIndices() const override { return indices_; }
  const std::vector<Attribute>& getAttributes() const override { return attrs_; }
  const uint32_t getNumAttrs() const override { return NUM_ATTRS; }

  void clear() override {
    vertices_.clear();
    indices_.clear();
    attrs_.clear();
  }

  uint32_t numVertices() const override { return vertices_.size(); }
  uint32_t numIndices() const override { return indices_.size(); }
  uint32_t numTriangles() const override { return indices_.size() / 3; }

  void addVertex(const std::vector<float>& v) override {
    CHECK_EQ(v.size(), NUM_ATTRS);
    vertices_.emplace_back(Vertex{});
    std::copy(v.begin(), v.end(), vertices_.back().begin());
  }

  // returns {number of vertices added, offset of first vertex}
  std::pair<uint32_t, uint32_t> addVertices(const std::vector<float>& vertices) override {
    uint32_t first_vertex = vertices_.size();
    auto num_values = vertices.size();
    CHECK_EQ(num_values % NUM_ATTRS, 0u);

    for (uint32_t value_index = 0; value_index < num_values; value_index += NUM_ATTRS) {
      Vertex vertex;
      for (uint32_t attr_index = 0; attr_index < NUM_ATTRS; ++attr_index) {
        vertex[attr_index] = vertices[value_index + attr_index];
      }
      vertices_.emplace_back(vertex);
    }
    return {num_values / NUM_ATTRS, first_vertex};
  }

  void reserveIndices(uint32_t num_indices) override {
    indices_.resize(indices_.size() + num_indices);
  }
  uint32_t* indexData() const override { return const_cast<uint32_t*>(indices_.data()); }
  size_t indexDataSize() const override { return sizeof(uint32_t) * indices_.size(); }
  void addIndex(uint32_t i) override { indices_.emplace_back(i); }

  void setIndices(std::vector<uint32_t> indices) override {
    indices_ = std::move(indices);
  }

  void addTriangle(uint32_t a, uint32_t b, uint32_t c) override {
    indices_.emplace_back(a);
    indices_.emplace_back(b);
    indices_.emplace_back(c);
  }

  void addQuad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) override {
    indices_.emplace_back(a);
    indices_.emplace_back(b);
    indices_.emplace_back(c);
    indices_.emplace_back(a);
    indices_.emplace_back(c);
    indices_.emplace_back(d);
  }

  void addAttributes(const std::vector<Attribute>& attrs) override {
    for (const auto& attr : attrs) {
      attrs_.push_back(attr);
    }
    // TODO: better attribute management, including checking that the
    // float attributes match NUM_ATTRS
    // TODO: More explicit handling of position vs other attrs
  }

  void appendMesh(const Mesh& src,
                  const std::vector<float>& position_offsets,
                  const float scale) override {
    CHECK_EQ(src.getNumAttrs(), NUM_ATTRS);
    CHECK(static_cast<const MeshImpl&>(src).attrs_ == attrs_);
    uint32_t num_position_components = position_offsets.size();
    CHECK_LE(num_position_components, NUM_ATTRS);

    // append vertices
    uint32_t start_vertex = vertices_.size();
    vertices_.resize(vertices_.size() + src.numVertices());
    auto* src_vertices = src.vertexData();
    for (uint32_t v = 0; v < src.numVertices(); ++v) {
      auto const* sv = &src_vertices[v * NUM_ATTRS];
      auto& dv = vertices_[v + start_vertex];
      for (uint32_t pi = 0; pi < NUM_ATTRS; ++pi) {
        if (pi < num_position_components) {
          dv[pi] = sv[pi] * scale + position_offsets[pi];
        } else {
          dv[pi] = sv[pi];
        }
      }
    }

    // append indices
    auto const& src_indices = src.getIndices();
    uint32_t start_index = indices_.size();
    indices_.resize(indices_.size() + src.numIndices());
    for (uint32_t i = 0; i < src.numIndices(); ++i) {
      indices_[i + start_index] = src_indices[i] + start_vertex;
    }
  }

  InterleavedBufferLayoutShPtr buildVertexBufferLayout() const override {
    auto buffer_layout = std::make_shared<InterleavedBufferLayout>();
    CHECK(buffer_layout != nullptr);
    for (auto const& [name, type] : attrs_) {
      buffer_layout->addAttribute(name, type);
    }
    return buffer_layout;
  }

  VertexResources buildVertexResources(ResourceManager& resource_mgr,
                                       const std::string& name,
                                       BufferUsageBits extra_usage_bits) const override {
    auto vertex_buffer_wrapper =
        resource_mgr.createBuffer(name + " VBO",
                                  {BufferType::kVertexBuffer,
                                   vertexDataSize(),
                                   BufferUsageBits::kLayoutBufferBit | extra_usage_bits});

    CHECK(vertex_buffer_wrapper != nullptr);
    auto buffer_layout = buildVertexBufferLayout();

    vertex_buffer_wrapper->updateSubDataWithLayout(
        vertexData(), vertexDataSize(), 0, buffer_layout);

    return {std::move(buffer_layout), std::move(vertex_buffer_wrapper)};
  }

  IndexedResources buildResources(ResourceManager& resource_mgr,
                                  const std::string& name,
                                  BufferUsageBits extra_usage_bits) const override {
    auto [buffer_layout, vertex_buffer_wrapper] =
        buildVertexResources(resource_mgr, name, extra_usage_bits);
    auto index_buffer_wrapper =
        resource_mgr.createBuffer(name + " IBO",
                                  {BufferType::kIndexBuffer,
                                   this->indices_.size() * sizeof(uint32_t),
                                   extra_usage_bits});
    index_buffer_wrapper->updateSubData(
        this->indices_.data(), index_buffer_wrapper->getNumBytes(), 0);

    return {std::move(buffer_layout),
            std::move(vertex_buffer_wrapper),
            std::move(index_buffer_wrapper)};
  }

 private:
  std::vector<Vertex> vertices_;
  std::vector<uint32_t> indices_;
  std::vector<Attribute> attrs_;
};

template <>
std::unique_ptr<Mesh> make_mesh<2>() {
  return std::make_unique<MeshImpl<2>>();
}

template <>
std::unique_ptr<Mesh> make_mesh<3>() {
  return std::make_unique<MeshImpl<3>>();
}

template <>
std::unique_ptr<Mesh> make_mesh<6>() {
  return std::make_unique<MeshImpl<6>>();
}

std::unique_ptr<Mesh> build_uv_sphere(uint32_t meridians,
                                      uint32_t parallels,
                                      bool build_normals) {
  constexpr double kPI = 3.14159265358979323846;

  std::unique_ptr<Mesh> mesh;
  if (build_normals) {
    mesh = make_mesh<6>();
  } else {
    mesh = make_mesh<3>();
  }

  if (build_normals) {
    mesh->addVertex({0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f});
  } else {
    mesh->addVertex({0.0f, 1.0f, 0.0f});
  }

  for (uint32_t j = 0; j < parallels - 1; ++j) {
    double polar = kPI * double(j + 1) / double(parallels);
    double sp = std::sin(polar);
    double cp = std::cos(polar);
    for (uint32_t i = 0; i < meridians; ++i) {
      double azimuth = 2.0 * kPI * double(i) / double(meridians);
      double sa = std::sin(azimuth);
      double ca = std::cos(azimuth);
      double x = sp * ca;
      double y = cp;
      double z = sp * sa;
      if (build_normals) {
        mesh->addVertex({(float)x, (float)y, (float)z, (float)x, (float)y, (float)z});
      } else {
        mesh->addVertex({(float)x, (float)y, (float)z});
      }
    }
  }

  if (build_normals) {
    mesh->addVertex({0.0f, -1.0f, 0.0f, 0.0f, -1.0, 0.0f});
  } else {
    mesh->addVertex({0.0f, -1.0f, 0.0f});
  }

  // cap
  for (uint32_t i = 0; i < meridians; ++i) {
    uint32_t a = i + 1;
    uint32_t b = (i + 1) % meridians + 1;
    mesh->addTriangle(b, a, 0);
  }

  // body
  for (uint32_t j = 0; j < parallels - 2; ++j) {
    uint32_t a_start = j * meridians + 1;
    uint32_t b_start = (j + 1) * meridians + 1;
    for (uint32_t i = 0; i < meridians; ++i) {
      uint32_t a = a_start + i;
      uint32_t a1 = a_start + (i + 1) % meridians;
      uint32_t b = b_start + i;
      uint32_t b1 = b_start + (i + 1) % meridians;
      mesh->addQuad(a, a1, b1, b);
    }
  }

  // cap
  for (uint32_t i = 0; i < meridians; ++i) {
    uint32_t a = i + meridians * (parallels - 2) + 1;
    uint32_t b = (i + 1) % meridians + meridians * (parallels - 2) + 1;
    mesh->addTriangle(mesh->numVertices() - 1, a, b);
  }

  mesh->addAttributes({{"in_position", BufferAttrType::kVec3f}});
  if (build_normals) {
    mesh->addAttributes({{"in_normal", BufferAttrType::kVec3f}});
  }

  return mesh;
}

std::unique_ptr<Mesh> build_cube(bool build_normals) {
  if (build_normals) {
    // clang-format off
    static std::vector<float> vertices = {
      // -z plane
      -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
      -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
       1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
       1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
      // +z plane
      -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
      -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
       1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
       1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
      // -x plane
      -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
      -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
      -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f,
      -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f,
      // +x plane
       1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
       1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
       1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,
       1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f,
      // -y plane
      -1.0f, -1.0f, -1.0f, 0.0f,  -1.0f,  0.0f,
       1.0f, -1.0f, -1.0f, 0.0f,  -1.0f,  0.0f,
      -1.0f, -1.0f,  1.0f, 0.0f,  -1.0f,  0.0f,
       1.0f, -1.0f,  1.0f, 0.0f,  -1.0f,  0.0f,
      // +y plane
      -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f,
       1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f,
      -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
       1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    };

    static std::vector<uint32_t> indices = {0, 1, 2, 2, 1, 3,
                                            6, 5, 4, 7, 5, 6,
                                            10, 9, 8, 11, 9, 10,
                                            12, 13, 14, 14, 13, 15,
                                            16, 17, 18, 18, 17, 19,
                                            22, 21, 20, 23, 21, 22};

    // clang-format on
    auto mesh_ptr = std::make_unique<MeshImpl<6>>();
    mesh_ptr->addVertices(vertices);
    mesh_ptr->setIndices(indices);
    mesh_ptr->addAttributes(
        {{"in_position", BufferAttrType::kVec3f}, {"in_normal", BufferAttrType::kVec3f}});
    return mesh_ptr;
  } else {
    // clang-format off
    static std::vector<float> vertices = {
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f};

    static std::vector<uint32_t> indices = {0, 1, 2, 2, 1, 3,
                                            2, 3, 4, 4, 3, 5,
                                            4, 5, 6, 6, 5, 7,
                                            6, 7, 0, 0, 7, 1,
                                            6, 0, 2, 2, 4, 6,
                                            7, 5, 3, 7, 3, 1};
    // clang-format on
    auto mesh_ptr = std::make_unique<MeshImpl<3>>();
    mesh_ptr->addVertices(vertices);
    mesh_ptr->setIndices(indices);
    mesh_ptr->addAttributes({{"in_position", BufferAttrType::kVec3f}});
    return mesh_ptr;
  }
}

#define PRINT_HEIGHTFIELD_TIMING false
// Use if constexpr to ensure that log messages still compile, but are elided
#define HF_TIMING_PRINT()                  \
  if constexpr (!PRINT_HEIGHTFIELD_TIMING) \
    ;                                      \
  else                                     \
    std::cout

std::unique_ptr<Mesh> build_heightfield(float u_scale,
                                        float v_scale,
                                        uint32_t u_segments,
                                        uint32_t v_segments,
                                        HeightFieldCB compute_height_callback) {
  CHECK_GT(u_segments, 0u);
  CHECK_GT(v_segments, 0u);
  auto mesh = std::make_unique<MeshImpl<6>>();
  auto num_attrs = mesh->getNumAttrs();
  uint32_t num_verts = (u_segments + 1) * (v_segments + 1);
  uint32_t num_faces = u_segments * v_segments * 2;
  uint32_t num_indices = num_faces * 3;

  // Generate vertices, calling compute_height_callback for each
  auto start_time = timer_start();
  mesh->reserveVertices(num_verts);
  auto* verts_ptr = mesh->vertexData();

  float u_step = u_scale / static_cast<float>(u_segments);
  float v_step = v_scale / static_cast<float>(v_segments);
  float u_start = -u_scale / 2.0f;
  float v_start = -v_scale / 2.0f;

  tbb::parallel_for(tbb::blocked_range<uint32_t>(0, v_segments + 1),
                    [&](const tbb::blocked_range<uint32_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      auto* vp = verts_ptr + (num_attrs * start_idx * (u_segments + 1));
                      for (uint32_t vs = start_idx; vs < end_idx; ++vs) {
                        float v = v_start + v_step * static_cast<float>(vs);
                        for (uint32_t us = 0; us <= u_segments; ++us, vp += 6) {
                          float u = u_start + u_step * static_cast<float>(us);
                          float h = compute_height_callback(u, v);
                          vp[0] = u;
                          vp[1] = h;
                          vp[2] = v;
                          // Don't write normals
                        }
                      }
                    });

  auto time = timer_stop(start_time);
  HF_TIMING_PRINT() << "Vertex time: " << time << " ms" << std::endl;

  // Write face indices
  start_time = timer_start();
  mesh->reserveIndices(num_indices);
  auto* indices = mesh->indexData();
  const auto u_stride = u_segments + 1;
  tbb::parallel_for(tbb::blocked_range<uint32_t>(0, v_segments),
                    [&](const tbb::blocked_range<uint32_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      auto index = start_idx * u_segments * 6;
                      for (uint32_t vs = start_idx; vs < end_idx; ++vs) {
                        for (uint32_t us = 0; us < u_segments; ++us) {
                          indices[index++] = (vs + 1) * u_stride + us + 1;
                          indices[index++] = vs * u_stride + us;
                          indices[index++] = vs * u_stride + us + 1;
                          indices[index++] = (vs + 1) * u_stride + us + 1;
                          indices[index++] = (vs + 1) * u_stride + us;
                          indices[index++] = vs * u_stride + us;
                        }
                      }
                    });

  time = timer_stop(start_time);
  HF_TIMING_PRINT() << "Index time: " << time << " ms" << std::endl;

  // Compute face normals
  start_time = timer_start();
  std::vector<glm::vec3> face_normals(num_faces, glm::vec3{0});
  tbb::parallel_for(tbb::blocked_range<uint32_t>(0, num_faces),
                    [&](const tbb::blocked_range<uint32_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      for (uint32_t fi = start_idx; fi < end_idx; ++fi) {
                        const auto index = fi * 3;
                        const auto vi1 = indices[index];
                        const auto vi2 = indices[index + 1];
                        const auto vi3 = indices[index + 2];
                        auto* v1 = &verts_ptr[vi1 * num_attrs];
                        auto* v2 = &verts_ptr[vi2 * num_attrs];
                        auto* v3 = &verts_ptr[vi3 * num_attrs];
                        glm::vec3 a{v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
                        glm::vec3 b{v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};
                        face_normals[fi] = glm::normalize(
                            glm::cross(glm::normalize(a), glm::normalize(b)));
                      }
                    });

  // Accumulate face normals into vertices
  // single-threaded to ensure consistency (no float atomics in c++17 and a mutex just
  // makes it slow)
  std::vector<glm::vec3> normal_accum(num_verts, glm::vec3{0});
  for (uint32_t fi = 0; fi < num_faces; ++fi) {
    const auto index = fi * 3;
    const auto vi1 = indices[index];
    const auto vi2 = indices[index + 1];
    const auto vi3 = indices[index + 2];

    const auto& N = face_normals[fi];

    normal_accum[vi1] += N;
    normal_accum[vi2] += N;
    normal_accum[vi3] += N;
  }

  tbb::parallel_for(tbb::blocked_range<uint32_t>(0, num_verts),
                    [&](const tbb::blocked_range<uint32_t>& r) {
                      const auto start_idx = r.begin();
                      const auto end_idx = r.end();
                      for (uint32_t index = start_idx; index < end_idx; ++index) {
                        const auto N = glm::normalize(normal_accum[index]);
                        uint32_t vi = index * num_attrs;
                        verts_ptr[vi + 3] = N.x;
                        verts_ptr[vi + 4] = N.y;
                        verts_ptr[vi + 5] = N.z;
                      }
                    });

  time = timer_stop(start_time);
  HF_TIMING_PRINT() << "normals time: " << time << " ms" << std::endl;

  mesh->addAttributes(
      {{"in_position", BufferAttrType::kVec3f}, {"in_normal", BufferAttrType::kVec3f}});
  return mesh;
}

MeshDrawData build_mesh_draw_data(ResourceManager& resource_mgr,
                                  const Mesh& mesh,
                                  const std::string& name,
                                  const std::vector<std::string>& material_attr_names,
                                  Material& reference_material) {
  MeshDrawData mesh_data;

  std::tie(mesh_data.layout, mesh_data.vbo_wrapper, mesh_data.ibo_wrapper) =
      mesh.buildResources(resource_mgr, name);
  mesh_data.vbo = static_cast<VertexBuffer*>(mesh_data.vbo_wrapper.get());
  mesh_data.ibo = static_cast<IndexBuffer*>(mesh_data.ibo_wrapper.get());
  mesh_data.num_indices = mesh.numIndices();

  // Build mesh / material attribute name pairs
  auto const& mesh_attrs = mesh.getAttributes();
  uint32_t num_attrs = mesh_attrs.size();
  CHECK_EQ(num_attrs, material_attr_names.size());
  VboAttrToShaderAttrPairs attr_pairs;
  for (uint32_t i = 0; i < num_attrs; ++i) {
    attr_pairs.push_back({mesh_attrs[i].first, material_attr_names[i]});
  }

  PrimitiveAssemblyAttrInfo attr_info = {{mesh_data.vbo, mesh_data.layout},
                                         std::move(attr_pairs)};
  mesh_data.pa = resource_mgr.createPrimitiveAssembly(name,
                                                      PrimitiveTopology::kTriangleList,
                                                      reference_material,
                                                      attr_info,
                                                      mesh_data.ibo);

  return mesh_data;
}

//
// BLASMeshAssembly
//
void BLASMeshAssembly::clear() {
  mesh_infos_.clear();
  blas_transforms_.clear();
  model_transforms_.clear();
  normal_transforms_.clear();
}

void BLASMeshAssembly::addTransform(const glm::mat4& transform) {
  blas_transforms_.emplace_back(glm::transpose(transform));
  model_transforms_.emplace_back(transform);
  normal_transforms_.emplace_back(glm::transpose(glm::inverse(glm::mat3(transform))));
}

void BLASMeshAssembly::addMesh(const Mesh& mesh, const glm::mat4& transform) {
  mesh_infos_.emplace_back(&mesh, 0u, 0u);
  addTransform(transform);
}

void BLASMeshAssembly::addMesh(const Mesh& mesh,
                               DeviceAddress vbo_address,
                               DeviceAddress ibo_address,
                               const glm::mat4& transform) {
  mesh_infos_.emplace_back(&mesh, vbo_address, ibo_address);
  addTransform(transform);
}

BLASMeshAssembly::BuildResult BLASMeshAssembly::build(
    ResourceManager& resource_mgr,
    AccelerationStructure::Builder& accel_builder,
    std::vector<MeshObjectDesc>& obj_descs) const {
  CHECK(!mesh_infos_.empty());
  BuildResult build_result;
  auto layout = mesh_infos_[0].mesh->buildVertexBufferLayout();
  auto num_meshes = mesh_infos_.size();

  // Find meshes without vertex and index buffers and add them to
  // the shared buffer sizes
  uint64_t shared_vbo_size{0};
  uint64_t shared_ibo_size{0};
  for (auto const& info : mesh_infos_) {
    if (info.vbo_address == 0u) {
      shared_vbo_size += info.mesh->vertexDataSize();
      shared_ibo_size += info.mesh->indexDataSize();
    }
  }

  // Create the shared buffers
  constexpr auto kAccelInputBits = BufferUsageBits::kDeviceAddressBit |
                                   BufferUsageBits::kAccelerationStructureReadOnlyBit |
                                   BufferUsageBits::kStorageBufferBit |
                                   BufferUsageBits::kLayoutBufferBit;

  build_result.shared_vbo = resource_mgr.createBuffer(
      "shared VBO", {BufferType::kVertexBuffer, shared_vbo_size, kAccelInputBits});

  build_result.shared_ibo = resource_mgr.createBuffer(
      "shared IBO",
      BufferCreateInfo{BufferType::kIndexBuffer, shared_ibo_size, kAccelInputBits});

  // Create secondary buffers:
  //  - BLAS transform (places object into BLAS space). This is a 3x4 row major matrix
  //  - Model transform: 4x4 column major matrix equivalent to BLAS transform
  //    The BLAS transform is NOT provided as a built-in during shading
  //  - Normal transform: 3x3 inverse transpose of model matrix (for shading)
  auto create_buffer = [&resource_mgr](std::string_view name,
                                       const void* data,
                                       size_t num_bytes) -> BufferWrapperUqPtr {
    auto buffer = resource_mgr.createBuffer(
        name, {BufferType::kUnspecified, num_bytes, kAccelInputBits});
    buffer->updateSubData(data, num_bytes, 0);
    return buffer;
  };
  build_result.blas_transform_buffer =
      create_buffer("blas transforms",
                    blas_transforms_.data(),
                    blas_transforms_.size() * sizeof(glm::mat3x4));

  auto model_transform_size = num_meshes * sizeof(glm::mat4);
  build_result.model_transform_buffer =
      create_buffer("model transforms", model_transforms_.data(), model_transform_size);

  auto normal_transform_size = num_meshes * sizeof(glm::mat3);
  build_result.normal_transform_buffer = create_buffer(
      "normal transforms", normal_transforms_.data(), normal_transform_size);

  // Get base device addresses for each buffer
  auto shared_vbo_address = build_result.shared_vbo->getDeviceAddress();
  auto shared_ibo_address = build_result.shared_ibo->getDeviceAddress();
  auto blas_transform_address = build_result.blas_transform_buffer->getDeviceAddress();
  auto model_transform_address = build_result.model_transform_buffer->getDeviceAddress();
  auto normal_transform_address =
      build_result.normal_transform_buffer->getDeviceAddress();

  // Add objects to the BLAS and the MeshObjectDescs vector
  // The latter is used by the calling application to build the final MeshObjectDesc
  // buffer
  uint64_t shared_vbo_offset{0};
  uint64_t shared_ibo_offset{0};
  auto vbo_stride = layout->getNumBytesPerItem();
  for (auto const& info : mesh_infos_) {
    auto const* mesh = info.mesh;

    auto add_mesh = [&](DeviceAddress vbo_address, DeviceAddress ibo_address) {
      obj_descs.emplace_back(
          vbo_address, ibo_address, model_transform_address, normal_transform_address);
      accel_builder.addTriangleData({vbo_address,
                                     mesh->numVertices(),
                                     vbo_stride,
                                     ibo_address,
                                     IndexBufferDataType::kUnsigned32,
                                     mesh->numIndices(),
                                     blas_transform_address,
                                     mesh->numTriangles()});
    };

    if (info.vbo_address == 0u) {
      auto mesh_vbo_size = mesh->vertexDataSize();
      auto mesh_ibo_size = mesh->indexDataSize();

      build_result.shared_vbo->updateSubData(
          mesh->vertexData(), mesh_vbo_size, shared_vbo_offset);
      build_result.shared_ibo->updateSubData(
          mesh->indexData(), mesh_ibo_size, shared_ibo_offset);

      add_mesh(shared_vbo_address + shared_vbo_offset,
               shared_ibo_address + shared_ibo_offset);

      shared_vbo_offset += mesh_vbo_size;
      shared_ibo_offset += mesh_ibo_size;
    } else {
      add_mesh(info.vbo_address, info.ibo_address);
    }

    blas_transform_address += sizeof(glm::mat3x4);
    model_transform_address += sizeof(glm::mat4);
    normal_transform_address += sizeof(glm::mat3);
  }

  return build_result;
}

void BLASMeshAssembly::BuildResult::destroyBuffers(ResourceManager& resource_mgr) {
  resource_mgr.destroyBuffer(std::move(shared_vbo));
  resource_mgr.destroyBuffer(std::move(shared_ibo));
  resource_mgr.destroyBuffer(std::move(blas_transform_buffer));
  resource_mgr.destroyBuffer(std::move(model_transform_buffer));
  resource_mgr.destroyBuffer(std::move(normal_transform_buffer));
}

}  // namespace gfx
