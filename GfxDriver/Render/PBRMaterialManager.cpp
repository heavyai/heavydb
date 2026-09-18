/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Render/PBRMaterialManager.h"

#include <vector>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Resources/BufferWrapper.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

namespace {
struct PBMaterialBufferData {
  glm::vec4 albedo;
  float opacity;
  union {
    float roughness;
    float glossiness;
  };
  union {
    float metallic;
    float specular_strength;
  };
  int shading_model;
  int albedo_source;
  int pad[3];  // match std430 layout on GPU
};
}  // namespace

PBRMaterialDescriptor::PBRMaterialDescriptor(uint32_t index,
                                             ShadingModel shading_model,
                                             ColorSource albedo_source,
                                             const glm::vec3& albedo,
                                             float opacity,
                                             float glossiness,
                                             float specular_strength,
                                             float roughness,
                                             float metallic)
    : shading_model{shading_model}
    , albedo_source{albedo_source}
    , albedo{albedo}
    , opacity{opacity}
    , glossiness{glossiness}
    , specular_strength{specular_strength}
    , roughness{roughness}
    , metallic{metallic}
    , index_{index} {}

uint32_t PBRMaterialDescriptor::getIndex() const {
  return index_;
}

//
// PBRMaterialManager::Impl
//

class PBRMaterialManager::Impl {
 public:
  Impl() : is_buffer_dirty_{false} {}

  PBRMaterialDescriptor addConstantColorMaterial(
      PBRMaterialDescriptor::ColorSource albedo_source,
      const glm::vec4& albedo,
      float opacity);

  PBRMaterialDescriptor addBlinnPhongMaterial(
      PBRMaterialDescriptor::ColorSource albedo_source,
      const glm::vec4& albedo,
      float opacity,
      float glossiness,
      float specular_strength);

  PBRMaterialDescriptor addCookTorranceMaterial(
      PBRMaterialDescriptor::ColorSource albedo_source,
      const glm::vec4& albedo,
      float opacity,
      float roughness,
      float metallic);

  void updateMaterialProperties(const PBRMaterialDescriptor& material_desc);

  void clear();

  int getNumMaterials() const;
  uint64_t getBufferDataSize() const;
  BufferWrapperUqPtr createBuffer(ResourceManager& resource_mgr,
                                  BufferAccessType access_type) const;

  void updateBufferData(BufferWrapper& buffer);
  void bindToMaterial(Material& material, BufferWrapper& buffer);

 private:
  std::vector<PBMaterialBufferData> materials_;
  bool is_buffer_dirty_;
};

PBRMaterialDescriptor PBRMaterialManager::Impl::addConstantColorMaterial(
    PBRMaterialDescriptor::ColorSource albedo_source,
    const glm::vec4& albedo,
    float opacity) {
  materials_.push_back({albedo,
                        opacity,
                        {0.0f},
                        {0.0f},
                        static_cast<int>(PBRMaterialDescriptor::ShadingModel::kConstant),
                        static_cast<int>(albedo_source)});
  is_buffer_dirty_ = true;

  return PBRMaterialDescriptor{static_cast<uint32_t>(materials_.size() - 1),
                               PBRMaterialDescriptor::ShadingModel::kConstant,
                               albedo_source,
                               albedo,
                               opacity,
                               0.0f,
                               0.0f,
                               0.0f,
                               0.0f};
}

PBRMaterialDescriptor PBRMaterialManager::Impl::addBlinnPhongMaterial(
    PBRMaterialDescriptor::ColorSource albedo_source,
    const glm::vec4& albedo,
    float opacity,
    float glossiness,
    float specular_strength) {
  materials_.push_back(
      {albedo,
       opacity,
       {glossiness},
       {specular_strength},
       static_cast<int>(PBRMaterialDescriptor::ShadingModel::kBlinnPhong),
       static_cast<int>(albedo_source)});
  is_buffer_dirty_ = true;

  return PBRMaterialDescriptor{static_cast<uint32_t>(materials_.size() - 1),
                               PBRMaterialDescriptor::ShadingModel::kBlinnPhong,
                               albedo_source,
                               albedo,
                               opacity,
                               glossiness,
                               specular_strength,
                               0.0f,
                               0.0f};
}

PBRMaterialDescriptor PBRMaterialManager::Impl::addCookTorranceMaterial(
    PBRMaterialDescriptor::ColorSource albedo_source,
    const glm::vec4& albedo,
    float opacity,
    float roughness,
    float metallic) {
  materials_.push_back(
      {albedo,
       opacity,
       {roughness},
       {metallic},
       static_cast<int>(PBRMaterialDescriptor::ShadingModel::kCookTorrance),
       static_cast<int>(albedo_source)});
  is_buffer_dirty_ = true;

  return PBRMaterialDescriptor{static_cast<uint32_t>(materials_.size() - 1),
                               PBRMaterialDescriptor::ShadingModel::kCookTorrance,
                               albedo_source,
                               albedo,
                               opacity,
                               0.0f,
                               0.0f,
                               roughness,
                               metallic};
}

void PBRMaterialManager::Impl::updateMaterialProperties(
    const PBRMaterialDescriptor& material_desc) {
  auto index = material_desc.getIndex();
  CHECK_LT(index, materials_.size());
  auto& m = materials_[index];

  m.albedo_source = static_cast<int>(material_desc.albedo_source);
  m.shading_model = static_cast<int>(material_desc.shading_model);
  m.opacity = material_desc.opacity;
  m.albedo = glm::vec4{material_desc.albedo, 1.0f};
  using SM = PBRMaterialDescriptor::ShadingModel;
  switch (material_desc.shading_model) {
    case SM::kConstant:
      break;
    case SM::kBlinnPhong:
      m.roughness = material_desc.glossiness;
      m.metallic = material_desc.specular_strength;
      break;
    case SM::kCookTorrance:
      m.roughness = material_desc.roughness;
      m.metallic = material_desc.metallic;
      break;
    case SM::kCOUNT:
      UNREACHABLE();
      break;
  }
  is_buffer_dirty_ = true;
}

void PBRMaterialManager::Impl::clear() {
  materials_.clear();
}

int PBRMaterialManager::Impl::getNumMaterials() const {
  return materials_.size();
}
uint64_t PBRMaterialManager::Impl::getBufferDataSize() const {
  return materials_.size() * sizeof(PBMaterialBufferData);
}

BufferWrapperUqPtr PBRMaterialManager::Impl::createBuffer(
    ResourceManager& resource_mgr,
    BufferAccessType access_type) const {
  return resource_mgr.createBuffer("Material SSBO",
                                   {gfx::BufferType::kUnspecified,
                                    getBufferDataSize(),
                                    gfx::BufferUsageBits::kStorageBufferBit,
                                    access_type});
}

void PBRMaterialManager::Impl::updateBufferData(BufferWrapper& buffer) {
  if (is_buffer_dirty_) {
    auto buffer_size = getBufferDataSize();
    if (!buffer.isUsable() || buffer.getNumBytes() < buffer_size) {
      buffer.rebuild(materials_.data(), buffer_size);
    } else {
      buffer.updateSubData(materials_.data(), buffer_size, 0);
    }
    is_buffer_dirty_ = false;
  }
}

void PBRMaterialManager::Impl::bindToMaterial(Material& material, BufferWrapper& buffer) {
  material.bindShaderStorageBufferToBlock("MATERIAL_PROPS_SSBO", buffer);
}

//
// PBRMaterialManager
//

PBRMaterialManager::PBRMaterialManager() : impl_{std::make_unique<Impl>()} {}

PBRMaterialManager::~PBRMaterialManager() {}

PBRMaterialDescriptor PBRMaterialManager::addConstantColorMaterial(
    PBRMaterialDescriptor::ColorSource albedo_source,
    const glm::vec4& albedo,
    float opacity) {
  return impl_->addConstantColorMaterial(albedo_source, albedo, opacity);
}

PBRMaterialDescriptor PBRMaterialManager::addBlinnPhongMaterial(
    PBRMaterialDescriptor::ColorSource albedo_source,
    const glm::vec4& albedo,
    float opacity,
    float glossiness,
    float specular_strength) {
  return impl_->addBlinnPhongMaterial(
      albedo_source, albedo, opacity, glossiness, specular_strength);
}

PBRMaterialDescriptor PBRMaterialManager::addCookTorranceMaterial(
    PBRMaterialDescriptor::ColorSource albedo_source,
    const glm::vec4& albedo,
    float opacity,
    float roughness,
    float metallic) {
  return impl_->addCookTorranceMaterial(
      albedo_source, albedo, opacity, roughness, metallic);
}

void PBRMaterialManager::updateMaterialProperties(
    const PBRMaterialDescriptor& material_desc) {
  impl_->updateMaterialProperties(material_desc);
}

void PBRMaterialManager::clear() {
  impl_->clear();
}

int PBRMaterialManager::getNumMaterials() const {
  return impl_->getNumMaterials();
}

uint64_t PBRMaterialManager::getBufferDataSize() const {
  return impl_->getBufferDataSize();
}

BufferWrapperUqPtr PBRMaterialManager::createBuffer(ResourceManager& resource_mgr,
                                                    BufferAccessType access_type) const {
  return impl_->createBuffer(resource_mgr, access_type);
}

void PBRMaterialManager::updateBufferData(BufferWrapper& buffer) {
  impl_->updateBufferData(buffer);
}

void PBRMaterialManager::bindToMaterial(Material& material, BufferWrapper& buffer) {
  impl_->bindToMaterial(material, buffer);
}

}  // namespace gfx
