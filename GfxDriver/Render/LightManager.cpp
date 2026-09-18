/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Render/LightManager.h"

#include <vector>

#include <glm/vec4.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

namespace {
struct LightBufferData {
  union {
    glm::vec4 position;
    glm::vec4 direction;
  };
  glm::vec4 color;
  int type;
  uint32_t flags;
  int pad[2];  // match std430 layout on GPU
};
}  // namespace

//
// LightDescriptor
//
LightDescriptor::LightDescriptor(uint32_t index,
                                 Type type,
                                 const glm::vec3& position,
                                 const glm::vec3& direction,
                                 const glm::vec3& color,
                                 float multiplier,
                                 bool do_decay)
    : type{type}
    , position{position}
    , direction{direction}
    , color{color}
    , multiplier{multiplier}
    , do_decay{do_decay}
    , index_{index} {}

uint32_t LightDescriptor::getIndex() const {
  return index_;
}

//
// LightManager::Impl
//

class LightManager::Impl {
 public:
  Impl() : is_buffer_dirty_{false} {}

  LightDescriptor addPointLight(const glm::vec3& position,
                                const glm::vec3& color,
                                float multiplier,
                                bool do_decay);
  LightDescriptor addParallelLight(const glm::vec3& direction,
                                   const glm::vec3& color,
                                   float multiplier);

  void updateLightProperties(const LightDescriptor& light_desc);

  void clear();

  int getNumLights() const;
  uint64_t getBufferDataSize() const;
  BufferWrapperUqPtr createBuffer(ResourceManager& resource_mgr,
                                  BufferAccessType access_type) const;

  void updateBufferData(BufferWrapper& buffer);
  void bindToMaterial(Material& material, BufferWrapper& buffer, float shadow_bias);

 private:
  std::vector<LightBufferData> lights_;
  bool is_buffer_dirty_;
};

LightDescriptor LightManager::Impl::addPointLight(const glm::vec3& position,
                                                  const glm::vec3& color,
                                                  float multiplier,
                                                  const bool do_decay) {
  uint32_t flags = do_decay ? 1 : 0;
  lights_.push_back(
      {{glm::vec4{position, 1.0f}}, glm::vec4{color, multiplier}, 0, flags});
  is_buffer_dirty_ = true;
  return LightDescriptor{static_cast<uint32_t>(lights_.size() - 1),
                         LightDescriptor::Type::kPoint,
                         {position},
                         glm::vec3{0},
                         color,
                         multiplier,
                         do_decay};
}

LightDescriptor LightManager::Impl::addParallelLight(const glm::vec3& direction,
                                                     const glm::vec3& color,
                                                     float multiplier) {
  // TODO: treat as plane
  lights_.push_back({{glm::vec4{direction, 1.0f}}, glm::vec4{color, multiplier}, 1, 0u});
  is_buffer_dirty_ = true;
  return LightDescriptor{static_cast<uint32_t>(lights_.size() - 1),
                         LightDescriptor::Type::kParallel,
                         glm::vec3{0},
                         direction,
                         color,
                         multiplier,
                         false};
}

void LightManager::Impl::updateLightProperties(const LightDescriptor& light_desc) {
  auto index = light_desc.getIndex();
  CHECK_LT(index, lights_.size());
  auto& lt = lights_[index];
  lt.type = static_cast<int>(light_desc.type);
  switch (light_desc.type) {
    case LightDescriptor::Type::kPoint:
      lt.position = glm::vec4{light_desc.position, 1.0f};
      break;
    case LightDescriptor::Type::kParallel:
      lt.direction = glm::vec4{light_desc.direction, 1.0f};
      break;
    case LightDescriptor::Type::kCOUNT:
      UNREACHABLE();
      break;
  }
  lt.color = glm::vec4{light_desc.color, light_desc.multiplier};
  lt.flags = light_desc.do_decay ? 1 : 0;
  is_buffer_dirty_ = true;
}

void LightManager::Impl::clear() {
  lights_.clear();
}

int LightManager::Impl::getNumLights() const {
  return lights_.size();
}

uint64_t LightManager::Impl::getBufferDataSize() const {
  return lights_.size() * sizeof(LightBufferData);
}

BufferWrapperUqPtr LightManager::Impl::createBuffer(ResourceManager& resource_mgr,
                                                    BufferAccessType access_type) const {
  return resource_mgr.createBuffer("Lights SSBO",
                                   {BufferType::kUnspecified,
                                    getBufferDataSize(),
                                    BufferUsageBits::kStorageBufferBit,
                                    access_type});
}

void LightManager::Impl::updateBufferData(BufferWrapper& buffer) {
  if (is_buffer_dirty_) {
    CHECK(any_bits_set(buffer.getUsageBits() & BufferUsageBits::kStorageBufferBit));
    auto buffer_size = getBufferDataSize();
    if (!buffer.isUsable() || buffer.getNumBytes() < buffer_size) {
      buffer.rebuild(lights_.data(), buffer_size);
    } else {
      buffer.updateSubData(lights_.data(), buffer_size, 0);
    }
    is_buffer_dirty_ = false;
  }
}

void LightManager::Impl::bindToMaterial(Material& material,
                                        BufferWrapper& buffer,
                                        float shadow_bias) {
  CHECK(any_bits_set(buffer.getUsageBits() & BufferUsageBits::kStorageBufferBit));
  material.bindShaderStorageBufferToBlock("LIGHT_DATA_SSBO", buffer);
  material.setUniformAttribute("numLights", static_cast<uint32_t>(lights_.size()));
  material.setUniformAttribute("shadowBias", shadow_bias);
}

std::ostream& operator<<(std::ostream& os, const LightDescriptor::Type value) {
  using e = LightDescriptor::Type;
  switch (value) {
    case e::kPoint:
      os << "Point";
      break;
    case e::kParallel:
      os << "Parallel";
      break;
    case e::kCOUNT:
      UNREACHABLE();
      break;
  }
  return os;
}

std::string to_string(const LightDescriptor::Type value) {
  using e = LightDescriptor::Type;
  switch (value) {
    case e::kPoint:
      return "Point";
    case e::kParallel:
      return "Parallel";
    case e::kCOUNT:
      UNREACHABLE();
  }
  return "";
}

//
// LightManager
//

LightManager::LightManager() : impl_{std::make_unique<Impl>()} {}

LightManager::~LightManager() {}

LightDescriptor LightManager::addPointLight(const glm::vec3& position,
                                            const glm::vec3& color,
                                            float multiplier,
                                            bool do_decay) {
  return impl_->addPointLight(position, color, multiplier, do_decay);
}

LightDescriptor LightManager::addParallelLight(const glm::vec3& direction,
                                               const glm::vec3& color,
                                               float multiplier) {
  return impl_->addParallelLight(direction, color, multiplier);
}

void LightManager::updateLightProperties(const LightDescriptor& light_desc) {
  impl_->updateLightProperties(light_desc);
}

void LightManager::clear() {
  impl_->clear();
}

int LightManager::getNumLights() const {
  return impl_->getNumLights();
}

uint64_t LightManager::getBufferDataSize() const {
  return impl_->getBufferDataSize();
}

BufferWrapperUqPtr LightManager::createBuffer(ResourceManager& resource_mgr,
                                              BufferAccessType access_type) const {
  return impl_->createBuffer(resource_mgr, access_type);
}

void LightManager::updateBufferData(BufferWrapper& buffer) {
  impl_->updateBufferData(buffer);
}

void LightManager::bindToMaterial(Material& material,
                                  BufferWrapper& buffer,
                                  float shadow_bias) {
  impl_->bindToMaterial(material, buffer, shadow_bias);
}

}  // namespace gfx
