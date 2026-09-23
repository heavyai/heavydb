/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <ostream>

#include <glm/vec3.hpp>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

//
// LightDescriptor
//
// Returned from LightManager from the add* methods
// Can be used with LightManager::updateLightProperties to modify an existing light
class LightDescriptor {
 public:
  enum class Type { kPoint, kParallel, kCOUNT };

  explicit LightDescriptor(uint32_t index,
                           Type type,
                           const glm::vec3& position,
                           const glm::vec3& direction,
                           const glm::vec3& color,
                           float multiplier,
                           bool do_decay);

  // Enable move construction to allow moving a returned descriptor
  // into a unique_ptr
  LightDescriptor(LightDescriptor&& other) = default;

  uint32_t getIndex() const;

  // Mutable properties
  Type type;
  glm::vec3 position;   // point light only
  glm::vec3 direction;  // parallel light only
  glm::vec3 color;
  float multiplier;
  bool do_decay;

 private:
  uint32_t index_;
};

std::ostream& operator<<(std::ostream& os, const LightDescriptor::Type value);
std::string to_string(const LightDescriptor::Type);

//
// LightManager
//
// Stores light properties used by lighting.glsl
// Currently only allows adding and modifying lights
// Handles updating a buffer (SSBO) to bind to the gfx::Material using lighting.glsl
//
class LightManager {
 public:
  LightManager();
  ~LightManager();

  LightDescriptor addPointLight(const glm::vec3& position,
                                const glm::vec3& color,
                                float multiplier,
                                bool do_decay);
  LightDescriptor addParallelLight(const glm::vec3& direction,
                                   const glm::vec3& color,
                                   float multiplier);

  // Update an exsiting light using the contents of a modified LightDescriptor
  // updateBufferData must be called once all Lights have been updated
  void updateLightProperties(const LightDescriptor& light_desc);

  // Clear the light list
  void clear();

  // Get the number of Lights
  int getNumLights() const;

  // Get the required size of the light properties buffer for all lights
  uint64_t getBufferDataSize() const;

  // Create a pre-sized buffer suitable for use as a light data buffer
  BufferWrapperUqPtr createBuffer(
      ResourceManager& resource_mgr,
      BufferAccessType access_type = BufferAccessType::kHostVisible) const;

  // Fill a SSBO with properties for all lights, to be used by the shader
  void updateBufferData(BufferWrapper& buffer);

  // Bind the SSBO to the material and set other uniforms
  void bindToMaterial(Material& material,
                      BufferWrapper& buffer,
                      float shadow_bias = 0.0f);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfx
