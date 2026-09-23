/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "GfxDriver/Render/LightManager.h"
#include "GfxDriver/Render/PBRMaterialManager.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

//
// PBRScene
//
// Helper class for setting up PBR shaders and lights
// Manages PBRMaterialManager and LightManager, providing facade functions
// Handles creation and updating of material and light data storage buffers
//
// For usage example see PBRSceneTest
//
class PBRScene {
 public:
  explicit PBRScene(ResourceManager& resource_mgr);
  ~PBRScene();

  // create/destroy buffer resources
  void destroyResources();

  //
  // PBRMaterialManager facade
  //

  // Create PBRMaterialDescriptors and add them to the data buffer
  // Parameters are declared in PBRMaterialManager.h

  // addConstantColorMaterial
  template <typename... Args>
  PBRMaterialDescriptor& addConstantColorMaterial(Args&&... args) {
    material_descs_.emplace_back(
        material_mgr_.addConstantColorMaterial(std::forward<Args>(args)...));
    return material_descs_.back();
  }
  // addBlinnPhongMaterial
  template <typename... Args>
  PBRMaterialDescriptor& addBlinnPhongMaterial(Args&&... args) {
    material_descs_.emplace_back(
        material_mgr_.addBlinnPhongMaterial(std::forward<Args>(args)...));
    return material_descs_.back();
  }
  // addCookTorranceMaterial
  template <typename... Args>
  PBRMaterialDescriptor& addCookTorranceMaterial(Args&&... args) {
    material_descs_.emplace_back(
        material_mgr_.addCookTorranceMaterial(std::forward<Args>(args)...));
    return material_descs_.back();
  }

  // Get a specific descriptor by index
  PBRMaterialDescriptor& getMaterialDescriptor(uint32_t index);

  // Iterate through all material descriptors calling f on each
  // If f modifies a PBRMaterialDescriptor it must also call updateMaterialProperties
  template <typename UnaryFunction>
  void iterateMaterialDescs(UnaryFunction f) {
    std::for_each(material_descs_.begin(), material_descs_.end(), f);
  }

  // Translate a PBRMaterialDescriptor to the internal format used by shaders
  // Must be called after creating or modifying a PBRMaterialDescriptor
  // Does not need to be called after createXXX
  void updateMaterialProperties(const PBRMaterialDescriptor& material_desc);

  // Copy internal data vector to storage buffer
  // Call after all PBRMaterialDescriptors are created and updated
  void updateMaterialDataBuffer();

  //
  // LightManager facade
  //

  // Create LightDescriptors and add them to the data buffer
  // Parameters are declared in LightManager.h

  // addPointLight
  template <typename... Args>
  LightDescriptor& addPointLight(Args&&... args) {
    light_descs_.emplace_back(light_mgr_.addPointLight(std::forward<Args>(args)...));
    return light_descs_.back();
  }
  // addParallelLight
  template <typename... Args>
  LightDescriptor& addParallelLight(Args&&... args) {
    light_descs_.emplace_back(light_mgr_.addParallelLight(std::forward<Args>(args)...));
    return light_descs_.back();
  }

  // Get a specific descriptor by index
  LightDescriptor& getLightDescriptor(uint32_t index);

  // Iterate through all light descriptors calling f on each
  // If f modifies a LightDescriptor it must also call updateLightProperties
  template <typename UnaryFunction>
  void iterateLightDescs(UnaryFunction f) {
    std::for_each(light_descs_.begin(), light_descs_.end(), f);
  }

  // Translate a LightDescriptor to the internal format used by shaders
  // Must be called after modifying a PBRMaterialDescriptor
  // Does not need to be called after createXXX
  void updateLightProperties(const LightDescriptor& light_desc);

  // Copy internal data vector to storage buffer
  // Call after all LightDesciptors are created and updated
  void updateLightDataBuffer();

  // Bind material and light data buffer descriptors to gfx::Material
  void bindBuffersToMaterial(Material& material);

 private:
  ResourceManager& resource_mgr_;

  PBRMaterialManager material_mgr_;
  std::vector<PBRMaterialDescriptor> material_descs_;
  BufferWrapperUqPtr material_data_buffer_;

  LightManager light_mgr_;
  std::vector<LightDescriptor> light_descs_;
  BufferWrapperUqPtr light_data_buffer_;
};

}  // namespace gfx
