/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Render/PBRScene.h"

#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

PBRScene::PBRScene(ResourceManager& resource_mgr) : resource_mgr_{resource_mgr} {}

PBRScene::~PBRScene() {
  destroyResources();
}

void PBRScene::destroyResources() {
  if (material_data_buffer_) {
    resource_mgr_.destroyBuffer(std::move(material_data_buffer_));
  }
  if (light_data_buffer_) {
    resource_mgr_.destroyBuffer(std::move(light_data_buffer_));
  }
}

PBRMaterialDescriptor& PBRScene::getMaterialDescriptor(uint32_t index) {
  CHECK_LT(index, material_descs_.size());
  return material_descs_[index];
}

void PBRScene::updateMaterialProperties(const PBRMaterialDescriptor& material_desc) {
  material_mgr_.updateMaterialProperties(material_desc);
}

void PBRScene::updateMaterialDataBuffer() {
  if (!material_data_buffer_) {
    material_data_buffer_ = material_mgr_.createBuffer(resource_mgr_);
  }
  CHECK(material_data_buffer_);
  material_mgr_.updateBufferData(*material_data_buffer_);
}

LightDescriptor& PBRScene::getLightDescriptor(uint32_t index) {
  CHECK_LT(index, light_descs_.size());
  return light_descs_[index];
}

void PBRScene::updateLightProperties(const LightDescriptor& light_desc) {
  light_mgr_.updateLightProperties(light_desc);
}

void PBRScene::updateLightDataBuffer() {
  if (!light_data_buffer_) {
    light_data_buffer_ = light_mgr_.createBuffer(resource_mgr_);
  }
  CHECK(light_data_buffer_);
  light_mgr_.updateBufferData(*light_data_buffer_);
}

void PBRScene::bindBuffersToMaterial(gfx::Material& material) {
  CHECK(material_data_buffer_);
  CHECK(light_data_buffer_);
  light_mgr_.bindToMaterial(material, *light_data_buffer_, 0.01f);
  material_mgr_.bindToMaterial(material, *material_data_buffer_);
}

}  // namespace gfx
