/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

//
// PBRMaterialDescriptor
//
// Returned from PBRMaterialManager from the add* methods
// Can be used with PBRMaterialManager::updateMaterialProperties to modify a material
//
class PBRMaterialDescriptor {
 public:
  enum class ColorSource { kConstant, kObjCoords, kCOUNT };
  enum class ShadingModel { kConstant, kBlinnPhong, kCookTorrance, kCOUNT };

  explicit PBRMaterialDescriptor(uint32_t index,
                                 ShadingModel shading_model,
                                 ColorSource albedo_source,
                                 const glm::vec3& albedo,
                                 float opacity,
                                 float glossiness,
                                 float specular_strength,
                                 float roughness,
                                 float metallic);

  // Enable move construction to allow moving returned descriptor
  // into a unique_ptr
  PBRMaterialDescriptor(PBRMaterialDescriptor&& other) = default;

  uint32_t getIndex() const;

  // Mutable properties
  // Common
  ShadingModel shading_model;
  ColorSource albedo_source;
  glm::vec3 albedo;
  float opacity;

  // Blinn-Phong
  float glossiness;
  float specular_strength;

  // Cook-Torrance
  float roughness;
  float metallic;

 private:
  uint32_t index_;
};

//
// PBRMaterialManager
//
// Stores material properties used by shading.glsl
// Currently only allows adding and modifying materials
// Handles updating a buffer (SSBO) which must be bound to the gfx::Material
// using shading.glsl
//
class PBRMaterialManager {
 public:
  PBRMaterialManager();
  ~PBRMaterialManager();

  // Add a new Material to the Materials vector
  // returns a PBRMaterialDescriptor which can be modified and
  // used in calls to updateMaterial
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

  // Update an existing material using the contents of a modified
  // PBRMaterialDescriptor
  // updateBufferData must be called once all Materials have been updated
  void updateMaterialProperties(const PBRMaterialDescriptor& material_desc);

  // Clear the material list
  void clear();

  // Get the number of Materials
  int getNumMaterials() const;

  // Get the required size of the material properties buffer for all Materials
  uint64_t getBufferDataSize() const;

  // Create a pre-sized buffer suitable for use as a properties buffer
  BufferWrapperUqPtr createBuffer(
      ResourceManager& resource_mgr,
      BufferAccessType access_type = BufferAccessType::kHostVisible) const;

  // Fill a buffer with properties for all materials, which will be used
  // by the shader
  void updateBufferData(BufferWrapper& buffer);

  // Bind the ShaderStorageBuffer to the material and set other uniforms
  void bindToMaterial(Material& material, BufferWrapper& buffer);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfx
