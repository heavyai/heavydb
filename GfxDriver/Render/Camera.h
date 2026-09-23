/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <memory>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

//
// Camera
//
// Class to place a camera in a scene with different projection options
// Generates both view and projection matrices
// Concatenate matrices to produce single viewproj matrix (projTM * viewTM)
class Camera {
 public:
  // Projection types:
  //  - kProjection: standard z divide perspective projection
  //  - kOrthographic: an infinite perspective projection
  enum Projection { kPerspective, kOrthographic };

  // Camera types:
  //  - kFree: a "FPS" style camera that can be move with the `translate` function
  //           Currently has limited functionality (no rotation)
  //  - kTarget: camera will always look at the target point in world space
  enum Type { kFree, kLookAt };

  explicit Camera(Projection proj, Type type);
  ~Camera();

  // Set all parameters for perspective projection
  void setPerspectiveParams(float fov_deg, float aspect, float near_clip, float far_clip);

  // Set projection properties for orthographic projection
  // view_min and view_max define the view (camera space) bounds
  // Example - orthographic camera that projects view space -1 to 1 with adjustments for
  //  non-square images, and reverse z-buffer going from 0.1 to 10:
  //  setOrthoParams({-aspect_ratio, -1.0f}, {aspect_ratio, 1.0f}, 10.0f, 0.1f);
  void setOrthoParams(const glm::vec2& view_min,
                      const glm::vec2& view_max,
                      float near_clip,
                      float far_clip);

  // Set the camera position in world space
  void setPosition(const glm::vec3& pos);

  // Set the target position. View matrix will align camera z to target
  void setTarget(const glm::vec3& target);

  // Translate the camera in world space
  void translate(const glm::vec3& t);

  // Change camera projection type (perspective or orthographic)
  void setProjection(Projection proj);

  // Change camera type (target or free)
  void setType(const Type type);

  // Set perspective field-of-view in degrees
  void setFOV(float fov_deg);
  // Set perspective aspect ratio
  void setAspect(float aspect);

  // Set depth clip range
  // For reverse z-buffer near > far
  void setClip(float near, float far);

  // Get transformation matrices. Dirty transforms will be recomputed
  const glm::mat4& getViewTM() const;
  const glm::mat4& getProjectionTM() const;

  // Get settings
  Projection getProjection() const;
  Type getType() const;
  const glm::vec3& getPosition() const;
  const float getNearClip() const;
  const float getFarClip() const;

  // Get required size of the projection properties UBO
  static uint64_t getBufferDataSize();

  // Create a pre-sized buffer suitable for use as a projection UBO
  static BufferWrapperUqPtr createBuffer(
      ResourceManager& resource_mgr,
      BufferAccessType access_type = BufferAccessType::kHostVisible);

  // Fill UBO with current properties
  void updateBufferData(BufferWrapper& buffer);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfx
