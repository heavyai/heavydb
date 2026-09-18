/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Render/Camera.h"

#include <glm/gtc/quaternion.hpp>

#include "GfxDriver/Resources/BufferWrapper.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

//
// Camera::Impl
//

class Camera::Impl {
 public:
  explicit Impl(Projection proj, Type type);

  void setPerspectiveParams(float fov_deg, float aspect, float near_clip, float far_clip);
  void setOrthoParams(const glm::vec2& min,
                      const glm::vec2& max,
                      float near_clip,
                      float far_clip);

  void setPosition(const glm::vec3& pos);
  void setTarget(const glm::vec3& target);
  void translate(const glm::vec3& t);

  void setProjection(Projection proj);
  void setType(Type type);
  void setFOV(float fov_deg);
  void setAspect(float aspect);
  void setClip(float near, float far);

  const glm::mat4& getViewTM();
  const glm::mat4& getProjectionTM();

  Projection getProjection() const;
  Type getType() const;
  const glm::vec3& getPosition() const;
  const float getNearClip() const;
  const float getFarClip() const;

  void updateBufferData(BufferWrapper& buffer);

 private:
  Projection projection_{Projection::kPerspective};
  Type type_{Type::kFree};

  bool view_tm_dirty_{true};
  bool projection_tm_dirty_{true};

  // perspective
  float fov_{50.0f};
  float aspect_{1.0f};

  // orthographic
  glm::vec2 ortho_min_{-1};
  glm::vec2 ortho_max_{1};

  // clipping
  float near_clip_{1.0f};
  float far_clip_{0.1f};

  // view
  glm::vec3 position_{0.0f};
  glm::vec3 front_{glm::vec3(0, 0, -1)};

  glm::vec3 target_pos_{0.0f};

  glm::mat4 view_tm_{1};
  glm::mat4 projection_tm_{1};

  void updateViewTM();
  void updateProjectionTM();
};

Camera::Impl::Impl(Projection proj, Type type) : projection_{proj}, type_{type} {}

void Camera::Impl::setProjection(Projection proj) {
  projection_ = proj;
  updateViewTM();
}

Camera::Projection Camera::Impl::getProjection() const {
  return projection_;
}

void Camera::Impl::setType(Type type) {
  type_ = type;
  updateViewTM();
}

Camera::Type Camera::Impl::getType() const {
  return type_;
}

const glm::mat4& Camera::Impl::getViewTM() {
  if (view_tm_dirty_) {
    updateViewTM();
  }
  return view_tm_;
}

const glm::mat4& Camera::Impl::getProjectionTM() {
  if (projection_tm_dirty_) {
    updateProjectionTM();
  }
  return projection_tm_;
}

void Camera::Impl::updateProjectionTM() {
  switch (projection_) {
    case Projection::kPerspective:
      projection_tm_ =
          glm::perspective(glm::radians(fov_), aspect_, near_clip_, far_clip_);
      break;
    case Projection::kOrthographic:
      projection_tm_ = glm::ortho(
          ortho_min_.x, ortho_max_.x, ortho_min_.y, ortho_max_.y, near_clip_, far_clip_);
      break;
  }
  projection_tm_dirty_ = false;
}

void Camera::Impl::updateViewTM() {
  static constexpr glm::vec3 up{0, 1, 0};
  switch (type_) {
    case Type::kLookAt:
      view_tm_ = glm::lookAt(position_, target_pos_, up);
      view_tm_dirty_ = false;
      break;
    case Type::kFree:
      view_tm_ = glm::lookAt(position_, position_ + front_, up);
      break;
  }
  view_tm_dirty_ = false;
}

void Camera::Impl::setPosition(const glm::vec3& pos) {
  position_ = pos;
  view_tm_dirty_ = true;
}

void Camera::Impl::setTarget(const glm::vec3& target_pos) {
  target_pos_ = target_pos;
  view_tm_dirty_ = true;
}

const glm::vec3& Camera::Impl::getPosition() const {
  return position_;
}

const float Camera::Impl::getNearClip() const {
  return near_clip_;
}

const float Camera::Impl::getFarClip() const {
  return far_clip_;
}

void Camera::Impl::setPerspectiveParams(float fov_deg,
                                        float aspect,
                                        float near_clip,
                                        float far_clip) {
  fov_ = fov_deg;
  aspect_ = aspect;
  near_clip_ = near_clip;
  far_clip_ = far_clip;
  projection_tm_dirty_ = true;
}

void Camera::Impl::setOrthoParams(const glm::vec2& min,
                                  const glm::vec2& max,
                                  float near_clip,
                                  float far_clip) {
  ortho_min_ = min;
  ortho_max_ = max;
  near_clip_ = near_clip;
  far_clip_ = far_clip;
  projection_tm_dirty_ = true;
}

void Camera::Impl::setFOV(float fov_deg) {
  fov_ = fov_deg;
  projection_tm_dirty_ = true;
}

void Camera::Impl::setAspect(float aspect) {
  aspect_ = aspect;
  projection_tm_dirty_ = true;
}

void Camera::Impl::setClip(float near, float far) {
  near_clip_ = near;
  far_clip_ = far;
  projection_tm_dirty_ = true;
}

void Camera::Impl::translate(const glm::vec3& t) {
  position_ += t;
  view_tm_dirty_ = true;
}

namespace {
struct ProjectionBufferData {
  glm::mat4 viewProj{1};
  glm::mat4 viewInverse{1};
  glm::mat4 projInverse{1};
  glm::vec4 camPos{0, 0, 10, 1};
  float nearClip{10.0f};
  float farClip{0.0f};
  int camType{0};
};
}  // namespace

void Camera::Impl::updateBufferData(BufferWrapper& buffer) {
  ProjectionBufferData p;
  auto const& view_tm = getViewTM();
  auto const& proj_tm = getProjectionTM();
  p.viewProj = proj_tm * view_tm;
  p.viewInverse = glm::inverse(view_tm);
  p.projInverse = glm::inverse(proj_tm);
  p.camPos = glm::vec4(position_, 1.0f);
  p.nearClip = near_clip_;
  p.farClip = far_clip_;
  p.camType = projection_ == Projection::kPerspective ? 0 : 1;

  buffer.updateSubData(&p, getBufferDataSize(), 0);
}

//
// Camera
//

Camera::Camera(Projection proj, Type type) : impl_{std::make_unique<Impl>(proj, type)} {}

Camera::~Camera() {}

void Camera::setProjection(Projection proj) {
  impl_->setProjection(proj);
}

Camera::Projection Camera::getProjection() const {
  return impl_->getProjection();
}

void Camera::setType(Type type) {
  impl_->setType(type);
}
Camera::Type Camera::getType() const {
  return impl_->getType();
}

const glm::mat4& Camera::getViewTM() const {
  return impl_->getViewTM();
}
const glm::mat4& Camera::getProjectionTM() const {
  return impl_->getProjectionTM();
}

void Camera::setPosition(const glm::vec3& pos) {
  impl_->setPosition(pos);
}
void Camera::setTarget(const glm::vec3& target) {
  impl_->setTarget(target);
}

const glm::vec3& Camera::getPosition() const {
  return impl_->getPosition();
}
const float Camera::getNearClip() const {
  return impl_->getNearClip();
}
const float Camera::getFarClip() const {
  return impl_->getFarClip();
}

void Camera::setPerspectiveParams(float fov_deg,
                                  float aspect,
                                  float near_clip,
                                  float far_clip) {
  impl_->setPerspectiveParams(fov_deg, aspect, near_clip, far_clip);
}
void Camera::setOrthoParams(const glm::vec2& min,
                            const glm::vec2& max,
                            float near_clip,
                            float far_clip) {
  impl_->setOrthoParams(min, max, near_clip, far_clip);
}

void Camera::setFOV(float fov_deg) {
  impl_->setFOV(fov_deg);
}
void Camera::setAspect(float aspect) {
  impl_->setAspect(aspect);
}
void Camera::setClip(float near, float far) {
  impl_->setClip(near, far);
}

void Camera::translate(const glm::vec3& t) {
  impl_->translate(t);
}

void Camera::updateBufferData(BufferWrapper& buffer) {
  impl_->updateBufferData(buffer);
}

uint64_t Camera::getBufferDataSize() {
  return sizeof(ProjectionBufferData);
}

BufferWrapperUqPtr Camera::createBuffer(ResourceManager& resource_mgr,
                                        BufferAccessType access_type) {
  return resource_mgr.createBuffer("Projection UBO",
                                   {BufferType::kUnspecified,
                                    getBufferDataSize(),
                                    BufferUsageBits::kUniformBufferBit,
                                    access_type});
}

}  // namespace gfx
