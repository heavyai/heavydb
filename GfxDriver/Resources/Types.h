/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>
#include <ostream>

#include "GfxDriver/Types.h"

namespace gfx {

// API-agnostic base class for graphics resources (buffers, shaders, textures etc.)
class Resource;
using ResourceUqPtr = std::unique_ptr<Resource>;

// resource interface classes
class Texture;
class Framebuffer;
class Buffer;
class Pipeline;
class GraphicsPipeline;
class ComputePipeline;
class RaytracingPipeline;
class RenderPass;
class AccelerationStructure;
// resource wrapper classes
class BufferWrapper;
class HostVisibleBufferWrapper;
using BufferWrapperUqPtr = std::unique_ptr<BufferWrapper>;
using HostVisibleBufferWrapperUqPtr = std::unique_ptr<HostVisibleBufferWrapper>;

// other shared classes
class AttachmentManager;
class IndirectDrawVertexBuffer;
class IndirectDrawIndexBuffer;
class IndexBuffer;
class VertexBuffer;

// layouts
class BaseBufferLayout;
class InterleavedBufferLayout;
class SequentialBufferLayout;
class ShaderBlockLayout;
using BufferLayoutShPtr = std::shared_ptr<BaseBufferLayout>;
using InterleavedBufferLayoutShPtr = std::shared_ptr<InterleavedBufferLayout>;
using SequentialBufferLayoutShPtr = std::shared_ptr<SequentialBufferLayout>;
using ShaderBlockLayoutShPtr = std::shared_ptr<ShaderBlockLayout>;

// API-agnostic resource API handle type (e.g. Vulkan VkHandle)
using ResourceHandle = uint64_t;

// Internal per-context IDs and Device/ID pairs
using ResourceId = uint32_t;
using UniqueResourceId = std::pair<DeviceId, ResourceId>;

// Data captured at resource creation for tracking purposes
struct ResourceTrackingData {
  std::string origin;
  std::string stackTrace;
};

// SemaphoreHandle
using SemaphoreHandle = uint64_t;

// DeviceAddress
using DeviceAddress = uint64_t;

// Logging callback
using LoggingCallback = std::function<void(std::ostream&)>;

// Resource wrappers
class PrimitiveAssembly;
using PrimitiveAssemblyUqPtr = std::unique_ptr<PrimitiveAssembly>;

// Abstract material
class Material;
using MaterialUqPtr = std::unique_ptr<Material>;

// Clear Texture Value
union ClearTextureValue {
  struct {
    float r;
    float g;
    float b;
    float a;
  };
  struct {
    float d;
    uint32_t s;
  };
  int32_t i;
  uint32_t u;

  constexpr ClearTextureValue() : r{0.0f}, g{0.0f}, b{0.0f}, a{0.0f} {}
  constexpr ClearTextureValue(float r, float g, float b, float a)
      : r{r}, g{g}, b{b}, a{a} {}
  constexpr ClearTextureValue(float d, uint32_t s) : d{d}, s{s} {}
  constexpr ClearTextureValue(int32_t i) : i{i} {}
  constexpr ClearTextureValue(uint32_t u) : u{u} {}
};

/*
 * conversions to string
 */

std::string to_string(const UniqueResourceId& id);

}  // namespace gfx

/*
 * streaming
 */

std::ostream& operator<<(std::ostream& os, const gfx::UniqueResourceId& value);
