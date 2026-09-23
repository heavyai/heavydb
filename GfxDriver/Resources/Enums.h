/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

#include "Shared/EnumBitmaskOps.h"

namespace gfx {
/**
 * Resource Types
 **/

enum class ResourceType : uint8_t {
  kShaderProgram,
  kShaderModule,
  kFramebuffer,
  kTexture,
  kBaseBuffer,
  kVertexBuffer,
  kIndexBuffer,
  kIndirectDrawVertexBuffer,
  kIndirectDrawIndexBuffer,
  kPixelBuffer,
  kVertexArray,
  kPipeline,
  kRenderPass,
  kAccelerationStructureBuffer,
  kAccelerationStructure,
  kShaderBindingTableBuffer,
  kSlabWrapperBuffer,
  kQueryPool
};

/**
 * Pixel formats
 * Used for all image-related objects (textures, framebuffers, PBOs)
 **/

enum class PixelFormat : uint8_t {
  kR8,
  kRG8,
  kRGBA8,
  kBGRA8,
  kR32UI,
  kR32I,
  kDepth,
  kDepthHighP,
  kDepthStencil,
  kDepthStencilHighP,
  kCOUNT
};

enum class ImageAspect : uint8_t { kColor, kDepth, kStencil, kDepthStencil, kCOUNT };

enum class ImageUsageBits {
  kNone = 0,
  kSampledBit = 1 << 0,
  kStorageBit = 1 << 1,
  kColorAttachmentBit = 1 << 2,
  kDepthStencilAttachmentBit = 1 << 3,
  kTransientAttachmentBit = 1 << 4,
  kInputAttachmentBit = 1 << 5,
  kExternalApiBit = 1 << 6,
  kMutableViewBit = 1 << 7
};

/**
 * Image layouts required for Vulkan renderpass image transitions
 * */
enum class ImageLayout {
  kUndefined,
  kGeneral,
  kAttachment,
  kShaderReadOnly,
  kTransferSrc,
  kTransferDst,
  kMapRead,
  kMapWrite,
  kPresentSrc
};

/**
 * Buffer access controls
 **/

enum class BufferAccessType : uint8_t {
  kHostVisible,
  kHostVisibleCached,
  kDeviceLocal,
  kExternalApi
};

/**
 * Texture Sampler controls
 **/

enum class SamplerFilterMode : uint8_t { kNearest, kLinear };

enum class SamplerMipmapMode : uint8_t { kNearest, kLinear };

enum class SamplerWrapMode : uint8_t {
  kRepeat,
  kMirrorRepeat,
  kClampEdge,
  kClampBorder,
  kCOUNT
};

/**
 * Buffers and Buffer Attributes
 **/

enum class BufferType : uint8_t {
  kVertexBuffer,
  kPixelBuffer,
  kIndexBuffer,
  kIndirectDrawVertexBuffer,
  kIndirectDrawIndexBuffer,
  kAccelerationStructureBuffer,
  kShaderBindingTableBuffer,
  kSlabWrapperBuffer,
  kUnspecified,
  kCOUNT
};

enum class BufferUsageBits {
  kNone = 0,
  kUniformBufferBit = 1 << 0,
  kStorageBufferBit = 1 << 1,
  kDeviceAddressBit = 1 << 2,
  kAccelerationStructureReadOnlyBit = 1 << 3,
  kLayoutBufferBit = 1 << 4
};

enum class BufferAttrType : uint8_t {
  kUint,
  kVec2ui,
  kVec3ui,
  kVec4ui,

  kInt,
  kVec2i,
  kVec3i,
  kVec4i,

  kFloat,
  kVec2f,
  kVec3f,
  kVec4f,

  kDouble,
  kVec2d,
  kVec3d,
  kVec4d,

  kUint64,
  kVec2ui64,
  kVec3ui64,
  kVec4ui64,

  kInt64,
  kVec2i64,
  kVec3i64,
  kVec4i64,

  kMat3x2f,
  kMat3x2d,

  kBool,

  kCOUNT
};

enum class BufferLayoutType : uint8_t { kInterleaved, kSequential };
enum class ShaderBlockType { kUniformBuffer, kStorageBuffer };

// index buffer data type. Note: the enum value is the byte size of the underlying type.
enum class IndexBufferDataType {
  kUnsigned16 = sizeof(uint16_t),
  kUnsigned32 = sizeof(uint32_t)
};

/**
 * Acceleration structures
 **/
enum class AccelerationStructureType { kBottomLevel, kTopLevel };

/**
 * Rasterization sample counts
 **/
enum class RasterSampleCount { k1, k2, k4, k8, k16, k32, k64, kCOUNT };

/**
 * conversions to string
 **/

std::string to_string(const ResourceType value);
std::string to_string(const PixelFormat value);
std::string to_string(const ImageAspect value);
std::string to_string(const BufferAccessType value);
std::string to_string(const SamplerFilterMode value);
std::string to_string(const SamplerMipmapMode value);
std::string to_string(const SamplerWrapMode value);
std::string to_string(const BufferLayoutType value);
std::string to_string(const ShaderBlockType value);
std::string to_string_glsl_decl(const BufferAttrType value);
std::string to_string(const BufferAttrType value);
std::string to_string(const IndexBufferDataType value);
std::string to_string(const BufferType value);
std::string to_string(const AccelerationStructureType value);
std::string to_string(const RasterSampleCount value);

/**
 * other useful helpers for enums
 **/

uint32_t pixelFormatDataSize(const PixelFormat pixelFormat);
bool is_color_pixel_format(const PixelFormat pixel_format);
bool is_blendable_pixel_format(PixelFormat pixel_format);
BufferAttrType get_float_equivalent_type(const BufferAttrType intype);
// Convert raster sample count enum to the number of samples
uint32_t raster_sample_count_enum_to_value(const RasterSampleCount enum_value);
RasterSampleCount value_to_raster_sample_count(const uint32_t value);
ResourceType resource_type_from_buffer_type(BufferType buffer_type);

/**
 * streaming
 **/

std::ostream& operator<<(std::ostream& os, const ResourceType value);
std::ostream& operator<<(std::ostream& os, const PixelFormat value);
std::ostream& operator<<(std::ostream& os, const ImageAspect value);
std::ostream& operator<<(std::ostream& os, const ImageLayout value);
std::ostream& operator<<(std::ostream& os, const BufferAccessType value);
std::ostream& operator<<(std::ostream& os, const SamplerFilterMode value);
std::ostream& operator<<(std::ostream& os, const SamplerMipmapMode value);
std::ostream& operator<<(std::ostream& os, const SamplerWrapMode value);
std::ostream& operator<<(std::ostream& os, const BufferLayoutType value);
std::ostream& operator<<(std::ostream& os, const ShaderBlockType value);
std::ostream& operator<<(std::ostream& os, const BufferAttrType value);
std::ostream& operator<<(std::ostream& os, const IndexBufferDataType value);
std::ostream& operator<<(std::ostream& os, const BufferType value);
std::ostream& operator<<(std::ostream& os, const AccelerationStructureType value);
std::ostream& operator<<(std::ostream& os, const RasterSampleCount value);

}  // namespace gfx

ENABLE_BITMASK_OPS(::gfx::ImageUsageBits);
ENABLE_BITMASK_OPS(::gfx::BufferUsageBits);
