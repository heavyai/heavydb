/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/Enums.h"

#include "GfxDriver/RenderError.h"
#include "Logger/Logger.h"

namespace gfx {

std::string to_string(const ResourceType value) {
  switch (value) {
    case ResourceType::kShaderProgram:
      return "Shader";
    case ResourceType::kShaderModule:
      return "ShaderModule";
    case ResourceType::kFramebuffer:
      return "Framebuffer";
    case ResourceType::kTexture:
      return "Texture";
    case ResourceType::kBaseBuffer:
      return "BaseBuffer";
    case ResourceType::kVertexBuffer:
      return "VertexBuffer";
    case ResourceType::kIndexBuffer:
      return "IndexBuffer";
    case ResourceType::kIndirectDrawVertexBuffer:
      return "IndirectDrawVertexBuffer";
    case ResourceType::kIndirectDrawIndexBuffer:
      return "IndirectDrawIndexBuffer";
    case ResourceType::kPixelBuffer:
      return "PixelBuffer";
    case ResourceType::kVertexArray:
      return "VertexArray";
    case ResourceType::kPipeline:
      return "Pipeline";
    case ResourceType::kRenderPass:
      return "RenderPass";
    case ResourceType::kAccelerationStructureBuffer:
      return "AccelerationStructureBuffer";
    case ResourceType::kAccelerationStructure:
      return "AccelerationStructure";
    case ResourceType::kShaderBindingTableBuffer:
      return "ShaderBindingTableBuffer";
    case ResourceType::kSlabWrapperBuffer:
      return "SlabWrapperBuffer";
    case ResourceType::kQueryPool:
      return "QueryPool";
  }
  return "";
}

std::string to_string(const PixelFormat value) {
  switch (value) {
    case PixelFormat::kR8:
      return "R8";
    case PixelFormat::kRG8:
      return "RG8";
    case PixelFormat::kRGBA8:
      return "RGBA8";
    case PixelFormat::kBGRA8:
      return "BGRA8";
    case PixelFormat::kR32UI:
      return "R32UI";
    case PixelFormat::kR32I:
      return "R32I";
    case PixelFormat::kDepth:
      return "Depth";
    case PixelFormat::kDepthHighP:
      return "Depth (High)";
    case PixelFormat::kDepthStencil:
      return "Depth/Stencil";
    case PixelFormat::kDepthStencilHighP:
      return "Depth/Stencil (High)";
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  return "";
}

std::string to_string(const ImageAspect value) {
  switch (value) {
    case ImageAspect::kColor:
      return "Color";
    case ImageAspect::kDepth:
      return "Depth";
    case ImageAspect::kStencil:
      return "Stencil";
    case ImageAspect::kDepthStencil:
      return "DepthStencil";
    case ImageAspect::kCOUNT:
      CHECK(false);
  }
  return "";
}

std::string to_string(const ImageLayout value) {
  switch (value) {
    case ImageLayout::kUndefined:
      return "Undefined";
    case ImageLayout::kGeneral:
      return "General";
    case ImageLayout::kAttachment:
      return "Attachment";
    case ImageLayout::kShaderReadOnly:
      return "ShaderReadOnly";
    case ImageLayout::kTransferSrc:
      return "TransferSrc";
    case ImageLayout::kTransferDst:
      return "TransferDst";
    case ImageLayout::kMapRead:
      return "MapRead";
    case ImageLayout::kMapWrite:
      return "MapWrite";
    case ImageLayout::kPresentSrc:
      return "PresentSrc";
  }
  UNREACHABLE();
  return "";
}

std::string to_string(const BufferAccessType value) {
  switch (value) {
    case BufferAccessType::kHostVisible:
      return "Host Visible";
    case BufferAccessType::kHostVisibleCached:
      return "Host Visible (Cached)";
    case BufferAccessType::kDeviceLocal:
      return "Device Local";
    case BufferAccessType::kExternalApi:
      return "External API Accessible";
  }
  return "";
}

std::string to_string(const SamplerFilterMode value) {
  switch (value) {
    case SamplerFilterMode::kNearest:
      return "Nearest";
    case SamplerFilterMode::kLinear:
      return "Linear";
  }
  return "";
}

std::string to_string(const SamplerMipmapMode value) {
  switch (value) {
    case SamplerMipmapMode::kNearest:
      return "Nearest";
    case SamplerMipmapMode::kLinear:
      return "Linear";
  }
  return "";
}

std::string to_string(const SamplerWrapMode value) {
  switch (value) {
    case SamplerWrapMode::kRepeat:
      return "Repeat";
    case SamplerWrapMode::kMirrorRepeat:
      return "Mirror Repeat";
    case SamplerWrapMode::kClampEdge:
      return "Clamp Edge";
    case SamplerWrapMode::kClampBorder:
      return "Clamp Border";
    default:
      CHECK(false);
  }
  return "";
}

std::string to_string(const BufferLayoutType value) {
  switch (value) {
    case BufferLayoutType::kInterleaved:
      return "Interleaved";
    case BufferLayoutType::kSequential:
      return "Sequential";
  }
  return "";
}

std::string to_string(const ShaderBlockType value) {
  switch (value) {
    case ShaderBlockType::kUniformBuffer:
      return "Uniform";
    case ShaderBlockType::kStorageBuffer:
      return "Storage";
  }
  return "";
}

std::string to_string_glsl_decl(const BufferAttrType value) {
  switch (value) {
    case BufferAttrType::kUint:
      return "uint";
    case BufferAttrType::kVec2ui:
      return "uvec2";
    case BufferAttrType::kVec3ui:
      return "uvec3";
    case BufferAttrType::kVec4ui:
      return "uvec4";
    case BufferAttrType::kInt:
      return "int";
    case BufferAttrType::kVec2i:
      return "ivec2";
    case BufferAttrType::kVec3i:
      return "ivec3";
    case BufferAttrType::kVec4i:
      return "ivec4";
    case BufferAttrType::kFloat:
      return "float";
    case BufferAttrType::kVec2f:
      return "vec2";
    case BufferAttrType::kVec3f:
      return "vec3";
    case BufferAttrType::kVec4f:
      return "vec4";
    case BufferAttrType::kDouble:
      return "double";
    case BufferAttrType::kVec2d:
      return "dvec2";
    case BufferAttrType::kVec3d:
      return "dvec3";
    case BufferAttrType::kVec4d:
      return "dvec4";
    case BufferAttrType::kUint64:
      return "uint64_t";
    case BufferAttrType::kVec2ui64:
      return "u64vec2";
    case BufferAttrType::kVec3ui64:
      return "u64vec3";
    case BufferAttrType::kVec4ui64:
      return "u64vec4";
    case BufferAttrType::kInt64:
      return "int64_t";
    case BufferAttrType::kVec2i64:
      return "i64vec2";
    case BufferAttrType::kVec3i64:
      return "i64vec3";
    case BufferAttrType::kVec4i64:
      return "i64vec4";
    case BufferAttrType::kMat3x2f:
      return "mat3x2";
    case BufferAttrType::kMat3x2d:
      return "dmat3x2";
    case BufferAttrType::kBool:
      return "bool";
    case BufferAttrType::kCOUNT:
      CHECK(false);
  }
  return "";
}

std::string to_string(const BufferAttrType value) {
  switch (value) {
    case BufferAttrType::kUint:
      return "UNSIGNED_INT";
    case BufferAttrType::kVec2ui:
      return "UNSIGNED_INT_VEC2";
    case BufferAttrType::kVec3ui:
      return "UNSIGNED_INT_VEC3";
    case BufferAttrType::kVec4ui:
      return "UNSIGNED_INT_VEC4";
    case BufferAttrType::kInt:
      return "INT";
    case BufferAttrType::kVec2i:
      return "INT_VEC2";
    case BufferAttrType::kVec3i:
      return "INT_VEC3";
    case BufferAttrType::kVec4i:
      return "INT_VEC4";
    case BufferAttrType::kFloat:
      return "FLOAT";
    case BufferAttrType::kVec2f:
      return "FLOAT_VEC2";
    case BufferAttrType::kVec3f:
      return "FLOAT_VEC3";
    case BufferAttrType::kVec4f:
      return "FLOAT_VEC4";
    case BufferAttrType::kDouble:
      return "DOUBLE";
    case BufferAttrType::kVec2d:
      return "DOUBLE_VEC2";
    case BufferAttrType::kVec3d:
      return "DOUBLE_VEC3";
    case BufferAttrType::kVec4d:
      return "DOUBLE_VEC4";
    case BufferAttrType::kUint64:
      return "UNSIGNED_INT64_ARB";
    case BufferAttrType::kVec2ui64:
      return "UNSIGNED_INT64_VEC2_ARB";
    case BufferAttrType::kVec3ui64:
      return "UNSIGNED_INT64_VEC3_ARB";
    case BufferAttrType::kVec4ui64:
      return "UNSIGNED_INT64_VEC4_ARB";
    case BufferAttrType::kInt64:
      return "INT64_ARB";
    case BufferAttrType::kVec2i64:
      return "INT64_VEC2_ARB";
    case BufferAttrType::kVec3i64:
      return "INT64_VEC3_ARB";
    case BufferAttrType::kVec4i64:
      return "INT64_VEC4_ARB";
    case BufferAttrType::kMat3x2f:
      return "FLOAT_MAT3x2";
    case BufferAttrType::kMat3x2d:
      return "DOUBLE_MAT3x2";
    case BufferAttrType::kBool:
      return "BOOL";
    case BufferAttrType::kCOUNT:
      CHECK(false);
  }
  return "";
}

std::string to_string(const IndexBufferDataType value) {
  switch (value) {
    case IndexBufferDataType::kUnsigned16:
      return "UNSIGNED_SHORT";
    case IndexBufferDataType::kUnsigned32:
      return "UNSIGNED_INT";
  }
  return "";
}

std::string to_string(const BufferType value) {
  switch (value) {
    case BufferType::kVertexBuffer:
      return "VERTEX";
    case BufferType::kPixelBuffer:
      return "PIXEL";
    case BufferType::kIndexBuffer:
      return "INDEX";
    case BufferType::kIndirectDrawVertexBuffer:
      return "INDIRECT_DRAW_VERTEX";
    case BufferType::kIndirectDrawIndexBuffer:
      return "INDIRECT_DRAW_INDEX";
    case BufferType::kAccelerationStructureBuffer:
      return "RAY_ACCELERATION_STRUCTURE";
    case BufferType::kShaderBindingTableBuffer:
      return "SHADER_BINDING_TABLE";
    case BufferType::kSlabWrapperBuffer:
      return "SLAB_WRAPPER";
    case BufferType::kUnspecified:
      return "UNSPECIFIED";
    case BufferType::kCOUNT:
      CHECK(false);
  }
  return "";
}

std::string to_string(const AccelerationStructureType value) {
  switch (value) {
    case AccelerationStructureType::kBottomLevel:
      return "BLAS";
    case AccelerationStructureType::kTopLevel:
      return "TLAS";
  }
  return "";
}

std::string to_string(const RasterSampleCount value) {
  CHECK_NE(value, RasterSampleCount::kCOUNT);
  static std::array<std::string, static_cast<int>(RasterSampleCount::kCOUNT)> string_lut =
      {"1", "2", "4", "8", "16", "32", "64"};
  return string_lut[static_cast<int>(value)];
}

uint32_t raster_sample_count_enum_to_value(const RasterSampleCount enum_value) {
  CHECK_NE(enum_value, RasterSampleCount::kCOUNT);
  static constexpr std::array<uint32_t, static_cast<int>(RasterSampleCount::kCOUNT)>
      value_lut = {1, 2, 4, 8, 16, 32, 64};
  return value_lut[static_cast<int>(enum_value)];
}

RasterSampleCount value_to_raster_sample_count(const uint32_t value) {
  switch (value) {
    case 1:
      return RasterSampleCount::k1;
    case 2:
      return RasterSampleCount::k2;
    case 4:
      return RasterSampleCount::k4;
    case 8:
      return RasterSampleCount::k8;
    case 16:
      return RasterSampleCount::k16;
    case 32:
      return RasterSampleCount::k32;
    case 64:
      return RasterSampleCount::k64;
  }
  CHECK(false) << "Invalid sample count. Valid values = 1, 2, 4, 8, 16, 32, 64";
  return RasterSampleCount::kCOUNT;
}

ResourceType resource_type_from_buffer_type(BufferType buffer_type) {
  switch (buffer_type) {
    case BufferType::kVertexBuffer:
      return ResourceType::kVertexBuffer;
    case BufferType::kPixelBuffer:
      return ResourceType::kPixelBuffer;
    case BufferType::kIndexBuffer:
      return ResourceType::kIndexBuffer;
    case BufferType::kIndirectDrawVertexBuffer:
      return ResourceType::kIndirectDrawVertexBuffer;
    case BufferType::kIndirectDrawIndexBuffer:
      return ResourceType::kIndirectDrawIndexBuffer;
    case BufferType::kAccelerationStructureBuffer:
      return ResourceType::kAccelerationStructureBuffer;
    case BufferType::kShaderBindingTableBuffer:
      return ResourceType::kShaderBindingTableBuffer;
    case BufferType::kSlabWrapperBuffer:
      return ResourceType::kSlabWrapperBuffer;
    case BufferType::kUnspecified:
      return ResourceType::kBaseBuffer;
    case BufferType::kCOUNT:
      break;
  }
  UNREACHABLE();
  return ResourceType{0};
}

uint32_t pixelFormatDataSize(const PixelFormat value) {
  switch (value) {
    case PixelFormat::kR8:
      return 1;
    case PixelFormat::kRG8:
      return 2;
    case PixelFormat::kRGBA8:
    case PixelFormat::kBGRA8:
      return 4;
    case PixelFormat::kR32UI:
      return 4;
    case PixelFormat::kR32I:
      return 4;
    case PixelFormat::kDepth:
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP:
      return 4;
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  return 0;
}

bool is_color_pixel_format(const PixelFormat value) {
  switch (value) {
    case PixelFormat::kR8:
    case PixelFormat::kRG8:
    case PixelFormat::kRGBA8:
    case PixelFormat::kBGRA8:
    case PixelFormat::kR32UI:
    case PixelFormat::kR32I:
      return true;
    case PixelFormat::kDepth:
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP:
      return false;
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  return false;
}

bool is_blendable_pixel_format(PixelFormat pixel_format) {
  switch (pixel_format) {
    case PixelFormat::kR8:
    case PixelFormat::kRG8:
    case PixelFormat::kRGBA8:
    case PixelFormat::kBGRA8:
      return true;
    case PixelFormat::kR32UI:
    case PixelFormat::kR32I:
    case PixelFormat::kDepth:
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP:
      return false;
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  UNREACHABLE();
  return false;
}

BufferAttrType get_float_equivalent_type(const BufferAttrType intype) {
  switch (intype) {
    case BufferAttrType::kInt:
    case BufferAttrType::kUint:
    case BufferAttrType::kFloat:
      return BufferAttrType::kFloat;
    case BufferAttrType::kInt64:
    case BufferAttrType::kUint64:
    case BufferAttrType::kDouble:
      return BufferAttrType::kDouble;
    default:
      THROW_RUNTIME_EX("Converting the buffer attr type " + to_string(intype) +
                       " to a floating point type is not supported.");
  }
}

std::ostream& operator<<(std::ostream& os, const ResourceType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const PixelFormat value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const ImageAspect value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const ImageLayout value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const BufferAccessType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const SamplerFilterMode value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const SamplerMipmapMode value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const SamplerWrapMode value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const BufferLayoutType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const ShaderBlockType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const BufferAttrType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const IndexBufferDataType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const BufferType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const AccelerationStructureType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const RasterSampleCount value) {
  os << to_string(value);
  return os;
}

}  // namespace gfx
