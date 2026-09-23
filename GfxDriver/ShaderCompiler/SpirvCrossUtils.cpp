/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/SpirvCrossUtils.h"

#include <iomanip>
#include <string>

// spirv-cross does not have a version define
// use glslang version as we update them together anyway
#include <glslang/build_info.h>

namespace gfx {

namespace {
using namespace spirv_cross;

std::string to_string(const SPIRType& type) {
  using spt = SPIRType;
  switch (type.basetype) {
    case spt::Unknown:
      return "Unknown";
    case spt::Void:
      return "Void";
    case spt::Boolean:
      return "Boolean";
    case spt::SByte:
      return "SByte";
    case spt::UByte:
      return "UByte";
    case spt::Short:
      return "Short";
    case spt::UShort:
      return "UShort";
    case spt::Int:
      return "Int";
    case spt::UInt:
      return "Uint";
    case spt::Int64:
      return "Int64";
    case spt::UInt64:
      return "Uint64";
    case spt::AtomicCounter:
      return "AtomicCounter";
    case spt::Half:
      return "Half";
    case spt::Float:
      return "Float";
    case spt::Double:
      return "Double";
    case spt::Struct:
      return "Struct";
    case spt::Image:
      return "Image";
    case spt::SampledImage:
      return "SampledImage";
    case spt::Sampler:
      return "Sampler";
    case spt::AccelerationStructure:
      return "AccelerationStructure";
    case spt::RayQuery:
      return "RayQuery";

    // spirv-cross internal
    case spt::ControlPointArray:
      return "ControlPointArray";
    case spt::Interpolant:
      return "Interpolant";
    case spt::Char:
      return "Char";
  }
  return "Invalid base type";
}

using ResourceVector = SmallVector<Resource>;

void write_spirv_struct(std::ostream& stream,
                        const Compiler& comp,
                        const SPIRType& struct_type,
                        int indent_level) {
  unsigned member_count = struct_type.member_types.size();
  for (unsigned i = 0; i < member_count; i++) {
    auto& member_type = comp.get_type(struct_type.member_types[i]);
    size_t member_size = comp.get_declared_struct_member_size(struct_type, i);
    const std::string& name = comp.get_member_name(struct_type.self, i);

    stream << std::left << std::setw(2 * indent_level) << ' ';
    std::string type_decl = to_string(member_type);

    size_t array_stride = 0;
    if (!member_type.array.empty()) {
      // Get array stride, (eg) float4 foo[]; will have an array stride of 16 bytes
      array_stride = comp.type_struct_member_array_stride(struct_type, i);
      // This is a bit hacky for array size (too tired to find the right accessor)
      type_decl = type_decl + "[" + std::to_string(member_size / array_stride) + "]";
    }

    stream << std::left << std::setw(10) << type_decl;
    stream << std::left << std::setw(35) << std::string("\"" + name + "\"");
    stream << "size: ";
    stream << std::left << std::setw(6) << member_size;

    // Get member offset within struct
    size_t offset = comp.type_struct_member_offset(struct_type, i);
    stream << "offset: ";
    stream << std::left << std::setw(6) << "offset: " << offset;

    if (!member_type.array.empty()) {
      stream << "  array stride: " << array_stride;
    }
    if (member_type.columns > 1) {
      // Get bytes stride between columns (if column major), for float 4x4 -> 16 bytes
      size_t matrix_stride = comp.type_struct_member_matrix_stride(struct_type, i);
      stream << "  matrix stride: " << matrix_stride;
    }

    stream << std::endl;

    if (member_type.basetype == SPIRType::Struct) {
      write_spirv_struct(stream, comp, member_type, indent_level + 1);
    }
  }
}

void write_spirv_buffer_resources(std::ostream& stream,
                                  const Compiler& comp,
                                  const ResourceVector& ubo_vec,
                                  const std::string& buf_type) {
  if (ubo_vec.size()) {
    stream << buf_type << ": " << ubo_vec.size() << "\n";
    for (auto const& res : ubo_vec) {
      unsigned set = comp.get_decoration(res.id, spv::DecorationDescriptorSet);
      unsigned binding = comp.get_decoration(res.id, spv::DecorationBinding);
      stream << "  [id=" << res.id << " set=" << set << " binding=" << binding << "] "
             << res.name;

      auto const& type = comp.get_type(res.base_type_id);
      size_t size = comp.get_declared_struct_size(type);
      stream << " [size: " << size << "]" << std::endl;

      write_spirv_struct(stream, comp, type, 2);

      stream << std::endl;
    }
    stream << std::endl;
  }
}

void write_spirv_stage_input_output(std::ostream& stream,
                                    Compiler& comp,
                                    ResourceVector& res_vec,
                                    bool is_output) {
  if (res_vec.size()) {
    stream << (is_output ? "Stage outputs:" : "Stage inputs: ");
    stream << res_vec.size() << std::endl;
    for (auto const& res : res_vec) {
      auto const& type = comp.get_type(res.base_type_id);
      stream << std::left << std::setw(8) << to_string(type) << " " << res.name
             << std::endl;
    }
    stream << std::endl;
  }
}

}  // namespace

void write_spirv_reflection(std::ostream& stream, Compiler& comp) {
  stream << "Reflection information (incomplete)\n" << std::endl;
  ShaderResources res = comp.get_shader_resources();
  write_spirv_stage_input_output(stream, comp, res.stage_inputs, false);
  write_spirv_stage_input_output(stream, comp, res.stage_outputs, true);
  write_spirv_buffer_resources(stream, comp, res.uniform_buffers, "Uniform Buffers");
  write_spirv_buffer_resources(stream, comp, res.storage_buffers, "Storage Buffers");
}

}  // namespace gfx
