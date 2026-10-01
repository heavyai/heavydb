/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/ShaderRedecorator.h"

#include <string_view>

#include <spirv_cross/spirv.hpp>
#include <spirv_cross/spirv_glsl.hpp>
#include <spirv_cross/spirv_parser.hpp>

#include "GfxDriver/ShaderCompiler/ShaderReflection.h"
#include "Logger/Logger.h"

#define DEBUG_LOG_REFLECTION false

namespace gfx {

ShaderRedecorator::ShaderRedecorator(std::string_view shader_name)
    : shader_name_{shader_name} {}

namespace {

void validate_buffer_attr_type(const spirv_cross::SPIRType::BaseType& base_type) {
  switch (base_type) {
    // these are the only types we currently support in UBO and SSBO
    case spirv_cross::SPIRType::BaseType::Int:
    case spirv_cross::SPIRType::BaseType::UInt:
    case spirv_cross::SPIRType::BaseType::Int64:
    case spirv_cross::SPIRType::BaseType::UInt64:
    case spirv_cross::SPIRType::BaseType::Float:
    case spirv_cross::SPIRType::BaseType::Double:
      return;
    default:
      break;
  }
  CHECK(false) << "Unsupported type (" << (int)base_type << ") in UBO/SSBO";
}

std::string execution_model_to_string(spv::ExecutionModel execution_model) {
  switch (execution_model) {
    case spv::ExecutionModelVertex:
      return "Vertex";
    case spv::ExecutionModelFragment:
      return "Fragment";
    case spv::ExecutionModelGeometry:
      return "Geometry";
    case spv::ExecutionModelRayGenerationKHR:
      return "RayGeneration";
    case spv::ExecutionModelIntersectionKHR:
      return "Intersection";
    case spv::ExecutionModelAnyHitKHR:
      return "AnyHit";
    case spv::ExecutionModelClosestHitKHR:
      return "ClosestHit";
    case spv::ExecutionModelMissKHR:
      return "Miss";
    case spv::ExecutionModelCallableKHR:
      return "Callable";
    default:
      return "spv::ExecutionModel(" + std::to_string(execution_model) + ")";
  }
}

std::string array_suffix(uint32_t array_size) {
  std::string result;
  if (array_size > 0) {
    result = "[" + std::to_string(array_size) + "]";
  }
  return result;
}

std::string resource_type_to_string(ShaderRedecorator::ResourceType resource_type) {
  switch (resource_type) {
    case ShaderRedecorator::ResourceType::kVertexAttr:
      return "Vertex Attr";
    case ShaderRedecorator::ResourceType::kUniformBuffer:
      return "Uniform Buffer";
    case ShaderRedecorator::ResourceType::kShaderStorageBuffer:
      return "Shader Storage Buffer";
    case ShaderRedecorator::ResourceType::kSampledImage:
      return "Sampler";
    case ShaderRedecorator::ResourceType::kStorageImage:
      return "Storage Image";
    case ShaderRedecorator::ResourceType::kAccelerationStructure:
      return "Acceleration Structure";
    default:
      return "UNKNOWN";
  }
}

std::string decoration_to_string(spv::Decoration decoration) {
  switch (decoration) {
    case spv::DecorationDescriptorSet:
      return "Set";
    case spv::DecorationBinding:
      return "Binding";
    case spv::DecorationLocation:
      return "Location";
    default:
      return "UNKNOWN";
  }
}

}  // namespace

void ShaderRedecorator::redecorate(const spirv_t& spirv,
                                   ShaderReflection& reflection,
                                   const std::string& template_name) {
  try {
    redecorateInternal(spirv, reflection);
  } catch (spirv_cross::CompilerError& e) {
    THROW_RUNTIME_EX("Failure during redecoration of shader '" + template_name +
                     "' (SPIRV-Cross exception: " + e.what() + ")");
  } catch (std::exception& e) {
    THROW_RUNTIME_EX("Failure during redecoration of shader '" + template_name + "' (" +
                     e.what() + ")");
  }
}

void ShaderRedecorator::redecorateInternal(const spirv_t& spirv,
                                           ShaderReflection& reflection) {
  // pass the blob to a compiler and get the resources
  spirv_cross::CompilerGLSL compiler(spirv.data(), spirv.size());
  spirv_cross::ShaderResources resources = compiler.get_shader_resources();

  // blob should have only one entry point
  auto entry_points_and_stages = compiler.get_entry_points_and_stages();
  CHECK(entry_points_and_stages.size() == 1);
  const auto& execution_model = entry_points_and_stages[0].execution_model;

  // log blob
  LOG_IF(INFO, DEBUG_LOG_REFLECTION)
      << "  Redecorating SPIR-V blob for " << execution_model_to_string(execution_model)
      << " shader '" << shader_name_ << "' (" << std::to_string(spirv.size())
      << " words)";

  // initialize the reflection
  reflection.initialize();

  // helper lambdas
  auto check_has_decoration = [&compiler](uint32_t id,
                                          const ResourceType resource_type,
                                          std::string_view name,
                                          const spv::Decoration decoration) -> uint32_t {
    CHECK(compiler.has_decoration(id, decoration))
        << "    " << resource_type_to_string(resource_type) << " '" << name
        << "' has no existing " << decoration_to_string(decoration) << " decoration!";
    return compiler.get_decoration(id, decoration);
  };
  auto get_resource_count = [&compiler](uint32_t type_id) -> uint32_t {
    const auto& type = compiler.get_type(type_id);
    CHECK(type.array.size() < 2) << "Multi-dimensional arrays not supported!";
    return type.array.size() ? type.array[0] : 1;
  };

  // capture vertex stage input attributes, at the locations glslang gave them
  if (execution_model == spv::ExecutionModelVertex) {
    if (resources.stage_inputs.size()) {
      LOG_IF(INFO, DEBUG_LOG_REFLECTION)
          << "  " << resources.stage_inputs.size() << " Vertex Inputs";
      for (auto& input : resources.stage_inputs) {
        auto const vertex_attr_location = check_has_decoration(
            input.id, ResourceType::kVertexAttr, input.name, spv::DecorationLocation);

        // how many locations?
        uint32_t num_locations = get_resource_count(input.type_id);

        // store in reflection
        reflection.addVertexAttr(input.name, vertex_attr_location, num_locations);

        // log
        LOG_IF(INFO, DEBUG_LOG_REFLECTION)
            << "    Vertex Attr '" << input.name << array_suffix(num_locations)
            << "' has location " << vertex_attr_location;
      }
    } else {
      LOG_IF(INFO, DEBUG_LOG_REFLECTION) << "  No Vertex Inputs";
    }
  } else if (execution_model == spv::ExecutionModelFragment) {
    if (resources.stage_outputs.size()) {
      for (auto& output : resources.stage_outputs) {
        reflection.addFragmentShaderOutputLocation(
            compiler.get_decoration(output.id, spv::DecorationLocation));
      }
    }
  }

  // Reads the set and binding glslang assigned, checking that the set is the only one
  // we support and that the binding is not already another resource's
  auto claim_binding = [&](const spirv_cross::Resource& resource,
                           ResourceType resource_type) -> uint32_t {
    auto const assigned_set = check_has_decoration(
        resource.id, resource_type, resource.name, spv::DecorationDescriptorSet);
    auto const assigned_binding = check_has_decoration(
        resource.id, resource_type, resource.name, spv::DecorationBinding);

    // No shader declares an explicit set, and the resolver assigns 0 without one, so
    // every resource lands in kDescriptorSet. Assert instead of branching, so that a
    // shader which does declare one fails loudly rather than taking a path nothing
    // exercises.
    CHECK_EQ(assigned_set, kDescriptorSet)
        << "    " << resource_type_to_string(resource_type) << " '" << resource.name
        << "' is in descriptor set " << assigned_set << ", which is not supported";

    // An array of resources is one binding with a descriptor count, not a binding per
    // element, so its size does not come into this. It still reaches the reflection,
    // which is where VulkanMaterial reads the count from.
    recordBinding(assigned_binding, resource_type, resource.name);
    return assigned_binding;
  };

  auto reflect_buffers = [&](const spirv_cross::SmallVector<spirv_cross::Resource>&
                                 buffers,
                             ResourceType resource_type) {
    LOG_IF(INFO, DEBUG_LOG_REFLECTION)
        << "  " << buffers.size() << " " << resource_type_to_string(resource_type) << "s";
    for (auto& buffer : buffers) {
      auto const buffer_binding = claim_binding(buffer, resource_type);
      auto const num_bindings = get_resource_count(buffer.type_id);

      // capture buffer attributes
      auto& buffer_type = compiler.get_type(buffer.base_type_id);
      size_t buffer_size = compiler.get_declared_struct_size(buffer_type);
      uint32_t buffer_member_count = buffer_type.member_types.size();

      // and store in reflection
      int reflection_set = static_cast<int>(kDescriptorSet);
      if (resource_type == ResourceType::kShaderStorageBuffer) {
        reflection.addShaderStorageBuffer(
            buffer.name, reflection_set, buffer_binding, buffer_size);
      } else {
        reflection.addUniformBuffer(
            buffer.name, reflection_set, buffer_binding, buffer_size);
      }

      // log
      LOG_IF(INFO, DEBUG_LOG_REFLECTION)
          << "    " << resource_type_to_string(resource_type) << " '" << buffer.name
          << array_suffix(num_bindings) << "' has binding " << buffer_binding
          << ", requires " << buffer_size << " bytes and has " << buffer_member_count
          << " members";

      // iterate members
      for (uint32_t i = 0; i < buffer_member_count; i++) {
        auto const& member_type = compiler.get_type(buffer_type.member_types[i]);
        auto const& member_name = compiler.get_member_name(buffer_type.self, i);
        auto const member_offset = compiler.type_struct_member_offset(buffer_type, i);
        auto const member_size = compiler.get_declared_struct_member_size(buffer_type, i);

        if (member_type.basetype == spirv_cross::SPIRType::Struct) {
          LOG_IF(INFO, DEBUG_LOG_REFLECTION)
              << "      Struct '" << member_name << "' has total size " << member_size;

          // iterate children and flatten
          uint32_t struct_member_count = member_type.member_types.size();
          for (uint32_t j = 0; j < struct_member_count; j++) {
            // get child
            auto const& child_member_type =
                compiler.get_type(member_type.member_types[j]);
            auto const& child_member_name = compiler.get_member_name(member_type.self, j);
            auto const child_member_offset =
                compiler.type_struct_member_offset(member_type, j);
            auto const child_member_size =
                compiler.get_declared_struct_member_size(member_type, j);

            // validate type
            CHECK(child_member_type.basetype != spirv_cross::SPIRType::Struct)
                << "Nested structs not supported in UBO/SSBO";
            validate_buffer_attr_type(child_member_type.basetype);

            // full name and offset
            // add prefix only for UBO elements
            // SSBO elements are referred to only by child name
            std::string child_full_name = child_member_name;
            if (resource_type == ResourceType::kUniformBuffer) {
              child_full_name = member_name + "." + child_full_name;
            }
            uint32_t child_full_offset = member_offset + child_member_offset;

            // store in reflection
            if (resource_type == ResourceType::kShaderStorageBuffer) {
              reflection.addShaderStorageBufferAttr(child_full_name,
                                                    reflection_set,
                                                    buffer_binding,
                                                    child_full_offset,
                                                    child_member_size);
            } else {
              reflection.addUniformBufferAttr(child_full_name,
                                              reflection_set,
                                              buffer_binding,
                                              child_full_offset,
                                              child_member_size);
            }

            LOG_IF(INFO, DEBUG_LOG_REFLECTION)
                << "        '" << child_full_name << "' has offset "
                << child_member_offset << ", size " << child_member_size;
          }
        } else {
          // validate type
          validate_buffer_attr_type(member_type.basetype);

          // store in reflection
          if (resource_type == ResourceType::kShaderStorageBuffer) {
            reflection.addShaderStorageBufferAttr(
                member_name, reflection_set, buffer_binding, member_offset, member_size);
          } else {
            reflection.addUniformBufferAttr(
                member_name, reflection_set, buffer_binding, member_offset, member_size);
          }

          LOG_IF(INFO, DEBUG_LOG_REFLECTION)
              << "      '" << member_name << "' has offset " << member_offset << ", size "
              << member_size;
        }
      }
    }
  };

  auto reflect_opaque_uniforms =
      [&](const spirv_cross::SmallVector<spirv_cross::Resource>& resources,
          ResourceType resource_type) {
        LOG_IF(INFO, DEBUG_LOG_REFLECTION)
            << "  " << resources.size() << " " << resource_type_to_string(resource_type)
            << "s";
        for (auto& resource : resources) {
          auto const uniform_binding = claim_binding(resource, resource_type);
          auto const num_bindings = get_resource_count(resource.type_id);

          // log
          LOG_IF(INFO, DEBUG_LOG_REFLECTION)
              << "    " << resource_type_to_string(resource_type) << " '" << resource.name
              << array_suffix(num_bindings) << "' has uniform binding "
              << uniform_binding;

          // store in reflection
          int reflection_set = static_cast<int>(kDescriptorSet);
          if (resource_type == ResourceType::kSampledImage) {
            reflection.addSampler(
                resource.name, reflection_set, uniform_binding, num_bindings);
          } else if (resource_type == ResourceType::kStorageImage) {
            reflection.addStorageImage(
                resource.name, reflection_set, uniform_binding, num_bindings);
          } else if (resource_type == ResourceType::kAccelerationStructure) {
            reflection.addAccelerationStructure(
                resource.name, reflection_set, uniform_binding, num_bindings);
          }
        }
      };

  // One pass now, rather than reserving the explicit bindings before allocating the
  // rest, because every binding arrives already assigned
  reflect_buffers(resources.uniform_buffers, ResourceType::kUniformBuffer);
  reflect_buffers(resources.storage_buffers, ResourceType::kShaderStorageBuffer);
  reflect_opaque_uniforms(resources.sampled_images, ResourceType::kSampledImage);
  reflect_opaque_uniforms(resources.storage_images, ResourceType::kStorageImage);
  reflect_opaque_uniforms(resources.acceleration_structures,
                          ResourceType::kAccelerationStructure);

  // Vulkan things that we shouldn't see with our shaders (yet)
  CHECK_EQ(resources.separate_images.size(), 0U);
  CHECK_EQ(resources.separate_samplers.size(), 0U);
  CHECK_EQ(resources.subpass_inputs.size(), 0U);
}

void ShaderRedecorator::recordBinding(uint32_t binding,
                                      ResourceType resource_type,
                                      const std::string& resource_name) {
  CHECK_LT(binding, kMaxBindingsPerSet)
      << "Binding overflow (set " << kDescriptorSet
      << ", " + resource_type_to_string(resource_type) + " '" << resource_name
      << "', shader '" << shader_name_ << "')";

  ReservedBindingEntry new_entry{resource_type, resource_name};
  auto const [itr, inserted] = reserved_bindings_.try_emplace(binding, new_entry);
  if (!inserted && new_entry != itr->second) {
    auto const existing_type = std::get<0>(itr->second);
    auto const existing_name = std::get<1>(itr->second);
    // One past everything claimed so far, which is where a shader author can safely
    // move whichever of the two resources they choose to renumber
    auto const first_available_binding = reserved_bindings_.rbegin()->first + 1u;
    THROW_RUNTIME_EX("Binding clash at " + std::to_string(binding) + ", Set " +
                     std::to_string(kDescriptorSet) + ", Shader '" + shader_name_ +
                     "', " + resource_type_to_string(resource_type) + " '" +
                     resource_name + "' clashes with " +
                     resource_type_to_string(existing_type) + " '" + existing_name +
                     "'. Next available binding is " +
                     std::to_string(first_available_binding));
  }
}

}  // namespace gfx
