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
    : shader_name_{shader_name}, num_vertex_attr_locations_{0} {}

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

void ShaderRedecorator::redecorate(spirv_t& spirv,
                                   ShaderReflection& reflection,
                                   const std::string& template_name) {
  int set = 0;
  try {
    redecorateInternal(spirv, set, reflection);
  } catch (spirv_cross::CompilerError& e) {
    THROW_RUNTIME_EX("Failure during redecoration of shader '" + template_name +
                     "' (SPIRV-Cross exception: " + e.what() + ")");
  } catch (std::exception& e) {
    THROW_RUNTIME_EX("Failure during redecoration of shader '" + template_name + "' (" +
                     e.what() + ")");
  }
}

void ShaderRedecorator::redecorateInternal(spirv_t& spirv,
                                           int set,
                                           ShaderReflection& reflection) {
  // pass the blob to a compiler and get the resources
  spirv_cross::CompilerGLSL compiler(spirv);
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
  auto update_decoration = [&compiler, &spirv](uint32_t id,
                                               const ResourceType resource_type,
                                               std::string_view name,
                                               const spv::Decoration decoration,
                                               const uint32_t decoration_value) {
    uint32_t offset = 0;
    CHECK(compiler.get_binary_offset_for_decoration(id, decoration, offset))
        << "    " << resource_type_to_string(resource_type) << " '" << name
        << "' has no binary offset for " << decoration_to_string(decoration)
        << " decoration!";
    spirv[offset] = decoration_value;
  };
  auto get_resource_count = [&compiler](uint32_t type_id) -> uint32_t {
    const auto& type = compiler.get_type(type_id);
    CHECK(type.array.size() < 2) << "Multi-dimensional arrays not supported!";
    return type.array.size() ? type.array[0] : 1;
  };

  // capture vertex stage input attributes and allocate locations
  if (execution_model == spv::ExecutionModelVertex) {
    if (resources.stage_inputs.size()) {
      LOG_IF(INFO, DEBUG_LOG_REFLECTION)
          << "  " << resources.stage_inputs.size() << " Vertex Inputs";
      for (auto& input : resources.stage_inputs) {
        check_has_decoration(
            input.id, ResourceType::kVertexAttr, input.name, spv::DecorationLocation);

        // how many locations?
        uint32_t num_locations = get_resource_count(input.type_id);

        // allocate location(s)
        const auto vertex_attr_location = num_vertex_attr_locations_;
        num_vertex_attr_locations_ += num_locations;

        // write the new location back to the SPIRV blob
        update_decoration(input.id,
                          ResourceType::kVertexAttr,
                          input.name,
                          spv::DecorationLocation,
                          vertex_attr_location);

        // and store in reflection
        reflection.addVertexAttr(input.name, vertex_attr_location, num_locations);

        // log
        LOG_IF(INFO, DEBUG_LOG_REFLECTION)
            << "    Vertex Attr '" << input.name << array_suffix(num_locations)
            << "' allocated location " << vertex_attr_location;
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

  auto reserve_bindings_for_resources =
      [&](const spirv_cross::SmallVector<spirv_cross::Resource>& resources,
          ResourceType resource_type) {
        LOG_IF(INFO, DEBUG_LOG_REFLECTION)
            << "  " << resources.size() << " " << resource_type_to_string(resource_type)
            << "s";
        for (auto& resource : resources) {
          // get any existing binding
          auto const existing_binding = check_has_decoration(
              resource.id, resource_type, resource.name, spv::DecorationBinding);

          // reserve?
          if (existing_binding != kUninitializedBinding) {
            auto const num_bindings = get_resource_count(resource.type_id);
            reserveBindings(
                set, existing_binding, num_bindings, resource_type, resource.name);
          }
        }
      };

  auto allocate_and_decorate_bindings_for_buffers =
      [&](const spirv_cross::SmallVector<spirv_cross::Resource>& buffers,
          ResourceType resource_type) {
        for (auto& buffer : buffers) {
          // validate set (Vulkan only) and binding
          auto const existing_set = check_has_decoration(
              buffer.id, resource_type, buffer.name, spv::DecorationDescriptorSet);
          auto const existing_binding = check_has_decoration(
              buffer.id, resource_type, buffer.name, spv::DecorationBinding);

          // how many bindings?
          auto const num_bindings = get_resource_count(buffer.type_id);

          // allocate new set and binding(s) or keep existing
          uint32_t buffer_set{0U}, buffer_binding{0U};
          if (existing_set == kUninitializedSet) {
            CHECK_GE(set, 0);
            buffer_set = static_cast<uint32_t>(set);
          } else {
            buffer_set = existing_set;
          }
          if (existing_binding == kUninitializedBinding) {
            buffer_binding =
                allocateBindings(set, num_bindings, resource_type, buffer.name);
          } else {
            buffer_binding = existing_binding;
          }

          // write the new set and binding back to the SPIRV blob
          update_decoration(buffer.id,
                            resource_type,
                            buffer.name,
                            spv::DecorationDescriptorSet,
                            buffer_set);
          update_decoration(buffer.id,
                            resource_type,
                            buffer.name,
                            spv::DecorationBinding,
                            buffer_binding);

          // capture buffer attributes
          auto& buffer_type = compiler.get_type(buffer.base_type_id);
          size_t buffer_size = compiler.get_declared_struct_size(buffer_type);
          uint32_t buffer_member_count = buffer_type.member_types.size();

          // and store in reflection
          int reflection_set = static_cast<int>(buffer_set);
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
              << array_suffix(num_bindings) << "' allocated binding " << buffer_binding
              << ", requires " << buffer_size << " bytes and has " << buffer_member_count
              << " members";

          // iterate members
          for (uint32_t i = 0; i < buffer_member_count; i++) {
            auto const& member_type = compiler.get_type(buffer_type.member_types[i]);
            auto const& member_name = compiler.get_member_name(buffer_type.self, i);
            auto const member_offset = compiler.type_struct_member_offset(buffer_type, i);
            auto const member_size =
                compiler.get_declared_struct_member_size(buffer_type, i);

            if (member_type.basetype == spirv_cross::SPIRType::Struct) {
              LOG_IF(INFO, DEBUG_LOG_REFLECTION) << "      Struct '" << member_name
                                                 << "' has total size " << member_size;

              // iterate children and flatten
              uint32_t struct_member_count = member_type.member_types.size();
              for (uint32_t j = 0; j < struct_member_count; j++) {
                // get child
                auto const& child_member_type =
                    compiler.get_type(member_type.member_types[j]);
                auto const& child_member_name =
                    compiler.get_member_name(member_type.self, j);
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
                reflection.addShaderStorageBufferAttr(member_name,
                                                      reflection_set,
                                                      buffer_binding,
                                                      member_offset,
                                                      member_size);
              } else {
                reflection.addUniformBufferAttr(member_name,
                                                reflection_set,
                                                buffer_binding,
                                                member_offset,
                                                member_size);
              }

              LOG_IF(INFO, DEBUG_LOG_REFLECTION)
                  << "      '" << member_name << "' has offset " << member_offset
                  << ", size " << member_size;
            }
          }
        }
      };

  auto allocate_and_decorate_bindings_for_opaque_uniforms =
      [&](const spirv_cross::SmallVector<spirv_cross::Resource>& resources,
          ResourceType resource_type) {
        for (auto& resource : resources) {
          auto const existing_set = check_has_decoration(
              resource.id, resource_type, resource.name, spv::DecorationDescriptorSet);
          auto const existing_binding = check_has_decoration(
              resource.id, resource_type, resource.name, spv::DecorationBinding);

          // how many bindings?
          auto const num_bindings = get_resource_count(resource.type_id);

          // allocate new set and binding(s) or keep existing
          uint32_t uniform_set{0U}, uniform_binding{0U};
          if (existing_set == kUninitializedSet) {
            CHECK_GE(set, 0);
            uniform_set = static_cast<uint32_t>(set);
          } else {
            uniform_set = existing_set;
          }
          if (existing_binding == kUninitializedBinding) {
            uniform_binding =
                allocateBindings(set, num_bindings, resource_type, resource.name);
          } else {
            uniform_binding = existing_binding;
          }

          // log
          LOG_IF(INFO, DEBUG_LOG_REFLECTION)
              << "    " << resource_type_to_string(resource_type) << " '" << resource.name
              << array_suffix(num_bindings) << "' allocated uniform binding "
              << uniform_binding;

          // write the new set and binding back to the SPIRV blob
          // update set and binding
          update_decoration(resource.id,
                            resource_type,
                            resource.name,
                            spv::DecorationDescriptorSet,
                            uniform_set);
          update_decoration(resource.id,
                            resource_type,
                            resource.name,
                            spv::DecorationBinding,
                            uniform_binding);

          // and store in reflection
          int reflection_set = static_cast<int>(uniform_set);
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

  // reserve any pre-specified bindings for all resources
  reserve_bindings_for_resources(resources.uniform_buffers, ResourceType::kUniformBuffer);
  reserve_bindings_for_resources(resources.storage_buffers,
                                 ResourceType::kShaderStorageBuffer);
  reserve_bindings_for_resources(resources.sampled_images, ResourceType::kSampledImage);
  reserve_bindings_for_resources(resources.storage_images, ResourceType::kStorageImage);
  reserve_bindings_for_resources(resources.acceleration_structures,
                                 ResourceType::kAccelerationStructure);

  // allocate and decorate new bindings
  allocate_and_decorate_bindings_for_buffers(resources.uniform_buffers,
                                             ResourceType::kUniformBuffer);
  allocate_and_decorate_bindings_for_buffers(resources.storage_buffers,
                                             ResourceType::kShaderStorageBuffer);
  allocate_and_decorate_bindings_for_opaque_uniforms(resources.sampled_images,
                                                     ResourceType::kSampledImage);
  allocate_and_decorate_bindings_for_opaque_uniforms(resources.storage_images,
                                                     ResourceType::kStorageImage);
  allocate_and_decorate_bindings_for_opaque_uniforms(
      resources.acceleration_structures, ResourceType::kAccelerationStructure);

  // Vulkan things that we shouldn't see with our shaders (yet)
  CHECK_EQ(resources.separate_images.size(), 0U);
  CHECK_EQ(resources.separate_samplers.size(), 0U);
  CHECK_EQ(resources.subpass_inputs.size(), 0U);
}

ShaderRedecorator::ReservedBindings& ShaderRedecorator::getReservedBindings(
    int set,
    ResourceType resource_type) {
  // otherwise we're in in Vulkan mode, and all the bindings are in one map per set
  // find or create an entry for this set and return it
  auto const itr = reserved_vulkan_bindings_.try_emplace(set, ReservedBindings()).first;
  CHECK(itr != reserved_vulkan_bindings_.end());
  return (*itr).second;
}

void ShaderRedecorator::reserveBindings(int set,
                                        uint32_t first_binding,
                                        uint32_t num_bindings,
                                        ResourceType resource_type,
                                        const std::string& resource_name) {
  // will the whole range fit?
  CHECK_LE(first_binding + num_bindings, kMaxBindingsPerSet)
      << "Binding overflow (set " << set
      << ", " + resource_type_to_string(resource_type) + " '" << resource_name
      << "', shader '" << shader_name_ << "')";

  // find or create the reserved bindings map for this set
  auto& reserved_bindings = getReservedBindings(set, resource_type);

  // reserve the range, unless there's a clash
  for (uint32_t i = first_binding; i < first_binding + num_bindings; i++) {
    ReservedBindingEntry new_entry{resource_type, resource_name, i - first_binding};
    auto const [itr, inserted] = reserved_bindings.try_emplace(i, new_entry);
    if (!inserted && new_entry != itr->second) {
      std::string range_msg = (num_bindings > 1u)
                                  ? "in range " + std::to_string(first_binding) + " to " +
                                        std::to_string(first_binding + num_bindings - 1)
                                  : "at " + std::to_string(first_binding);
      auto const existing_type = std::get<0>(itr->second);
      auto const existing_name = std::get<1>(itr->second);
      uint32_t max_index{0u};
      for (auto const& entry : reserved_bindings) {
        max_index = std::max(max_index, std::get<2>(entry.second));
      }
      auto const first_available_binding = first_binding + max_index + 1;
      THROW_RUNTIME_EX("Binding clash " + range_msg + ", Set " + std::to_string(set) +
                       ", Shader '" + shader_name_ + "', " +
                       resource_type_to_string(resource_type) + " '" + resource_name +
                       "' clashes with " + resource_type_to_string(existing_type) + " '" +
                       existing_name + "'. Next available binding is " +
                       std::to_string(first_available_binding));
    }
  }
}

uint32_t ShaderRedecorator::allocateBindings(int set,
                                             uint32_t num_bindings,
                                             ResourceType resource_type,
                                             const std::string& resource_name) {
  // find or create the reserved bindings map for this set
  auto& reserved_bindings = getReservedBindings(set, resource_type);

  // find available binding range
  uint32_t first_binding{0U};
  while (first_binding < kMaxBindingsPerSet) {
    // will the whole range fit?
    CHECK_LE(first_binding + num_bindings, kMaxBindingsPerSet)
        << "Binding overflow (set " << set
        << ", " + resource_type_to_string(resource_type) + " '" << resource_name
        << "', shader '" << shader_name_ << "')";

    // check the required range of bindings is available
    bool range_available{true};
    uint32_t first_unavailable{0U};
    auto const itr = reserved_bindings.lower_bound(first_binding);
    if (itr != reserved_bindings.end() && itr->first < first_binding + num_bindings) {
      range_available = false;
      first_unavailable = itr->first;
    }

    // if so, reserve the whole range and we're done
    // otherwise restart search after first unavailable
    if (range_available) {
      for (uint32_t j = first_binding; j < first_binding + num_bindings; j++) {
        CHECK(reserved_bindings
                  .try_emplace(
                      j, std::make_tuple(resource_type, resource_name, j - first_binding))
                  .second);
      }
      return first_binding;
    } else {
      first_binding = first_unavailable + 1u;
    }
  }

  // unavailable
  THROW_RUNTIME_EX("Failed to allocate " + std::to_string(num_bindings) +
                   " bindings (set " + std::to_string(set) + ", " +
                   resource_type_to_string(resource_type) + " '" + resource_name +
                   "', shader '" + shader_name_ + "')");
}

}  // namespace gfx
