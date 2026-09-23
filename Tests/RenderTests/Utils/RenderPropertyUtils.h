/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"

//
// Helper functions for handling QueryRenderer::RenderProperty mocking in tests
//

// Basic information about the RenderProperty
// create a vector of these to pass to the various functions
struct PropertyInfo {
  std::string name;
  bool is_uniform{false};
  gfx::BufferAttrType type{gfx::BufferAttrType::kCOUNT};
};

// Write type information for a property required by the shader to the stream
// Injects 4 defines for all properties:
//   #define inT<name>
//   #define inT<name>Enum
//   #define outT<name>
//   #define outT<name>Enum
// Injects 1 additional define for uniform properties
//   #define useU<name> 1
void stream_property_type_info(std::ostream& os, const PropertyInfo& info);

// Write a vertex attribute declaration to the stream
// Output is in the form
//   layout (location = N) in <glsldecl> <name>
// Example
//   layout (location = 1) in vec4 fillColor;
void stream_vertex_attr_decl(std::ostream& os,
                             const std::string& name,
                             gfx::BufferAttrType type,
                             int& location);

// Write a get* function for the property to the stream
// Output is in the form
//   <glsldecl> get<name>(<glsldecl> <name>) {
//     return <name>;
//   }
// Example
//   float getsize(float size) {
//     return size;
//   }
void stream_property_getter(std::ostream& os, const PropertyInfo& info);

// Struct containing the various injection builders required to build a functional shader
struct RenderPropInjectors {
  RenderPropInjectors(const std::string& ubo_name) : ubo_struct_builder{ubo_name} {}
  std::stringstream type_info_stream;
  std::stringstream vertex_attr_stream;
  std::stringstream getter_stream;
  gfx::GlslStructBuilder ubo_struct_builder;
};

// Create the 3 required injectors and populate them with the contents of 'props'
RenderPropInjectors create_render_prop_injectors(const std::vector<PropertyInfo>& props,
                                                 const std::string& ubo_name);
