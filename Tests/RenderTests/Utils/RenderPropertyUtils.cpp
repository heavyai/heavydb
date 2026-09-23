/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/Utils/RenderPropertyUtils.h"

// Write type information for a property required by the shader to the stream
void stream_property_type_info(std::ostream& os, const PropertyInfo& info) {
  auto type_str = gfx::to_string_glsl_decl(info.type);
  auto enum_str = gfx::to_string(info.type);
  if (info.is_uniform) {
    os << "#define useU" << info.name << " 1\n";
  }
  os << "#define inT" << info.name << " " << type_str << "\n";
  os << "#define inT" << info.name << "Enum " << enum_str << "\n";
  os << "#define outT" << info.name << " " << type_str << "\n";
  os << "#define outT" << info.name << "Enum " << enum_str << "\n\n";
}

// Write a vertex attribute declaration to the stream
void stream_vertex_attr_decl(std::ostream& os,
                             const std::string& name,
                             gfx::BufferAttrType type,
                             int& location) {
  os << "layout (location = " << location << ") in " << to_string_glsl_decl(type) << " "
     << name << ";\n";
  location++;
};

void stream_property_getter(std::ostream& os, const PropertyInfo& info) {
  auto type_str =
      info.type == gfx::BufferAttrType::kBool ? "int" : to_string_glsl_decl(info.type);
  os << type_str << " get" << info.name << "(" << type_str << " " << info.name
     << ") {\n  return " << info.name << ";\n}\n";
}

// Create the 3 required injectors and populate them with the contents of 'props'
RenderPropInjectors create_render_prop_injectors(const std::vector<PropertyInfo>& props,
                                                 const std::string& ubo_name) {
  RenderPropInjectors rtn{ubo_name};
  int location = 0;
  for (auto const& prop : props) {
    stream_property_type_info(rtn.type_info_stream, prop);
    if (prop.is_uniform) {
      rtn.ubo_struct_builder.addMember(prop.name,
                                       prop.type == gfx::BufferAttrType::kBool
                                           ? gfx::BufferAttrType::kInt
                                           : prop.type);
    } else {
      stream_vertex_attr_decl(rtn.vertex_attr_stream, prop.name, prop.type, location);
    }
    stream_property_getter(rtn.getter_stream, prop);
  }
  return rtn;
};
