/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/TypeGLSL.h"

namespace gfx {

class GlslStructBuilder {
 public:
  enum class Qualifier { kFlat };

  explicit GlslStructBuilder(const std::string& name);

  void addMember(const std::string& name,
                 BufferAttrType type,
                 const std::vector<Qualifier>& qualifiers = {},
                 bool is_array = false,
                 int array_size = 1);
  void addMember(const std::string& name,
                 const BaseTypeGLSL& type,
                 const std::vector<Qualifier>& qualifiers = {},
                 bool is_array = false,
                 int array_size = 1);

  // Add struct members for each attribute in a BufferLayout
  void addMembersFromLayout(const BaseBufferLayout& layout);

  // Create a copy of this builder with a unique name
  GlslStructBuilder clone(const std::string& new_name);

  // Create a struct string that is legal in GLSL and C++ (when using glm)
  // Ignores qualifiers
  std::string createStructString() const;

  // Create a string that can serve as a block definition in GLSL
  // Always includes qualifiers
  // do_auto_locations will add the `layout (location=N)` declaration to each member
  // and is useful for shader interface blocks (e.g. vertex to fragment)
  // reserved_locations will be skipped when do_auto_locations is enabled
  using locations_ref_type = std::reference_wrapper<const std::set<int>>;
  std::string createInterfaceBlockString(
      bool do_auto_locations,
      std::optional<locations_ref_type> reserved_locations = std::nullopt,
      bool is_geometry_shader = false) const;

 private:
  struct MemberInfo {
    std::string name;
    BufferAttrType type = BufferAttrType::kCOUNT;
    std::vector<Qualifier> qualifiers;
    bool is_array = false;
    int array_size = 1;
  };

  std::string name_;
  std::vector<MemberInfo> member_vector_;  // maintains ordering
  std::set<std::string> member_set_;       // enforces uniqueness
};

}  // namespace gfx
