/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index_container.hpp>

#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/Data/Types.h"
#include "Shared/sqltypes.h"

namespace QueryRenderer {

struct LayoutAttrInfo {
  const std::string attr_name;
  const SQLTypeInfo type_info;
  const gfx::BufferLayoutShPtr buffer_layout;

  // NOTE: this constructor is currently needed for std container emplace-like methods
  // only. It could be removed and the emplace methods replaced to push-like methods.
  LayoutAttrInfo(const std::string& in_attr_name,
                 const SQLTypeInfo& in_type_info,
                 const gfx::BufferLayoutShPtr& in_buffer_layout)
      : attr_name(in_attr_name)
      , type_info(in_type_info)
      , buffer_layout(in_buffer_layout) {}

  static QueryDataType getTypeFromLayoutAttr(const LayoutAttrInfo& attr_info);
};

// NOTE: this could be a std::unordered_set in c++20 when "equivalent" key
// comparisons become available
using LayoutAttrInfoSet = ::boost::multi_index_container<
    LayoutAttrInfo,
    ::boost::multi_index::indexed_by<::boost::multi_index::hashed_unique<
        ::boost::multi_index::
            member<LayoutAttrInfo, const std::string, &LayoutAttrInfo::attr_name>>>>;

}  // namespace QueryRenderer
