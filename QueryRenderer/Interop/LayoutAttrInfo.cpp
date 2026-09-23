/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Interop/ThrustInteropBuffer/LayoutAttrInfo.h"

#include "Logger/Logger.h"
#include "QueryRenderer/Interop/LayoutAttrInfo.h"

namespace QueryRenderer {

QueryDataType LayoutAttrInfo::getTypeFromLayoutAttr(const LayoutAttrInfo& attr_info) {
  CHECK(attr_info.buffer_layout);
  const auto& layout_attr_info =
      attr_info.buffer_layout->getAttributeInfo(attr_info.attr_name);
  return convertToQueryDataType(layout_attr_info.type);
}

}  // namespace QueryRenderer
