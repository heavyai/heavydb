/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/CoordAttrInfo2d.h"
#include "QueryRenderer/Marks/BaseRenderProperty.h"

namespace QueryRenderer {

OptionalStr CoordAttrInfo2d::constructOptFromPropCoord(
    const BaseRenderProperty& coord_prop) {
  auto const& coord_attr = coord_prop.getDataColumnNameRef();
  if (coord_attr.size()) {
    return coord_attr;
  }
  return std::nullopt;
}

OptionalStr CoordAttrInfo2d::getXAttr() const {
  if (x_prop) {
    return constructOptFromPropCoord(*x_prop);
  }
  return std::nullopt;
}

OptionalStr CoordAttrInfo2d::getYAttr() const {
  if (y_prop) {
    return constructOptFromPropCoord(*y_prop);
  }
  return std::nullopt;
}

}  // namespace QueryRenderer
