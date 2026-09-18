/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

struct CoordAttrInfo2d {
  const BaseRenderProperty* const x_prop;
  const BaseRenderProperty* const y_prop;

  CoordAttrInfo2d(const BaseRenderProperty* in_x_prop,
                  const BaseRenderProperty* in_y_prop)
      : x_prop{in_x_prop}, y_prop{in_y_prop} {}

  static OptionalStr constructOptFromPropCoord(const BaseRenderProperty& coord_prop);

  OptionalStr getXAttr() const;
  OptionalStr getYAttr() const;
};

}  // namespace QueryRenderer
