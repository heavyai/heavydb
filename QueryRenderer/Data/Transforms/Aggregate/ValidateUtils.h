/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"

namespace QueryRenderer {

struct ValidateUtils {
  static void validateNumOrDictEncodedStrInput(const BaseDataTableShPtr& in_data,
                                               const LayoutAttrInfoSet& inputs,
                                               const AggOp* curr_op);
};

}  // namespace QueryRenderer
