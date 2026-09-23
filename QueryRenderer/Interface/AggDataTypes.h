/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace QueryRenderer {

class AnyDataType;

using AnyDataTypeShPtr = std::shared_ptr<AnyDataType>;
using AggDataList = std::vector<AnyDataTypeShPtr>;

using AggDataMap = std::unordered_map<
    std::string,
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::unordered_map<std::string, AggDataList>>>>;

}  // namespace QueryRenderer
