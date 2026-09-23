/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace QueryRenderer {

class BaseMark;
using BaseMarkUqPtr = std::unique_ptr<BaseMark>;

class BaseRenderProperty;
using BaseRenderPropertyUqPtr = std::unique_ptr<BaseRenderProperty>;

}  // namespace QueryRenderer
