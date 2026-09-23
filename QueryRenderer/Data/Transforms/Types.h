/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace QueryRenderer {

class BaseXform;
using XformShPtr = std::shared_ptr<BaseXform>;
using XformWkPtr = std::weak_ptr<BaseXform>;

class AggXform;
using AggXformShPtr = std::shared_ptr<AggXform>;

class XformOp;
using XformOpWkPtr = std::weak_ptr<XformOp>;
using XformOpShPtr = std::shared_ptr<XformOp>;

class AggOp;
using AggOpShPtr = std::shared_ptr<AggOp>;

}  // namespace QueryRenderer
