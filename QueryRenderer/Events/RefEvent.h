/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <ostream>
#include <string>

#include "QueryRenderer/JSONRefObject.h"

namespace QueryRenderer {

enum class RefEventType { kUpdate, kRemove, kReplace, kAll };

std::string to_string(const RefEventType ref_event_type);

using RefEventCallback = std::function<void(RefEventType, const RefObjShPtr&)>;

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os,
                         const ::QueryRenderer::RefEventType ref_event_type);
