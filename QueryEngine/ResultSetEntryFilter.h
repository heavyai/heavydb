/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Copyright 2026 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "Shared/sqldefs.h"
#include "Shared/sqltypes.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ResultSetEntryLiteral {
  SQLTypeInfo type_info;
  bool is_null{false};
  int64_t int_val{0};
  double double_val{0.0};
  bool bool_val{false};
};

struct ResultSetEntryComparison {
  size_t target_idx{0};
  SQLOps op{kINVALID_OP};
  ResultSetEntryLiteral literal;
};

using ResultSetEntryFilter = std::vector<ResultSetEntryComparison>;
