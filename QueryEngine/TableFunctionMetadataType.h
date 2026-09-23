/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

enum class TableFunctionMetadataType {
  kUnknown,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kFloat,
  kDouble,
  kBool
};
