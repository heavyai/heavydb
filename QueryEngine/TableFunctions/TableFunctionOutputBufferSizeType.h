/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    RelAlgExecutionUnit.h
 * @brief   Execution unit for relational algebra. It's a low-level description
 *          of any relational algebra operation in a format understood by our VM.
 *
 */

#pragma once

namespace table_functions {

enum class OutputBufferSizeType {
  kConstant,
  kUserSpecifiedConstantParameter,
  kUserSpecifiedRowMultiplier,
  kTableFunctionSpecifiedParameter,
  kPreFlightParameter
};

}  // namespace table_functions
