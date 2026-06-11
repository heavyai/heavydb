/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultSetReductionOps.h"

#include <optional>
#include <unordered_map>

class ReductionInterpreter {
 public:
  union EvalValue {
    int64_t int_val;
    double double_val;
    float float_val;
    const void* ptr;
    void* mutable_ptr;
  };

  static EvalValue run(const size_t execution_id,
                       const Function* function,
                       const std::vector<EvalValue>& inputs);

  template <typename T>
  static EvalValue MakeEvalValue(const T& val) {
    EvalValue ret;
    if constexpr (std::is_integral<T>::value) {
      ret.int_val = static_cast<int64_t>(val);
    } else if constexpr (std::is_same<T, float>::value) {
      ret.float_val = val;
    } else if constexpr (std::is_same<T, double>::value) {
      ret.double_val = val;
    } else if constexpr (std::is_pointer<T>::value) {
      ret.ptr = val;
    }
    return ret;
  }

  static std::optional<EvalValue> run(
      const size_t execution_id,
      const std::vector<std::unique_ptr<Instruction>>& body,
      const std::vector<EvalValue>& vars);
};
