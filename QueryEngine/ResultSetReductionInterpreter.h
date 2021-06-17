/*
 * Copyright 2019 OmniSci, Inc.
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

  static EvalValue run(const Function* function, const std::vector<EvalValue>& inputs);

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
      const std::vector<std::unique_ptr<Instruction>>& body,
      const std::vector<EvalValue>& vars);
};
