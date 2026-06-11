/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvidiaKernel.h"
#include "ResultSetReductionInterpreter.h"
#include "ResultSetReductionJIT.h"

// Generates wrappers of runtime functions with fixed signature which can be used from the
// interpreter.
class StubGenerator {
 public:
  // output_handle is ReductionInterpreter::EvalValue*, inputs_handle is a
  // std::vector<ReductionInterpreter::EvalValue>*.
  using Stub = ReductionInterpreter::EvalValue (*)(void* output_handle,
                                                   const void* inputs_handle);
  using InputsType = std::vector<ReductionInterpreter::EvalValue>;

  static Stub generateStub(const size_t executor_id,
                           const std::string& name,
                           const std::vector<Type>& arg_types,
                           const Type ret_type,
                           const bool is_external);
};

bool is_integer_type(const Type type);
bool is_pointer_type(const Type type);
