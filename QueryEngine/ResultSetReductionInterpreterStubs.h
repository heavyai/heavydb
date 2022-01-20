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
