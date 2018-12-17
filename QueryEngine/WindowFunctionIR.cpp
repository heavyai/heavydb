/*
 * Copyright 2018 OmniSci, Inc.
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

#include "Execute.h"
#include "WindowContext.h"

llvm::Value* Executor::codegenWindowFunction(const Analyzer::WindowFunction* window_func,
                                             const size_t target_index,
                                             const CompilationOptions& co) {
  const auto window_func_context =
      WindowProjectNodeContext::get()->activateWindowFunctionContext(target_index);
  switch (window_func->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER:
    case SqlWindowFunctionKind::RANK:
    case SqlWindowFunctionKind::DENSE_RANK:
    case SqlWindowFunctionKind::NTILE: {
      return cgen_state_->emitCall(
          "row_number_window_func",
          {ll_int(reinterpret_cast<const int64_t>(window_func_context->output())),
           posArg(nullptr)});
    }
    case SqlWindowFunctionKind::PERCENT_RANK:
    case SqlWindowFunctionKind::CUME_DIST: {
      return cgen_state_->emitCall(
          "percent_window_func",
          {ll_int(reinterpret_cast<const int64_t>(window_func_context->output())),
           posArg(nullptr)});
    }
    case SqlWindowFunctionKind::LAG:
    case SqlWindowFunctionKind::LEAD:
    case SqlWindowFunctionKind::FIRST_VALUE:
    case SqlWindowFunctionKind::LAST_VALUE: {
      CHECK(WindowProjectNodeContext::get());
      const auto& args = window_func->getArgs();
      CHECK(!args.empty());
      const auto lag_lvs = codegen(args.front().get(), true, co);
      CHECK_EQ(lag_lvs.size(), size_t(1));
      return lag_lvs.front();
    }
    default: { LOG(FATAL) << "Invalid window function kind"; }
  }
  return nullptr;
}
