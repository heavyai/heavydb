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

#include "CodeGenerator.h"
#include "Execute.h"

std::vector<llvm::Value*> CodeGenerator::codegenGeoExpr(Analyzer::GeoExpr const* geo_expr,
                                                        CompilationOptions const& co) {
  using ValueVector = std::vector<llvm::Value*>;
  ValueVector argument_list;

  for (const auto& arg : geo_expr->getArgs()) {
    const auto arg_lvs = codegen(arg.get(), true, co);
    argument_list.insert(argument_list.end(), arg_lvs.begin(), arg_lvs.end());
    const auto arg_ti = arg->get_type_info();
    if (arg_ti.is_array() && arg_lvs.size() == 1) {
      // Array size is sometimes discarded during codegen, e.g. for cast ArrayExpr
      // which needs to be addressed (TODO)
      // Append constant array size if codegen hasn't supplied it
      const auto arr_size = arg_ti.get_size();
      if (arr_size > 0) {
        argument_list.push_back(cgen_state_->llInt(int32_t(arr_size)));
      } else {
        throw std::runtime_error("Unable to handle Geo construction.");
      }
    }
  }
  return argument_list;
}
