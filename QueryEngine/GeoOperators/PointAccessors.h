/*
 * Copyright 2021 OmniSci, Inc.
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

#include "QueryEngine/GeoOperators/Codegen.h"

namespace spatial_type {

// ST_X and ST_Y
class PointAccessors : public Codegen {
 public:
  PointAccessors(const Analyzer::GeoOperator* geo_operator) : Codegen(geo_operator) {
    CHECK_EQ(operator_->size(), size_t(1));
  }

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kBOOLEAN); }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    UNREACHABLE();
    llvm::Value* array_buff_ptr{nullptr};
    llvm::Value* is_null{nullptr};
    return std::make_tuple(std::vector<llvm::Value*>{array_buff_ptr}, is_null);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    UNREACHABLE();
    return {nullptr};
  }
};

}  // namespace spatial_type
