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

class StartEndPoint : public Codegen {
 public:
  StartEndPoint(const Analyzer::GeoOperator* geo_operator) : Codegen(geo_operator) {
    CHECK_EQ(operator_->size(), size_t(1));
    // nulls not supported yet
    this->is_nullable_ = false;
  }

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kNULLT); }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(pos_lvs.size(), size());
    // TODO: add null handling
    if (arg_lvs.size() == size_t(1)) {
      // col byte stream from column on disk
      auto operand = getOperand(0);
      CHECK(operand);
      const auto& geo_ti = operand->get_type_info();
      CHECK(geo_ti.get_type() == kLINESTRING);

      std::vector<llvm::Value*> array_operand_lvs;
      array_operand_lvs.push_back(
          cgen_state->emitExternalCall("array_buff",
                                       llvm::Type::getInt8PtrTy(cgen_state->context_),
                                       {arg_lvs.front(), pos_lvs.front()}));
      const bool is_nullable = !geo_ti.get_notnull();
      std::string size_fn_name = "array_size";
      if (is_nullable) {
        size_fn_name += "_nullable";
      }
      uint32_t elem_sz = 1;  // TINYINT coords array
      std::vector<llvm::Value*> array_sz_args{
          arg_lvs.front(), pos_lvs.front(), cgen_state->llInt(log2_bytes(elem_sz))};
      if (is_nullable) {
        array_sz_args.push_back(
            cgen_state->llInt(static_cast<int32_t>(inline_int_null_value<int32_t>())));
      }
      array_operand_lvs.push_back(cgen_state->emitExternalCall(
          size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args));
      return std::make_tuple(array_operand_lvs, nullptr);
    }
    CHECK_EQ(arg_lvs.size(), size_t(2));  // ptr, size
    return std::make_tuple(arg_lvs, nullptr);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    UNREACHABLE();
    return {nullptr, nullptr};
  }
};

}  // namespace spatial_type
