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

#include "Logger/Logger.h"
#include "QueryEngine/ScalarExprVisitor.h"

class TransientStringLiteralsVisitor : public ScalarExprVisitor<void*> {
 public:
  TransientStringLiteralsVisitor(StringDictionaryProxy* sdp) : sdp_(sdp) { CHECK(sdp); }

  void* visitConstant(const Analyzer::Constant* constant) const override {
    if (constant->get_type_info().is_string() && !constant->get_is_null()) {
      CHECK(constant->get_constval().stringval);
      sdp_->getOrAddTransient(*constant->get_constval().stringval);
    }
    return nullptr;
  }

 protected:
  void* defaultResult() const override { return nullptr; }

 private:
  mutable StringDictionaryProxy* sdp_;
};

class TransientDictIdVisitor : public ScalarExprVisitor<int> {
 public:
  int visitUOper(const Analyzer::UOper* uoper) const override {
    const auto& expr_ti = uoper->get_type_info();
    if (uoper->get_optype() == kCAST && expr_ti.is_string() &&
        expr_ti.get_compression() == kENCODING_DICT) {
      return expr_ti.get_comp_param();
    }
    return defaultResult();
  }

  int visitCaseExpr(const Analyzer::CaseExpr* case_expr) const override {
    const auto& expr_ti = case_expr->get_type_info();
    if (expr_ti.is_string() && expr_ti.get_compression() == kENCODING_DICT) {
      return expr_ti.get_comp_param();
    }
    return defaultResult();
  }

 protected:
  int defaultResult() const override { return -1; }
};
