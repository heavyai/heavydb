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
  TransientStringLiteralsVisitor(StringDictionaryProxy* sdp, Executor* executor)
      : sdp_(sdp), executor_(executor) {
    CHECK(sdp);
  }

  void* visitConstant(const Analyzer::Constant* constant) const override {
    if (constant->get_type_info().is_string() && !constant->get_is_null()) {
      CHECK(constant->get_constval().stringval);
      sdp_->getOrAddTransient(*constant->get_constval().stringval);
    }
    return defaultResult();
  }

  // visitUOper is for handling casts between dictionary encoded text
  // columns that do not share string dictionaries. For these
  // we need to run the translation again on the aggregator
  // so that we know how to interpret the transient literals added
  // by the leaves via string-to-string casts

  // Todo(todd): It is inefficient to do the same translation on
  // the aggregator and each of the leaves, explore storing these
  // translations/literals on the remote dictionary server instead
  // so the translation happens once and only once

  void* visitUOper(const Analyzer::UOper* uoper) const override {
    visit(uoper->get_operand());
    const auto& uoper_ti = uoper->get_type_info();
    const auto& operand_ti = uoper->get_operand()->get_type_info();
    if (!(uoper->get_optype() == kCAST && uoper_ti.is_dict_encoded_string() &&
          operand_ti.is_dict_encoded_string())) {
      // If we are not casting from a dictionary-encoded string
      // to a dictionary-encoded string
      return defaultResult();
    }
    if (uoper_ti.get_comp_param() != sdp_->getDictId()) {
      // If we are not casting to our dictionary (sdp_
      return defaultResult();
    }
    if (uoper_ti.get_comp_param() == operand_ti.get_comp_param()) {
      // If cast is inert, i.e. source and destination dict ids are same
      return defaultResult();
    }
    if (uoper_ti.is_dict_intersection()) {
      // Intersection translations don't add transients to the dest proxy,
      // and hence can be ignored for the purposes of populating transients
      return defaultResult();
    }
    executor_->getStringProxyTranslationMap(
        operand_ti.get_comp_param(),
        uoper_ti.get_comp_param(),
        RowSetMemoryOwner::StringTranslationType::SOURCE_UNION,
        executor_->getRowSetMemoryOwner(),
        true);  // with_generation
    return defaultResult();
  }

 protected:
  void* defaultResult() const override { return nullptr; }

 private:
  mutable StringDictionaryProxy* sdp_;
  mutable Executor* executor_;
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
