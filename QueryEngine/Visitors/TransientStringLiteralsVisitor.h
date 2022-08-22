/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "StringOps/StringOps.h"

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
    const auto& uoper_ti = uoper->get_type_info();
    const auto& operand_ti = uoper->get_operand()->get_type_info();
    if (!(uoper->get_optype() == kCAST && uoper_ti.is_dict_encoded_string())) {
      return defaultResult();
    }
    const bool outputs_target_sdp = uoper_ti.get_comp_param() == sdp_->getDictId();

    if (!parent_feeds_sdp_ && !outputs_target_sdp) {
      // If we are not casting to our dictionary (sdp_)
      return defaultResult();
    }
    if (uoper_ti.is_dict_intersection()) {
      // Intersection translations don't add transients to the dest proxy,
      // and hence can be ignored for the purposes of populating transients
      return defaultResult();
    }
    const bool parent_feeds_sdp_already_set = parent_feeds_sdp_;
    parent_feeds_sdp_ = true;

    visit(uoper->get_operand());

    if (!parent_feeds_sdp_already_set) {
      parent_feeds_sdp_ = false;
    }

    if (operand_ti.is_dict_encoded_string() &&
        uoper_ti.get_comp_param() != operand_ti.get_comp_param()) {
      executor_->getStringProxyTranslationMap(
          operand_ti.get_comp_param(),
          uoper_ti.get_comp_param(),
          RowSetMemoryOwner::StringTranslationType::SOURCE_UNION,
          {},
          executor_->getRowSetMemoryOwner(),
          true);  // with_generation
    }
    return defaultResult();
  }

  void* visitStringOper(const Analyzer::StringOper* string_oper) const override {
    CHECK_GE(string_oper->getArity(), 1UL);
    const auto str_operand = string_oper->getArg(0);
    const auto& string_oper_ti = string_oper->get_type_info();
    const auto& str_operand_ti = str_operand->get_type_info();
    const auto string_oper_kind = string_oper->get_kind();
    if (!string_oper_ti.is_string() || !str_operand_ti.is_string()) {
      return defaultResult();
    }
    if (string_oper->getNonLiteralsArity() >= 2UL) {
      return defaultResult();
    }
    const bool parent_feeds_sdp_already_set = parent_feeds_sdp_;
    const bool outputs_target_sdp = string_oper_ti.get_comp_param() == sdp_->getDictId();
    if (string_oper_ti.is_dict_encoded_string() &&
        str_operand_ti.is_dict_encoded_string() &&
        (parent_feeds_sdp_ || outputs_target_sdp)) {
      parent_feeds_sdp_ = true;
      visit(str_operand);
      if (!parent_feeds_sdp_already_set) {
        parent_feeds_sdp_ = false;
      }
      // Todo(todd): Dedup the code to get string_op_infos from the same
      // in StringOpsIR.cpp (needs thought as Analyzer and StringOps
      // deliberately are oblivious to each other)

      std::vector<StringOps_Namespace::StringOpInfo> string_op_infos;
      const auto chained_string_op_exprs = string_oper->getChainedStringOpExprs();
      for (const auto& chained_string_op_expr : chained_string_op_exprs) {
        auto chained_string_op =
            dynamic_cast<const Analyzer::StringOper*>(chained_string_op_expr.get());
        CHECK(chained_string_op);
        StringOps_Namespace::StringOpInfo string_op_info(
            chained_string_op->get_kind(),
            chained_string_op->get_type_info(),
            chained_string_op->getLiteralArgs());
        string_op_infos.emplace_back(string_op_info);
      }

      executor_->getStringProxyTranslationMap(
          str_operand_ti.get_comp_param(),
          string_oper_ti.get_comp_param(),
          RowSetMemoryOwner::StringTranslationType::SOURCE_UNION,
          string_op_infos,
          executor_->getRowSetMemoryOwner(),
          true);  // with_generation
    } else if ((parent_feeds_sdp_ || outputs_target_sdp) &&
               (string_oper->getLiteralsArity() == string_oper->getArity())) {
      // This is likely dead code due to ExpressionRewrite of all-literal string ops
      // (meaning when this visitor gets to a string op with all literal args it
      // (would have already been rewritten as a literal string)
      // Todo(todd): Verify and remove if so
      const StringOps_Namespace::StringOpInfo string_op_info(
          string_oper_kind, string_oper->get_type_info(), string_oper->getLiteralArgs());
      CHECK_EQ(string_op_info.numLiterals(), string_oper->getArity());
      const auto str_result_and_null_status =
          StringOps_Namespace::apply_string_op_to_literals(string_op_info);
      if (string_oper->get_type_info().is_string() &&
          !str_result_and_null_status.second &&
          !str_result_and_null_status.first
               .empty()) {  // Todo(todd): Is there a central/non-magic function/constant
                            // to determine if a none-encoded string is null
        sdp_->getOrAddTransient(str_result_and_null_status.first);
      }
    }
    return defaultResult();
  }

 protected:
  void* defaultResult() const override { return nullptr; }

 private:
  mutable StringDictionaryProxy* sdp_;
  mutable Executor* executor_;
  mutable bool parent_feeds_sdp_{false};
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

  int visitStringOper(const Analyzer::StringOper* string_oper) const override {
    const auto& expr_ti = string_oper->get_type_info();
    if (expr_ti.is_string() && expr_ti.get_compression() == kENCODING_DICT) {
      return expr_ti.get_comp_param();
    }
    return defaultResult();
  }

 protected:
  int defaultResult() const override { return -1; }
};
