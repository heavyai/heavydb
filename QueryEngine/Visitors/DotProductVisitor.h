/*
 * Copyright 2024 HEAVY.AI, Inc.
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

/**
 * @file   DotProductVisitor.h
 * @brief  Manage dot product expressions by homogenizing array types, casting decimal
 *         constants, and converting Analyzer::ArrayExpr to Analyzer::Constant to allow
 *         hoisting of literal arrays.
 */

#pragma once

#include "QueryEngine/DeepCopyVisitor.h"

class DotProductVisitor final : public DeepCopyVisitor {
 protected:
  std::shared_ptr<Analyzer::Expr> visitDotProduct(
      Analyzer::DotProductExpr const* dot_product_expr) const override {
    SQLTypeInfo const& return_type = dot_product_expr->get_type_info();
    auto array1 = homogenizeArrayExpr(return_type, dot_product_expr->get_own_arg1());
    auto array2 = homogenizeArrayExpr(return_type, dot_product_expr->get_own_arg2());
    return makeExpr<Analyzer::DotProductExpr>(array1, array2);
  }

 private:
  /// Conditionally add a cast to constant decimal expressions because it is unwieldy
  /// to carry their precision and scale into the dot_product calculation.
  void castConstantFromDecimal(SQLTypeInfo const& return_type,
                               Analyzer::Expr* expr) const {
    if (auto* constant = dynamic_cast<Analyzer::Constant*>(expr)) {
      if (constant->get_type_info().is_decimal() &&
          constant->get_type_info() != return_type) {
        constant->add_cast(return_type);
      }
    }
  }

  /// When an Array is encountered under a DOT_PRODUCT(), if its elements are decimal,
  /// cast them to return_type_ by castConstantFromDecimal(). If the array consists only
  /// of constants, then return it as a Constant so that it can be hoisted with
  /// CgenState::getOrAddLiteral().
  std::shared_ptr<Analyzer::Expr> homogenizeArrayExpr(
      SQLTypeInfo const& return_type,
      std::shared_ptr<Analyzer::Expr> expr) const {
    if (auto* array_expr = dynamic_cast<Analyzer::ArrayExpr const*>(expr.get())) {
      std::vector<std::shared_ptr<Analyzer::Expr>> elems;
      elems.reserve(array_expr->getElementCount());
      for (size_t i = 0; i < array_expr->getElementCount(); ++i) {
        elems.push_back(visit(array_expr->getElement(i)));
        castConstantFromDecimal(return_type, elems.back().get());
      }
      // The SQLTypeInfo type is kARRAY and the subtype is the elements' type.
      // The other SQLTypeInfo params are copied from the elements' type.
      SQLTypeInfo type_info =
          elems.empty() ? return_type : elems.front()->get_type_info();
      if (!elems.empty()) {
        type_info.set_subtype(type_info.get_type());
        type_info.set_type(kARRAY);
      }
      if (std::all_of(elems.begin(), elems.end(), isConstant)) {
        constexpr bool is_null = false;
        return makeExpr<Analyzer::Constant>(
            type_info,
            is_null,
            std::list<std::shared_ptr<Analyzer::Expr>>{elems.begin(), elems.end()});
      } else {
        return makeExpr<Analyzer::ArrayExpr>(type_info,
                                             std::move(elems),
                                             array_expr->isNull(),
                                             array_expr->isLocalAlloc());
      }
    }
    return visit(expr.get());
  }

  static bool isConstant(std::shared_ptr<Analyzer::Expr> const& expr) {
    return static_cast<bool>(dynamic_cast<Analyzer::Constant*>(expr.get()));
  }
};
