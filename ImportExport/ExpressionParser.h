/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file ExpressionParser.h
 * @brief General Expression Parser using muparserx
 *
 */

#pragma once

#include <memory>
#include <string>

namespace mup {
class ParserX;
}

namespace import_export {

class ExpressionParser {
 public:
  ExpressionParser();

  void setExpression(const std::string& expression);

  void setStringConstant(const std::string& name, const std::string& value);
  void setIntConstant(const std::string& name, const int value);

  std::string evalAsString();
  int evalAsInt();
  double evalAsDouble();
  bool evalAsBool();

 private:
  struct ParserDeleter {
    void operator()(mup::ParserX* parser);
  };
  std::unique_ptr<mup::ParserX, ParserDeleter> parser_;
};

}  // namespace import_export
