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
