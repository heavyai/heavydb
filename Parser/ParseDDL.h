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
#pragma once

#include <exception>
#include <list>
#include <memory>
#include <string>

#include "Parser/ParserNode.h"
#include "Parser/parser.h"

namespace Parser {

template <typename DDLSTMT>
auto parseDDL(const std::string& stmt_type, const std::string& stmt_str) {
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  SQLParser parser;
  if (parser.parse(stmt_str, parse_trees, last_parsed) > 0) {
    throw std::runtime_error("Syntax error in " + stmt_type + " \"" + stmt_str +
                             "\" at " + last_parsed);
  }
  if (auto ddl_stmt = dynamic_cast<DDLSTMT*>(parse_trees.front().get())) {
    parse_trees.front().release();
    return std::unique_ptr<DDLSTMT>(ddl_stmt);
  }
  throw std::runtime_error("Expected " + stmt_type + " is not found in \"" + stmt_str +
                           "\"");
}

}  // namespace Parser
