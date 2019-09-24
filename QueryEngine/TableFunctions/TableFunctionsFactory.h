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

#include <string>
#include <vector>

#include "QueryEngine/ExtensionFunctionsWhitelist.h"

class TableFunction {
 public:
  TableFunction(const std::string& name,
                const std::vector<ExtArgumentType>& input_args,
                const std::vector<ExtArgumentType>& return_args)
      : name_(name), input_args_(input_args), return_args_(return_args) {}

  std::vector<ExtArgumentType> getArgs() const { return input_args_; }

 private:
  const std::string name_;
  const std::vector<ExtArgumentType> input_args_;
  const std::vector<ExtArgumentType> return_args_;
};

class TableFunctionsFactory {
 public:
  static void add(const std::string& name, const std::vector<ExtArgumentType>& args);

 private:
  static void init();

  static std::unordered_map<std::string, TableFunction> functions_;

  friend class ExtensionFunctionsWhitelist;
};
