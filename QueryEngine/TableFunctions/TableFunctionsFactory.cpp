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

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"

#include <mutex>

void TableFunctionsFactory::add(const std::string& name,
                                const std::vector<ExtArgumentType>& args) {
  functions_.insert(std::make_pair(name, TableFunction(name, args, {})));
}

std::once_flag init_flag;

void TableFunctionsFactory::init() {
  std::call_once(init_flag, []() {
    TableFunctionsFactory::add("row_copier",
                               std::vector<ExtArgumentType>{ExtArgumentType::PDouble,
                                                            ExtArgumentType::PInt32,
                                                            ExtArgumentType::PInt64,
                                                            ExtArgumentType::PInt64,
                                                            ExtArgumentType::PInt64});
  });
}

std::unordered_map<std::string, TableFunction> TableFunctionsFactory::functions_;
