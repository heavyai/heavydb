/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @file    ExtensionFunctionsWhitelist.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Supported runtime functions management and retrieval.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_EXTENSIONFUNCTIONSWHITELIST_H
#define QUERYENGINE_EXTENSIONFUNCTIONSWHITELIST_H

#include <string>
#include <unordered_map>
#include <vector>

enum class ExtArgumentType { Int16, Int32, Int64, Float, Double, PInt16, PInt32, PInt64, PFloat, PDouble };

class ExtensionFunction {
 public:
  ExtensionFunction(const std::string& name, const std::vector<ExtArgumentType>& args, const ExtArgumentType ret)
      : name_(name), args_(args), ret_(ret) {}

  const std::string& getName() const { return name_; }

  const std::vector<ExtArgumentType>& getArgs() const { return args_; }

  const ExtArgumentType getRet() const { return ret_; }

 private:
  const std::string name_;
  const std::vector<ExtArgumentType> args_;
  const ExtArgumentType ret_;
};

class ExtensionFunctionsWhitelist {
 public:
  static void add(const std::string& json_func_sigs);

  static std::vector<ExtensionFunction>* get(const std::string& name);

  static std::vector<std::string> getLLVMDeclarations();

 private:
  // Function overloading not supported, they're uniquely identified by name.
  static std::unordered_map<std::string, std::vector<ExtensionFunction>> functions_;
};

#endif  // QUERYENGINE_EXTENSIONFUNCTIONSWHITELIST_H
