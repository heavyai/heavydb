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

enum class ExtArgumentType { Int16, Int32, Int64, Float, Double };

class ExtensionFunction {
 public:
  ExtensionFunction(const std::string& name, const std::vector<ExtArgumentType>& args, const ExtArgumentType ret)
      : name_(name), args_(args), ret_(ret) {}

 private:
  const std::string name_;
  const std::vector<ExtArgumentType> args_;
  const ExtArgumentType ret_;
};

class ExtensionFunctionsWhitelist {
 public:
  void add(const std::string& json_func_sigs);

  ExtensionFunction* get(const std::string& name);

 private:
  // Function overloading not supported, they're uniquely identified by name.
  static std::unordered_map<std::string, ExtensionFunction> functions_;
};

#endif  // QUERYENGINE_EXTENSIONFUNCTIONSWHITELIST_H
