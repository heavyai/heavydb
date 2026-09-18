/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

namespace QueryRenderer {

class ProjectionShader {
 public:
  ProjectionShader(const std::string& name) : name_(name) {}
  ~ProjectionShader() {}

  void addDefine(const std::string& name, const std::string& value);
  void addUniform(const std::string& type, const std::string& name);
  void addFunctionArgument(const std::string& type, const std::string& name);
  void setFunctionBody(const std::string& body);

  std::string getFunctionDeclaration(const std::string& func_name,
                                     const std::string& return_type) const;
  std::string getFunctionBody() const;
  std::string getDecl() const;

  operator std::string() const;

 protected:
  std::string name_;

  std::vector<std::string> defines_;
  std::vector<std::string> uniforms_;
  std::vector<std::string> func_args_;
  std::string body_;
};

}  // namespace QueryRenderer
