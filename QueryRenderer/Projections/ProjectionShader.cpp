/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Projections/ProjectionShader.h"

#include <boost/algorithm/string/join.hpp>

namespace QueryRenderer {

void ProjectionShader::addDefine(const std::string& name, const std::string& value) {
  defines_.push_back("#define " + name + " " + value);
}

void ProjectionShader::addUniform(const std::string& type, const std::string& name) {
  uniforms_.push_back(type + " " + name + ";");
}

void ProjectionShader::addFunctionArgument(const std::string& type,
                                           const std::string& name) {
  func_args_.push_back("in " + type + " " + name);
}

void ProjectionShader::setFunctionBody(const std::string& body) {
  body_ = body;
}

std::string ProjectionShader::getFunctionDeclaration(
    const std::string& func_name,
    const std::string& return_type) const {
  std::string ret;

  ret += return_type;
  ret += " ";
  ret += func_name;
  ret += "(";
  ret += boost::algorithm::join(func_args_, " ");

  ret += ")";

  return ret;
}

std::string ProjectionShader::getFunctionBody() const {
  return body_;
}

std::string ProjectionShader::getDecl() const {
  std::string ret;

  for (auto& define_str : defines_) {
    ret += define_str + "\n";
  }
  ret += "\n";

  // TODO(scb): Builder should handle UBO and tag
  if (uniforms_.size()) {
    ret += "layout(std430) uniform PROJECTION_UBO_TYPE_" + name_ + " {\n";
    for (auto& uniform_str : uniforms_) {
      ret += "  " + uniform_str + "\n";
    }
    ret += "};\n";
  }

  return ret;
}

ProjectionShader::operator std::string() const {
  return "ProjectionShader(name: " + name_ + ")";
}

}  // namespace QueryRenderer
