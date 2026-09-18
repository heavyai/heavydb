/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <set>
#include <string>

#include <glslang/Public/ShaderLang.h>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

class GlslangIncluder : public glslang::TShader::Includer {
 public:
  explicit GlslangIncluder(const Library& library);
  ~GlslangIncluder() override;

  // TODO(scb): any use for system vs local (currently both are treated the same)?
  IncludeResult* includeSystem(const char* header_name,
                               const char* includer_name,
                               size_t inclusion_depth) override;
  IncludeResult* includeLocal(const char* header_name,
                              const char* includer_name,
                              size_t inclusion_depth) override;

  void releaseInclude(IncludeResult*) override;

  GlslangIncluder() = delete;

 private:
  using IncludeResultUqPtr = std::unique_ptr<IncludeResult>;
  using IncludeResultSet = std::set<IncludeResultUqPtr>;

  IncludeResultSet result_set_;
  const Library& library_;
};

class GlslangWrapper {
 public:
  using CompileResult = std::pair<spirv_t, std::string>;

  explicit GlslangWrapper(const Library& library);
  ~GlslangWrapper();

  CompileResult glslToSpirv(const std::string& pretty_name,
                            const std::string& source,
                            const std::string& entry_point,
                            const ShaderStage shader_stage,
                            const SubroutineMap& func_rebind_map);
  GlslangWrapper() = delete;
  GlslangWrapper(const GlslangWrapper&) = delete;
  GlslangWrapper& operator=(const GlslangWrapper&) = delete;

 private:
  GlslangIncluder includer_;
  TBuiltInResource resources_;

  using TShaderUqPtr = std::unique_ptr<glslang::TShader>;
  using TProgramUqPtr = std::unique_ptr<glslang::TProgram>;
};

}  // namespace gfx
