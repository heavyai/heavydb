/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/GlslangWrapper.h"

#include <iomanip>
#include <sstream>

#include <glslang/MachineIndependent/iomapper.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/ShaderCompiler/Library.h"
#include "GfxDriver/ShaderCompiler/ResourceLimits.h"
#include "GfxDriver/ShaderCompiler/ShaderRedecorator.h"
#include "GfxDriver/ShaderCompiler/TShaderIRUtils.h"

namespace gfx {

namespace {
static EShLanguage shader_stage_to_glslang_enum(ShaderStage eStage) {
  switch (eStage) {
    case ShaderStage::kVertex:
      return EShLangVertex;
    case ShaderStage::kFragment:
      return EShLangFragment;
    case ShaderStage::kGeometry:
      return EShLangGeometry;
    case ShaderStage::kTessControl:
      return EShLangTessControl;
    case ShaderStage::kTessEval:
      return EShLangTessEvaluation;
    case ShaderStage::kCompute:
      return EShLangCompute;
    case ShaderStage::kRayGen:
      return EShLangRayGen;
    case ShaderStage::kAnyHit:
      return EShLangAnyHit;
    case ShaderStage::kClosestHit:
      return EShLangClosestHit;
    case ShaderStage::kMiss:
      return EShLangMiss;
    case ShaderStage::kIntersection:
      return EShLangIntersect;
    case ShaderStage::kCallable:
      return EShLangCallable;
    case ShaderStage::kMesh:
      return EShLangMesh;
    case ShaderStage::kTask:
      return EShLangTask;
  }
  THROW_RUNTIME_EX("Unknown shader stage");
  return EShLangVertex;
}

}  // namespace

//
// GlslangIncluder
//

GlslangIncluder::GlslangIncluder(const Library& library) : library_(library) {}

GlslangIncluder::~GlslangIncluder() {
  result_set_.clear();
}

glslang::TShader::Includer::IncludeResult* GlslangIncluder::includeSystem(
    const char* header_name,
    const char* includer_name,
    size_t inclusion_depth) {
  const auto& source = library_.get(header_name).code;

  if (source.empty()) {
    LOG(ERROR) << "Failed to find shader \'" << header_name << "\' included from \'"
               << includer_name << "\'";
    return nullptr;
  }

  // TODO: store lengths
  return result_set_
      .insert(std::make_unique<IncludeResult>(
          header_name, source.c_str(), source.size(), nullptr))
      .first->get();
}

glslang::TShader::Includer::IncludeResult* GlslangIncluder::includeLocal(
    const char* header_name,
    const char* includer_name,
    size_t inclusion_depth) {
  // Support system includes only for now. The glslang parser will automatically call
  // IncludeSystem if this method returns null
  return nullptr;
}

void GlslangIncluder::releaseInclude(glslang::TShader::Includer::IncludeResult* result) {
  if (result != nullptr) {
    for (auto&& r : result_set_) {
      if (r.get() == result) {
        result_set_.erase(r);
        return;
      }
    }
  }
}

//
// IoMapResolver
//

class IoMapResolver : public glslang::TIoMapResolver {
 public:
  ~IoMapResolver() override = default;

  bool validateBinding(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return true;
  }

  int resolveBinding(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    if (!ent.symbol->getType().getQualifier().hasBinding()) {
      return ent.newBinding = ShaderRedecorator::kUninitializedBinding;
    }
    return -1;
  }

  int resolveSet(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    if (!ent.symbol->getType().getQualifier().hasSet()) {
      return ent.newSet = ShaderRedecorator::kUninitializedSet;
    }
    return -1;
  }

  int resolveUniformLocation(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    // we have no regular (non-opaque) uniforms
    // opaque uniforms (samplers, images) have binding, not location
    return -1;
  }

  bool validateInOut(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return true;
  }

  int resolveInOutLocation(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    // assign only non-built-in vertex shader pipeline inputs
    auto const& type = ent.symbol->getType();
    if (!type.getQualifier().hasLocation()) {
      auto const& name = ent.symbol->getName();
      if (stage == EShLangVertex && type.getQualifier().isPipeInput() && !name.empty() &&
          std::string(name).substr(0, 3) != "gl_") {
        return ent.newLocation = ShaderRedecorator::kUninitializedLocation;
      }
    }
    return ent.newLocation = -1;
  }

  int resolveInOutComponent(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    // do not assign
    return -1;
  }

  int resolveInOutIndex(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    // do not assign
    return -1;
  }

  // all the rest, do nothing
  void notifyBinding(EShLanguage stage, glslang::TVarEntryInfo& ent) override {}
  void notifyInOut(EShLanguage stage, glslang::TVarEntryInfo& ent) override {}
  void endNotifications(EShLanguage stage) override {}
  void beginNotifications(EShLanguage stage) override {}
  void beginResolve(EShLanguage stage) override {}
  void endResolve(EShLanguage stage) override {}

  // Added with update to 7.12.3352
  // These facilitate auto-mapping of sets / bindings / locations across multiple stages

  // Called by mapIO when it starts its symbol collect for teh given stage
  void beginCollect(EShLanguage stage) override {}
  // Called by mapIO when it has finished the symbol collect
  void endCollect(EShLanguage stage) override {}
  // Called by TSlotCollector to resolve storage locations or bindings
  void reserverStorageSlot(glslang::TVarEntryInfo& ent, TInfoSink& infoSink) override {}
  // Called by TSlotCollector to resolve resource locations or bindings
  void reserverResourceSlot(glslang::TVarEntryInfo& ent, TInfoSink& infoSink) override {}
  // Called by mapIO.addStage to set shader stage mask to mark a stage be added to this
  // pipeline
  void addStage(EShLanguage stage, glslang::TIntermediate& stageIntermediate) override {}
};

//
// GlslangWrapper
//

GlslangWrapper::GlslangWrapper(const Library& library)
    : includer_(library)
    // TODO (scb): Get limits from devices and reconcile deltas in the parent driver
    , resources_(DefaultTBuiltInResource) {
  glslang::InitializeProcess();
  resources_ = DefaultTBuiltInResource;
}

GlslangWrapper::~GlslangWrapper() {
  glslang::FinalizeProcess();
}

#ifndef NDEBUG
namespace {
int find_error_line_number(const std::string& info_log) {
  // Extract line number from a message formatted:
  // ERROR: 0:129: '' :  syntax error...
  // Where 129 is the line number
  auto start = info_log.find(":", 7);  // skip "ERROR: 0" where 0 is string #
  CHECK_NE(start, std::string::npos);
  auto end = info_log.find(":", start + 1);  // find ":" after line #
  CHECK_NE(end, std::string::npos);
  auto substr = info_log.substr(start + 1, end - start - 1);
  return std::atoi(substr.c_str());
}

std::string add_line_numbers(const std::string& code, int error_line_num) {
  std::stringstream ss;
  std::istringstream input;
  input.str(code);
  int line_num = 1;
  for (std::string line; std::getline(input, line); ++line_num) {
    ss << (line_num == error_line_num ? "-->" : "   ");
    ss << std::right << std::setw(5) << line_num << "  " << line << "\n";
  }
  return ss.str();
}
}  // namespace
#endif

GlslangWrapper::CompileResult GlslangWrapper::glslToSpirv(
    const std::string& pretty_name,
    const std::string& shader_source,
    const std::string& entry_point,
    const ShaderStage in_shader_stage,
    const SubroutineMap& func_rebind_map) {
  // These may become function parameters in the future so using constexpr
  // instead of #defines
  constexpr bool dump_AST = false;
  constexpr bool dump_spv_build_log = false;

  auto glslang_shader_stage = shader_stage_to_glslang_enum(in_shader_stage);
  auto shader = std::make_unique<glslang::TShader>(glslang_shader_stage);

  auto const* src_c_str = shader_source.c_str();
  auto str_len = static_cast<int>(shader_source.size());
  spirv_t rtn_spirv;
  std::string error_string;

  shader->setStringsWithLengths(&src_c_str, &str_len, 1);
  if (!entry_point.empty()) {
    // These both need to be set if source language is GLSL (HLSL is different)
    shader->setEntryPoint(entry_point.c_str());
    shader->setSourceEntryPoint(entry_point.c_str());
  }

  EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
  if (dump_AST) {
    messages = (EShMessages)(messages | EShMsgAST);
  }

  shader->setAutoMapBindings(true);
  shader->setAutoMapLocations(true);
  shader->setEnvInput(
      glslang::EShSourceGlsl, glslang_shader_stage, glslang::EShClientVulkan, 100);
  shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
  shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_4);

  if (!shader->parse(&resources_, 100, ECoreProfile, false, false, messages, includer_)) {
    // Just log the error but don't throw yet. This allows ShaderManager to save
    // artifacts and handle the error
    std::string info_log(shader->getInfoLog());
    error_string = "Error parsing shader \"" + pretty_name + "\":\n" + info_log;
#ifndef NDEBUG
    auto error_line_num = find_error_line_number(info_log);
    error_string += "Shader Source:\n" + add_line_numbers(shader_source, error_line_num) +
                    "End Shader Source\n";
#endif
    return {std::move(rtn_spirv), std::move(error_string)};
  }

  if (!func_rebind_map.empty()) {
    rebind_tshader_function_calls(*shader, func_rebind_map);
  }

  auto program = std::make_unique<glslang::TProgram>();

  program->addShader(shader.get());

  // From this point on the Program must be destroyed before Shaders so we'll
  // explicitly null it before returning but just let TShaders go out of scope

  if (!program->link(messages)) {
    error_string = "Error merging shader IR for shader \"" + pretty_name + "\":\n" +
                   program->getInfoLog();
    program = nullptr;
    return {rtn_spirv, error_string};
  }

  if (!program->getIntermediate(glslang_shader_stage)) {
    error_string =
        "Error getting IR for shader \"" + pretty_name + "\":\n" + program->getInfoLog();
    program = nullptr;
    return {rtn_spirv, error_string};
  }

  IoMapResolver io_map_resolver;
  glslang::TGlslIoMapper glsl_io_mapper;
  if (!program->mapIO(&io_map_resolver, &glsl_io_mapper)) {
    error_string =
        "Error mapping I/O for shader \"" + pretty_name + "\":\n" + program->getInfoLog();
    program = nullptr;
    return {rtn_spirv, error_string};
  }

  if (!program->buildReflection()) {
    error_string = "Error building reflection for shader \"" + pretty_name + "\":\n" +
                   program->getInfoLog();
    program = nullptr;
    return {rtn_spirv, error_string};
  }

  if (dump_AST) {
    LOG(INFO) << program->getInfoDebugLog();
  }

  spv::SpvBuildLogger spv_logger;
  glslang::GlslangToSpv(
      *program->getIntermediate(glslang_shader_stage), rtn_spirv, &spv_logger);

  if (rtn_spirv.empty()) {
    error_string = "Spirv generation from IR failed for shader \"" + pretty_name +
                   "\":\n" + spv_logger.getAllMessages();
  }

  if (dump_spv_build_log) {
    LOG(INFO) << "Spirv build log:";
    LOG(INFO) << spv_logger.getAllMessages();
  }

  // cleanup and check result before returning
  program = nullptr;

  return {rtn_spirv, error_string};
}

}  // namespace gfx
