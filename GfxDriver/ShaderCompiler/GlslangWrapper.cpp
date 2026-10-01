/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/GlslangWrapper.h"

#include <charconv>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>

// LiveTraverser.h must precede iomapper.h, which no longer includes it and
// relies on its callers to have brought in the glslang AST types.
#include <glslang/MachineIndependent/LiveTraverser.h>
#include <glslang/MachineIndependent/iomapper.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/ShaderCompiler/Library.h"
#include "GfxDriver/ShaderCompiler/ResourceLimits.h"
#include "GfxDriver/ShaderCompiler/ShaderRedecorator.h"

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

// Reads a run of decimal digits terminated by ':', advancing cursor past that
// colon. Returns nullopt if the text at cursor is not that shape, in which case
// cursor is left unspecified.
std::optional<int> parse_colon_terminated_number(const std::string& text,
                                                 size_t& cursor) {
  auto const* const first = text.data() + cursor;
  auto const* const last = text.data() + text.size();
  int value{};
  auto const [end, error] = std::from_chars(first, last, value);
  if (error != std::errc{} || end == last || *end != ':') {
    return std::nullopt;
  }
  cursor = static_cast<size_t>(end - text.data()) + 1;
  return value;
}

// glslang exposes diagnostics only as log text, formatted
// "ERROR: <string#>:<line>: ...", for example "ERROR: 0:129: '' : syntax error".
// Entries not matching that shape are skipped rather than asserted on. The
// format is not contractual, and the log also carries summary lines such as
// "ERROR: 1 compilation errors." that legitimately have no line number, so a
// format change should cost us the source annotation rather than abort the
// process while it is trying to report a shader bug.
std::set<int> find_error_line_numbers(const std::string& info_log) {
  constexpr std::string_view kPrefix = "ERROR: ";
  std::set<int> line_numbers;
  for (auto pos = info_log.find(kPrefix); pos != std::string::npos;
       pos = info_log.find(kPrefix, pos + kPrefix.size())) {
    auto cursor = pos + kPrefix.size();
    if (!parse_colon_terminated_number(info_log, cursor)) {
      continue;  // the string number, whose value we never need
    }
    if (auto const line_number = parse_colon_terminated_number(info_log, cursor)) {
      line_numbers.insert(*line_number);
    }
  }
  return line_numbers;
}

std::string add_line_numbers(const std::string& code,
                             const std::set<int>& error_line_nums) {
  std::stringstream ss;
  std::istringstream input;
  input.str(code);
  int line_num = 1;
  for (std::string line; std::getline(input, line); ++line_num) {
    ss << (error_line_nums.count(line_num) ? "-->" : "   ");
    ss << std::right << std::setw(5) << line_num << "  " << line << "\n";
  }
  return ss.str();
}

}  // namespace
#endif

GlslangWrapper::CompileResults GlslangWrapper::glslToSpirv(
    const std::vector<StageSource>& stages) {
  // These may become function parameters in the future so using constexpr
  // instead of #defines
  constexpr bool dump_AST = false;
  constexpr bool dump_spv_build_log = false;

  CompileResults results;
  results.stages.resize(stages.size());

  EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
  if (dump_AST) {
    messages = (EShMessages)(messages | EShMsgAST);
  }

  // A program holds pointers into its shaders, so it has to be destroyed first. The
  // declaration order below gives that for free on every exit from this function. Both
  // are held until the end rather than per stage because the shared resolver reads the
  // glslang objects as it assigns, so they all have to outlive the whole material.
  std::vector<TShaderUqPtr> shaders;
  std::vector<TProgramUqPtr> programs;
  shaders.reserve(stages.size());
  programs.reserve(stages.size());

  // A shader keeps the arrays it is handed rather than copying them, so these have to
  // outlive it, not merely the parse call below
  std::vector<const char*> source_pointers(stages.size());
  std::vector<int> source_lengths(stages.size());

  // Parse every stage before anything else, so that the shaders array stays aligned
  // with the stages array even where one fails
  bool parsed_all{true};
  for (size_t i = 0; i < stages.size(); ++i) {
    auto const& stage = stages[i];
    auto const glslang_shader_stage = shader_stage_to_glslang_enum(stage.shader_stage);
    auto& shader =
        *shaders.emplace_back(std::make_unique<glslang::TShader>(glslang_shader_stage));

    source_pointers[i] = stage.source.c_str();
    source_lengths[i] = static_cast<int>(stage.source.size());
    shader.setStringsWithLengths(&source_pointers[i], &source_lengths[i], 1);
    if (!stage.entry_point.empty()) {
      // These both need to be set if source language is GLSL (HLSL is different)
      shader.setEntryPoint(stage.entry_point.c_str());
      shader.setSourceEntryPoint(stage.entry_point.c_str());
    }

    shader.setAutoMapBindings(true);
    shader.setAutoMapLocations(true);
    shader.setEnvInput(
        glslang::EShSourceGlsl, glslang_shader_stage, glslang::EShClientVulkan, 100);
    // Matches kMinVulkanDeviceApiVersion, the floor every device we accept
    // already clears. SPIR-V 1.6 is core in Vulkan 1.3, and is also the highest
    // version Vulkan 1.4 accepts, so this is the ceiling either way.
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

    if (!shader.parse(
            &resources_, 100, ECoreProfile, false, false, messages, includer_)) {
      // Just log the error but don't throw yet. This allows ShaderManager to save
      // artifacts and handle the error
      std::string info_log(shader.getInfoLog());
      auto& error_string = results.stages[i].second;
      error_string = "Error parsing shader \"" + stage.pretty_name + "\":\n" + info_log;
#ifndef NDEBUG
      auto const error_line_nums = find_error_line_numbers(info_log);
      error_string += "Shader Source:\n" +
                      add_line_numbers(stage.source, error_line_nums) +
                      "End Shader Source\n";
#endif
      // Report it against the material too, so that the stages which did parse have
      // something better than a bare "no spirv" to offer
      results.material_error += error_string;
      parsed_all = false;
    }
  }

  // The stages share a resolver, so assigning bindings from an incomplete material
  // would assign the wrong ones. One parse failure therefore ends the whole compile.
  if (!parsed_all) {
    return results;
  }

  // One program per stage, since a material may hold two shaders of the same stage and
  // a program has room for only one intermediate per stage
  for (size_t i = 0; i < stages.size(); ++i) {
    auto const& stage = stages[i];
    auto& program = *programs.emplace_back(std::make_unique<glslang::TProgram>());
    program.addShader(shaders[i].get());

    if (!program.link(messages)) {
      results.stages[i].second = "Error merging shader IR for shader \"" +
                                 stage.pretty_name + "\":\n" + program.getInfoLog();
      return results;
    }

    if (!program.getIntermediate(shader_stage_to_glslang_enum(stage.shader_stage))) {
      results.stages[i].second = "Error getting IR for shader \"" + stage.pretty_name +
                                 "\":\n" + program.getInfoLog();
      return results;
    }
  }

  // Shared across the stages so that a resource appearing in more than one of them is
  // assigned the same binding in each. Declared after the programs so that it is
  // destroyed before them: a resolver that remembers anything by name remembers it in
  // a glslang TString, which belongs to the pool of whichever program was current when
  // it was recorded.
  IoMapResolver io_map_resolver;

  for (size_t i = 0; i < stages.size(); ++i) {
    auto const& stage = stages[i];
    auto& program = *programs[i];

    glslang::TGlslIoMapper glsl_io_mapper;
    if (!program.mapIO(&io_map_resolver, &glsl_io_mapper)) {
      results.stages[i].second = "Error mapping I/O for shader \"" + stage.pretty_name +
                                 "\":\n" + program.getInfoLog();
      return results;
    }

    if (!program.buildReflection()) {
      results.stages[i].second = "Error building reflection for shader \"" +
                                 stage.pretty_name + "\":\n" + program.getInfoLog();
      return results;
    }

    if (dump_AST) {
      LOG(INFO) << program.getInfoDebugLog();
    }

    spv::SpvBuildLogger spv_logger;
    auto& spirv = results.stages[i].first;
    glslang::GlslangToSpv(
        *program.getIntermediate(shader_stage_to_glslang_enum(stage.shader_stage)),
        spirv,
        &spv_logger);

    if (spirv.empty()) {
      results.stages[i].second = "Spirv generation from IR failed for shader \"" +
                                 stage.pretty_name + "\":\n" +
                                 spv_logger.getAllMessages();
    }

    if (dump_spv_build_log) {
      LOG(INFO) << "Spirv build log for \"" << stage.pretty_name << "\":";
      LOG(INFO) << spv_logger.getAllMessages();
    }
  }

  return results;
}

}  // namespace gfx
