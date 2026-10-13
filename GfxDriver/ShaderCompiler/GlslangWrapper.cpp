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

// glslang's own resolver does all the work here, with one thing changed: it assigns a
// binding only to a resource the stage actually uses, and we need one for every resource
// the shader declares. Our shader library routinely declares resources a given stage
// never reads, and ShaderReflection is what Material binds descriptors from, so an
// unassigned resource is both missing from the reflection and liable to collide, every
// unassigned one being left to default to binding zero. Forcing the liveness flag is the
// one behaviour the hand-written resolver this replaced had that glslang's does not.
//
// It is held rather than derived from because glslang is built without RTTI, so the
// typeinfo that a class derived from it would refer to was never emitted and the link
// fails. Deriving from TIoMapResolver is fine by contrast: it has no out-of-line virtual
// to anchor its typeinfo, so the compiler emits that here instead. Hence the forwarding
// below, which is otherwise uninteresting.
class IoMapResolver : public glslang::TIoMapResolver {
 public:
  explicit IoMapResolver(const glslang::TIntermediate& intermediate)
      : resolver_{intermediate} {}

  // Set by the resolver itself rather than reported through the mapper's return value
  bool hasError() const { return resolver_.hasError; }

  // Any inter-stage varying that arrived without a location of its own. See
  // resolveInOutLocation below for why that cannot be allowed to pass.
  const std::vector<std::string>& unlocatedVaryings() const {
    return unlocated_varyings_;
  }

  int resolveBinding(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    ent.live = true;
    return resolver_.resolveBinding(stage, ent);
  }

  bool validateBinding(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return resolver_.validateBinding(stage, ent);
  }
  int resolveSet(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return resolver_.resolveSet(stage, ent);
  }
  int resolveUniformLocation(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return resolver_.resolveUniformLocation(stage, ent);
  }
  bool validateInOut(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return resolver_.validateInOut(stage, ent);
  }
  int resolveInOutLocation(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    auto const& type = ent.symbol->getType();
    auto const& qualifier = type.getQualifier();

    // A vertex input and a fragment output are both matched against the reflection
    // rather than against another stage, so either is free to be assigned here. Every
    // other varying has a stage on the far side of it.
    auto const is_vertex_input = stage == EShLangVertex && qualifier.isPipeInput();
    auto const is_fragment_output = stage == EShLangFragment && qualifier.isPipeOutput();
    if (!is_vertex_input && !is_fragment_output && !type.isBuiltIn() &&
        !qualifier.hasSpirvDecorate()) {
      recordIfUnlocated(ent);
    }

    // Our inter-stage interface blocks declare a location on each member, and giving the
    // block variable one as well is what spirv-val rejects as a member location. glslang
    // only declines to assign to a block whose first member is a built-in, so decline for
    // the rest of them here. Plain variables still get one, which vertex inputs rely on.
    if (type.isStruct()) {
      return ent.newLocation = -1;
    }
    return resolver_.resolveInOutLocation(stage, ent);
  }
  int resolveInOutComponent(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return resolver_.resolveInOutComponent(stage, ent);
  }
  int resolveInOutIndex(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    return resolver_.resolveInOutIndex(stage, ent);
  }
  void notifyBinding(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    resolver_.notifyBinding(stage, ent);
  }
  void notifyInOut(EShLanguage stage, glslang::TVarEntryInfo& ent) override {
    resolver_.notifyInOut(stage, ent);
  }
  void beginNotifications(EShLanguage stage) override {
    resolver_.beginNotifications(stage);
  }
  void endNotifications(EShLanguage stage) override { resolver_.endNotifications(stage); }
  void beginResolve(EShLanguage stage) override { resolver_.beginResolve(stage); }
  void endResolve(EShLanguage stage) override { resolver_.endResolve(stage); }
  void beginCollect(EShLanguage stage) override { resolver_.beginCollect(stage); }
  void endCollect(EShLanguage stage) override { resolver_.endCollect(stage); }
  void reserverStorageSlot(glslang::TVarEntryInfo& ent, TInfoSink& info_sink) override {
    resolver_.reserverStorageSlot(ent, info_sink);
  }
  void reserverResourceSlot(glslang::TVarEntryInfo& ent, TInfoSink& info_sink) override {
    resolver_.reserverResourceSlot(ent, info_sink);
  }
  void addStage(EShLanguage stage, glslang::TIntermediate& intermediate) override {
    resolver_.addStage(stage, intermediate);
  }

 private:
  // glslang lines a varying up across stages only where the shader declares its
  // location. Its own comment on the Vulkan path is that it "does not do proper
  // cross-stage lining up", and the OpenGL path we use here matches by name only for
  // resources, not across an interface. So a location assigned here agrees with the
  // next stage only by luck of declaration order, and does so silently: there is no
  // link step across our per-stage programs to catch the mismatch, and the shader
  // renders wrong rather than failing. Collect the offenders for glslToSpirv to fail
  // the compile on.
  //
  // Everything in the shader library declares its locations today, interface blocks
  // doing so on each member, so this should never fire. It exists because nothing else
  // would notice if that stopped being true.
  void recordIfUnlocated(const glslang::TVarEntryInfo& ent) {
    auto const& type = ent.symbol->getType();
    auto const name = std::string{ent.symbol->getAccessName().c_str()};
    if (type.isStruct()) {
      // A block may carry one location for the whole of it, its members then taking
      // consecutive locations from there, or one on each member. Either form fully
      // determines it, and both are in use here. Only a block with neither cannot be
      // matched, so report its members rather than the block, that being where the
      // locations are missing from.
      if (type.getQualifier().hasLocation()) {
        return;
      }
      // Built-in members are skipped for the same reason a built-in variable is: the
      // implicit gl_PerVertex output block every vertex shader has declares no
      // locations and needs none, and it is the members that are marked built-in
      // rather than the block, which is why testing the block above does not catch it.
      for (auto const& member : *type.getStruct()) {
        if (!member.type->isBuiltIn() && !member.type->getQualifier().hasLocation()) {
          unlocated_varyings_.emplace_back(name + "." +
                                           member.type->getFieldName().c_str());
        }
      }
    } else if (!type.getQualifier().hasLocation()) {
      unlocated_varyings_.emplace_back(name);
    }
  }

  glslang::TDefaultGlslIoResolver resolver_;
  std::vector<std::string> unlocated_varyings_;
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
  if (stages.empty()) {
    return results;
  }

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
  // assigned the same binding in each: the resolver keeps its slot maps keyed by
  // resource name and nothing clears them between stages. Declared after the programs
  // so that it is destroyed before them, because those names are glslang TStrings and
  // belong to the pool of whichever program was current when they were recorded.
  //
  // The reference intermediate only supplies settings that are identical across our
  // stages, such as the client and the auto-map flags, so any stage's will do.
  auto const first_stage = shader_stage_to_glslang_enum(stages.front().shader_stage);
  IoMapResolver io_map_resolver(*programs.front()->getIntermediate(first_stage));

  for (size_t i = 0; i < stages.size(); ++i) {
    auto const& stage = stages[i];
    auto& program = *programs[i];

    glslang::TGlslIoMapper glsl_io_mapper;
    if (!program.mapIO(&io_map_resolver, &glsl_io_mapper)) {
      results.stages[i].second = "Error mapping I/O for shader \"" + stage.pretty_name +
                                 "\":\n" + program.getInfoLog();
      return results;
    }

    // The resolver carries its own error flag, which the mapper's return value does
    // not report: TGlslIoMapper::addStage returns its own hadError, while a resource
    // whose explicit binding disagrees with the one it was given in another stage sets
    // hasError on the resolver. Without this the message only reaches the info log.
    if (io_map_resolver.hasError()) {
      results.stages[i].second = "Error resolving I/O for shader \"" + stage.pretty_name +
                                 "\":\n" + program.getInfoLog();
      return results;
    }

    // Checked per stage rather than once at the end so that the message names the stage
    // the varying wants a location adding to
    auto const& unlocated = io_map_resolver.unlocatedVaryings();
    if (!unlocated.empty()) {
      std::string error_string = "Shader \"" + stage.pretty_name +
                                 "\" has inter-stage varyings without an explicit "
                                 "location, which cannot be matched to the next stage:";
      for (auto const& varying : unlocated) {
        error_string += "\n    " + varying;
      }
      results.stages[i].second = error_string;
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
