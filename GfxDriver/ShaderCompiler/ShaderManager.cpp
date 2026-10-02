/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/ShaderManager.h"

// path management for builder serialization
// TODO(scb) std::filesystem all the things
#include <pwd.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <utility>

// string manipulation
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include <absl/strings/string_view.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/schema.h>
#include <rapidjson/stringbuffer.h>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/ShaderCompiler/GlslangWrapper.h"
#include "GfxDriver/ShaderCompiler/JSONSchemas.h"
#include "GfxDriver/ShaderCompiler/SpirvArtifacts.h"
#include "GfxDriver/ShaderCompiler/Types.h"
#include "GfxDriver/TypeGLSL.h"

#define GENERATE_COMPILE_STATS false
// WIP spirv-opt support
#define USE_SPIRV_OPT false

#if GENERATE_COMPILE_STATS
#include <iostream>
#include "GfxDriver/ShaderCompiler/ShaderCompilerStats.h"
#include "Shared/measure.h"
#endif

#if USE_SPIRV_OPT
#include <spirv-tools/optimizer.hpp>
#endif

namespace gfx {

#if GENERATE_COMPILE_STATS
static ShaderCompilerStatsReporter g_stats_reporter;
#endif

std::string stage_to_extension(ShaderStage stage) {
  switch (stage) {
    case ShaderStage::kVertex:
      return "vert";
    case ShaderStage::kFragment:
      return "frag";
    case ShaderStage::kGeometry:
      return "geom";
    case ShaderStage::kTessControl:
      return "tesc";
    case ShaderStage::kTessEval:
      return "tese";
    case ShaderStage::kRayGen:
      return "raygen";
    case ShaderStage::kAnyHit:
      return "anyhit";
    case ShaderStage::kClosestHit:
      return "closest";
    case ShaderStage::kMiss:
      return "miss";
    case ShaderStage::kIntersection:
      return "isect";
    case ShaderStage::kCallable:
      return "call";
    case ShaderStage::kCompute:
      return "comp";
    case ShaderStage::kMesh:
      return "mesh";
    case ShaderStage::kTask:
      return "task";
  }
  return "";
}

ShaderStage template_type_to_shader_stage(Library::TemplateType type) {
  switch (type) {
    case Library::TemplateType::kVertex:
      return ShaderStage::kVertex;
    case Library::TemplateType::kFragment:
      return ShaderStage::kFragment;
    case Library::TemplateType::kGeometry:
      return ShaderStage::kGeometry;
    case Library::TemplateType::kTessEval:
      return ShaderStage::kTessEval;
    case Library::TemplateType::kTessControl:
      return ShaderStage::kTessControl;
    case Library::TemplateType::kCompute:
      return ShaderStage::kCompute;
    case Library::TemplateType::kRayGen:
      return ShaderStage::kRayGen;
    case Library::TemplateType::kAnyHit:
      return ShaderStage::kAnyHit;
    case Library::TemplateType::kClosestHit:
      return ShaderStage::kClosestHit;
    case Library::TemplateType::kMiss:
      return ShaderStage::kMiss;
    case Library::TemplateType::kIntersection:
      return ShaderStage::kIntersection;
    case Library::TemplateType::kCallable:
      return ShaderStage::kCallable;
    case Library::TemplateType::kMesh:
      return ShaderStage::kMesh;
    case Library::TemplateType::kTask:
      return ShaderStage::kTask;
    // TODO(scb) need to handle this better without breaking
    // the ShaderStage enum
    case Library::TemplateType::kGlsl:
      return ShaderStage::kVertex;
  }
  return ShaderStage::kVertex;
}

constexpr char tag_open = '<';
constexpr char tag_close = '>';
std::string make_tag_string(const std::string& tag_base) {
  return std::string(tag_open + tag_base + tag_close);
}

//
// Builder Serialization / Deserialization
//
namespace {
using OpType = ShaderManager::Builder::OpType;
static std::unordered_map<std::string, OpType> string_to_operator_map = {
    {"kAddPreamble", OpType::kAddPreamble},
    {"kReplaceFirstTag", OpType::kReplaceFirstTag},
    {"kReplaceAllTags", OpType::kReplaceAllTags},
    {"kReplaceAll", OpType::kReplaceAll},
    {"kReplaceAllMultiple", OpType::kReplaceAllMultiple},
    {"kPrependTemplate", OpType::kPrependTemplate},
    {"kAppendTemplate", OpType::kAppendTemplate},
    {"kReplaceTemplate", OpType::kReplaceTemplate},
    {"kInsertBeforeFunc", OpType::kInsertBeforeFunc},
    {"kReplaceFuncCall", OpType::kReplaceFuncCall},
    {"kReplaceFuncDef", OpType::kReplaceFuncDef},
    {"kReplaceWithSubBuilder", OpType::kReplaceWithSubBuilder},
    {"kAppendSubBuilder", OpType::kAppendSubBuilder}};

std::string to_string(const ShaderManager::Builder::OpType op) {
  switch (op) {
    case OpType::kAddPreamble:
      return "kAddPreamble";
    case OpType::kReplaceFirstTag:
      return "kReplaceFirstTag";
    case OpType::kReplaceAllTags:
      return "kReplaceAllTags";
    case OpType::kReplaceAll:
      return "kReplaceAll";
    case OpType::kReplaceAllMultiple:
      return "kReplaceAllMultiple";
    case OpType::kPrependTemplate:
      return "kPrependTemplate";
    case OpType::kAppendTemplate:
      return "kAppendTemplate";
    case OpType::kReplaceTemplate:
      return "kReplaceTemplate";
    case OpType::kInsertBeforeFunc:
      return "kInsertBeforeFunc";
    case OpType::kReplaceFuncCall:
      return "kReplaceFuncCall";
    case OpType::kReplaceFuncDef:
      return "kReplaceFuncDef";
    case OpType::kReplaceWithSubBuilder:
      return "kReplaceWithSubBuilder";
    case OpType::kAppendSubBuilder:
      return "kAppendSubBuilder";
  }
  return "";
}

void validate_builder_json_schema(const rapidjson::Document& document) {
  using namespace rapidjson;

  // Just use statics here to avoid pulling rapidjson into the main header
  static std::unique_ptr<SchemaDocument> schema_doc;
  static std::unique_ptr<SchemaValidator> validator;
  if (!validator) {
    // Prepare schema validator
    Document sd;
    if (sd.Parse(get_shader_builder_schema().c_str()).HasParseError()) {
      LOG(ERROR) << "Invalid schema";
    }
    schema_doc = std::make_unique<SchemaDocument>(sd);
    validator = std::make_unique<SchemaValidator>(*schema_doc);
  } else {
    validator->Reset();
  }

  if (!document.Accept(*validator)) {
    // Invalid according to schema, build error string.
    StringBuffer sb;
    validator->GetInvalidSchemaPointer().StringifyUriFragment(sb);
    std::string err("Schema validation failed: " + std::string(sb.GetString()) +
                    "\nSchema keyword: " + validator->GetInvalidSchemaKeyword());
    sb.Clear();
    validator->GetInvalidDocumentPointer().StringifyUriFragment(sb);
    err += std::string("\nDocument pointer: ") + sb.GetString();
    THROW_RUNTIME_EX(err);
  }
}
}  // namespace

struct ShaderManager::Serializer {
  // Converts a Builder into a JSON Object. Any sub-builders contained in the operators
  // list are also serialized to JSON objects recursively. This is hidden away in a
  // forward-declared struct so that we can keep rapidjson out of this class' headers.
  static void serialize_builder_to_json(const ShaderManager::Builder& builder,
                                        rapidjson::Value& builder_json,
                                        rapidjson::Document::AllocatorType& allocator) {
    // Shader template
    std::string name = builder.root_item_.internal_name;
    builder_json.AddMember("baseTemplate", name, allocator);

    // Entrypoint
    if (!builder.entry_point_.empty()) {
      builder_json.AddMember("entryPoint", builder.entry_point_, allocator);
    }

    // Operators
    if (!builder.op_list_.empty()) {
      rapidjson::Value op_array(rapidjson::kArrayType);
      for (auto const& op : builder.op_list_) {
        rapidjson::Value op_json(rapidjson::kObjectType);
        op_json.AddMember("name", to_string(op.type), allocator);
        op_json.AddMember("string1", op.str1, allocator);
        op_json.AddMember("string2", op.str2, allocator);
        // Recursively encode and add sub-builder object
        if (op.sub_builder) {
          rapidjson::Value sub_builder_json(rapidjson::kObjectType);
          serialize_builder_to_json(*op.sub_builder, sub_builder_json, allocator);
          op_json.AddMember("sub_builder", sub_builder_json, allocator);
        }
        op_json.AddMember("required", op.is_required, allocator);
        // Serialize the name_map (used by ReplaceAllMultiple)
        if (!op.name_map.empty()) {
          rapidjson::Value name_array(rapidjson::kArrayType);
          for (auto const& name_pair : op.name_map) {
            rapidjson::Value name_pair_json(rapidjson::kObjectType);
            name_pair_json.AddMember("string1", name_pair.first, allocator);
            name_pair_json.AddMember("string2", name_pair.second, allocator);
            name_array.PushBack(name_pair_json, allocator);
          }
          op_json.AddMember("name_map", name_array, allocator);
        }
        op_array.PushBack(op_json, allocator);
      }
      builder_json.AddMember("operators", op_array, allocator);
    }

    // Subroutines
    if (!builder.subroutine_map_.empty()) {
      rapidjson::Value sub_array(rapidjson::kArrayType);
      for (const auto& sub : builder.subroutine_map_) {
        rapidjson::Value sub_json(rapidjson::kObjectType);
        sub_json.AddMember("call", sub.first, allocator);
        sub_json.AddMember("target", sub.second.first, allocator);
        sub_json.AddMember("required", sub.second.second, allocator);
        sub_array.PushBack(sub_json, allocator);
      }
      builder_json.AddMember("subroutines", sub_array, allocator);
    }
  }

  // Recursive function to deserialize a (sub-Builder)
  static ShaderManager::BuilderUqPtr deserialize_builder_fragment(
      rapidjson::Value& doc,
      const ShaderManager& shader_mgr) {
    using namespace rapidjson;

    Value& template_name = doc["baseTemplate"];
    std::string entry_point;
    if (doc.HasMember("entryPoint")) {
      entry_point = doc["entryPoint"].GetString();
    }

    auto builder = shader_mgr.createBuilder(
        template_name.GetString(), Builder::Requirements::kNothing, entry_point);
    CHECK(builder);

    if (doc.HasMember("operators")) {
      Value& operators = doc["operators"];
      if (operators.IsArray()) {
        for (auto& op : operators.GetArray()) {
          Value& val_name = op["name"];
          Value& val_str1 = op["string1"];
          Value& val_str2 = op["string2"];
          Value& val_req = op["required"];

          auto op_itr = string_to_operator_map.find(val_name.GetString());
          CHECK(op_itr != string_to_operator_map.end());

          std::string str1 = val_str1.GetString();
          std::string str2 = val_str2.GetString();
          bool req = val_req.GetBool();

          switch (op_itr->second) {
            case OpType::kAddPreamble:
              builder->addPreambleString(str1);
              break;
            case OpType::kReplaceFirstTag:
              builder->replaceFirstTag(str1, str2);
              break;
            case OpType::kReplaceAllTags:
              builder->replaceAllTags(str1, str2);
              break;
            case OpType::kReplaceAll:
              builder->replaceAll(str1, str2);
              break;
            case OpType::kReplaceAllMultiple: {
              CHECK(op.HasMember("name_map"));
              Value& val_name_map = op["name_map"];
              CHECK(val_name_map.IsArray());
              Builder::NameMap name_map;
              for (auto& name_pair : val_name_map.GetArray()) {
                std::string str1 = name_pair["string1"].GetString();
                std::string str2 = name_pair["string2"].GetString();
                name_map.emplace_back(std::make_pair(std::move(str1), std::move(str2)));
              }
              builder->replaceAllMultiple(std::move(name_map));
            } break;
            case OpType::kPrependTemplate:
              builder->prependTemplate(str1);
              break;
            case OpType::kAppendTemplate:
              builder->appendTemplate(str1);
              break;
            case OpType::kReplaceTemplate:
              builder->replaceTagWithTemplate(str1, str2);
              break;
            case OpType::kInsertBeforeFunc:
              builder->insertBeforeFunction(str1, str2);
              break;
            case OpType::kReplaceFuncCall:
              builder->replaceFunctionCall(str1, str2);
              break;
            case OpType::kReplaceFuncDef:
              builder->replaceFunctionDefinition(str1, str2, req);
              break;
            case OpType::kReplaceWithSubBuilder: {
              Value& val_sub_builder = op["sub_builder"];
              BuilderShPtr sub_builder =
                  deserialize_builder_fragment(val_sub_builder, shader_mgr);
              builder->replaceFunctionWithSubBuilder(str1, sub_builder, req);
            } break;
            case OpType::kAppendSubBuilder: {
              Value& val_sub_builder = op["sub_builder"];
              BuilderShPtr sub_builder =
                  deserialize_builder_fragment(val_sub_builder, shader_mgr);
              builder->appendSubBuilder(sub_builder);
            } break;
          }
        }
      }
    }

    if (doc.HasMember("subroutines")) {
      Value& subroutines = doc["subroutines"];
      if (subroutines.IsArray()) {
        for (auto& sr : subroutines.GetArray()) {
          Value& val_call = sr["call"];
          Value& val_targ = sr["target"];
          Value& val_req = sr["required"];

          builder->addSubroutineBinding(
              val_call.GetString(), val_targ.GetString(), val_req.GetBool());
        }
      }
    }

    return builder;
  }
};

void ShaderManager::serializeBuilder(const Builder& builder,
                                     const std::string& filename) const {
  if (shader_artifacts_enabled_in_build()) {
    if (!can_save_artifacts_) {
      LOG(WARNING) << "Unable to serialize shader builder, artifact saving unavailable";
      return;
    }

    rapidjson::Document doc;
    doc.SetObject();
    auto& allocator = doc.GetAllocator();
    Serializer::serialize_builder_to_json(builder, doc, allocator);

    // Sanity check against the schema
    validate_builder_json_schema(doc);

    // convert to a pretty json string
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetIndent(' ', 2);
    doc.Accept(writer);

    // write to file
    std::string fullpath(artifact_path_ + filename + ".builder");
    std::fstream file(fullpath, std::fstream::out);
    if (!file) {
      LOG(WARNING) << "Failed to open builder serialization file \'" + fullpath + "\'.";
      return;
    }

    file << buffer.GetString();
  }
}

ShaderManager::BuilderUqPtr ShaderManager::deserializeBuilder(
    const std::string& json_string) {
  // Initial parsing
  rapidjson::Document doc;
  if (doc.Parse(json_string.c_str()).HasParseError()) {
    // Not a valid JSON
    std::string err(GetParseError_En(doc.GetParseError()));
    THROW_RUNTIME_EX("Error parsing builder serialization file: " + err);
  }

  validate_builder_json_schema(doc);

  auto builder = Serializer::deserialize_builder_fragment(doc, *this);

  return builder;
}

ShaderManager::Builder::Operator::Operator(OpType type,
                                           std::string str1,
                                           std::string str2,
                                           BuilderShPtr sub_builder,
                                           bool is_required)
    : type{type}
    , str1{str1}
    , str2{str2}
    , sub_builder{std::move(sub_builder)}
    , is_required{is_required} {}

ShaderManager::Builder::Operator::Operator(OpType type, NameMap&& name_map)
    : type{type}, name_map{std::move(name_map)} {}

ShaderManager::Builder::Builder(const Library& library,
                                const Library::Item& lib_item,
                                Requirements requirements,
                                const std::string& entry_point)
    : library_{library}
    , root_item_{lib_item}
    , item_indices_{lib_item.index}
    , shader_stage_{template_type_to_shader_stage(lib_item.template_type)}
    , requirements_{requirements}
    , entry_point_{entry_point}
    , raytracing_hit_group_index_{0} {}

ShaderManager::Builder::~Builder() {}

void ShaderManager::Builder::addOperator(OpType type,
                                         std::string str1,
                                         std::string str2,
                                         BuilderShPtr sub_builder,
                                         bool is_required) {
  // TODO(is_required to std::optional)
  is_processed_ = false;
  op_list_.emplace_back(
      type, std::move(str1), std::move(str2), std::move(sub_builder), is_required);
}

void ShaderManager::Builder::addOperator(OpType type, NameMap&& name_map) {
  is_processed_ = false;
  op_list_.emplace_back(type, std::move(name_map));
}

const std::string& ShaderManager::Builder::getTemplateName() const {
  return root_item_.internal_name;
}

ShaderStage ShaderManager::Builder::getShaderStage() const {
  return shader_stage_;
}

const std::string& ShaderManager::Builder::getEntryPoint() const {
  return entry_point_;
}

std::string ShaderManager::Builder::makeTag(const std::string& name) const {
  return make_tag_string(name);
}

void ShaderManager::Builder::addPreambleString(std::string str) {
  addOperator(OpType::kAddPreamble, std::move(str));
}

void ShaderManager::Builder::replaceAll(std::string search_str, std::string new_str) {
  addOperator(OpType::kReplaceAll, std::move(search_str), std::move(new_str));
}

void ShaderManager::Builder::replaceAllMultiple(NameMap&& name_map) {
  addOperator(OpType::kReplaceAllMultiple, std::move(name_map));
}

void ShaderManager::Builder::replaceFirstTag(std::string tag_str, std::string new_str) {
  addOperator(OpType::kReplaceFirstTag, std::move(tag_str), std::move(new_str));
}

void ShaderManager::Builder::replaceAllTags(std::string tag_str, std::string new_str) {
  addOperator(OpType::kReplaceAllTags, std::move(tag_str), std::move(new_str));
}

void ShaderManager::Builder::prependTemplate(std::string name_str) {
  auto const& item = library_.get(name_str);
  item_indices_.emplace_back(item.index);
  // TODO(scb) Store Item index instead of string in operator
  addOperator(OpType::kPrependTemplate, std::move(name_str));
}

void ShaderManager::Builder::appendTemplate(std::string name_str) {
  auto const& item = library_.get(name_str);
  item_indices_.emplace_back(item.index);
  // TODO(scb) Store Item index instead of string in operator
  addOperator(OpType::kAppendTemplate, std::move(name_str));
}

void ShaderManager::Builder::replaceTagWithTemplate(std::string tag_str,
                                                    std::string template_name) {
  auto const& item = library_.get(template_name);
  item_indices_.emplace_back(item.index);
  // TODO(scb) Store Item index instead of string in operator
  addOperator(OpType::kReplaceTemplate, std::move(tag_str), std::move(template_name));
}

void ShaderManager::Builder::setAttributeType(const std::string& name, bool is_uniform) {
  replaceFirstTag(std::string("useU") + name, (is_uniform ? "1" : "0"));
}

void ShaderManager::Builder::setPropertyInOutTypes(const std::string& prop_name,
                                                   const gfx::BaseTypeGLSL& in_type,
                                                   const gfx::BaseTypeGLSL& out_type) {
  std::string in_name("inT" + prop_name);
  replaceFirstTag(in_name + "Type", in_type.declString());
  replaceFirstTag(in_name + "Enum", in_type.enumString());

  std::string out_name = ("outT" + prop_name);
  replaceFirstTag(out_name + "Type", out_type.declString());
  replaceFirstTag(out_name + "Enum", out_type.enumString());
}

void ShaderManager::Builder::setAllPropertyTypes(const std::string& name,
                                                 bool is_uniform,
                                                 BufferAttrType type) {
  replaceFirstTag(std::string("useU") + name, (is_uniform ? "1" : "0"));

  std::string in_name("inT" + name);
  replaceFirstTag(in_name + "Enum", to_string(type));
  replaceFirstTag(in_name + "Type", to_string_glsl_decl(type));

  std::string out_name("outT" + name);
  replaceFirstTag(out_name + "Enum", to_string(type));
  replaceFirstTag(out_name + "Type", to_string_glsl_decl(type));
}

void ShaderManager::Builder::insertBeforeFunction(std::string search_str,
                                                  std::string new_str) {
  addOperator(OpType::kInsertBeforeFunc, std::move(search_str), std::move(new_str));
}

void ShaderManager::Builder::replaceFunctionCall(std::string orig_sig,
                                                 std::string new_sig) {
  addOperator(OpType::kReplaceFuncCall, std::move(orig_sig), std::move(new_sig));
}

void ShaderManager::Builder::replaceFunctionDefinition(std::string name,
                                                       std::string body,
                                                       bool is_required) {
  addOperator(
      OpType::kReplaceFuncDef, std::move(name), std::move(body), nullptr, is_required);
}

void ShaderManager::Builder::replaceFunctionWithSubBuilder(std::string name,
                                                           BuilderShPtr sub_builder,
                                                           bool is_required) {
  addOperator(OpType::kReplaceWithSubBuilder,
              std::move(name),
              std::string(),
              std::move(sub_builder),
              is_required);
}

void ShaderManager::Builder::appendSubBuilder(BuilderShPtr sub_builder) {
  addOperator(
      OpType::kAppendSubBuilder, std::string(), std::string(), std::move(sub_builder));
}

void ShaderManager::Builder::addSubroutineBinding(std::string call,
                                                  std::string target,
                                                  bool is_required) {
  subroutine_map_[call] = {target, is_required};
}

void ShaderManager::Builder::setExternalUniformBuffers(
    std::set<std::string_view>&& external_uniform_buffer_names) {
  external_uniform_buffer_names_ = std::move(external_uniform_buffer_names);
}

void ShaderManager::Builder::addExternalUniformBuffer(
    std::string_view external_uniform_buffer_name) {
  auto [itr, did_insert] =
      external_uniform_buffer_names_.insert(external_uniform_buffer_name);
  CHECK(did_insert) << "Redundant external uniform buffer declaration for \'" << *itr
                    << "\'";
}

void ShaderManager::Builder::setRaytracingHitGroupIndex(uint32_t index) {
  raytracing_hit_group_index_ = index;
}

namespace {
using str_itr = std::string::iterator;
using str_itr_range = boost::iterator_range<str_itr>;

bool is_identifier_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

// The identifier immediately before `pos`, ignoring whitespace. Empty if whatever
// precedes it is punctuation or an operator.
std::string preceding_identifier(const std::string& code_str, size_t pos) {
  while (pos > 0 && std::isspace(static_cast<unsigned char>(code_str[pos - 1])) != 0) {
    --pos;
  }
  auto const end = pos;
  while (pos > 0 && is_identifier_char(code_str[pos - 1])) {
    --pos;
  }
  return code_str.substr(pos, end - pos);
}

// Every place `func_name` is named and then opens an argument or parameter list.
//
// A definition names its return type immediately before the function name and
// nothing else does, bar a `return`. That is the whole test. Neither the parameter
// list nor what follows it can be relied on, because a list may carry preprocessor
// directives: quantitativeScaleTemplate.vert picks the type of the last parameter
// with an #if, and in one case leaves the body on the far side of the #endif.
struct FunctionReference {
  size_t position;
  bool is_definition;
};

std::vector<FunctionReference> find_function_references(const std::string& code_str,
                                                        const std::string& func_name) {
  // Keywords that can stand where a return type otherwise would
  static const std::set<std::string> kNotATypeName = {"return", "else", "do", "case"};

  std::vector<FunctionReference> references;
  for (auto pos = code_str.find(func_name); pos != std::string::npos;
       pos = code_str.find(func_name, pos + func_name.size())) {
    if (pos > 0 && is_identifier_char(code_str[pos - 1])) {
      continue;  // a longer name that merely ends with this one
    }
    auto after = pos + func_name.size();
    while (after < code_str.size() &&
           (code_str[after] == ' ' || code_str[after] == '\t')) {
      ++after;
    }
    if (after >= code_str.size() || code_str[after] != '(') {
      continue;  // named but not called, so nothing to retarget
    }
    auto const preceding = preceding_identifier(code_str, pos);
    references.push_back(
        {pos, !preceding.empty() && kNotATypeName.count(preceding) == 0});
  }
  return references;
}

bool has_function_definition(const std::string& code_str, const std::string& func_name) {
  auto const references = find_function_references(code_str, func_name);
  return std::any_of(references.begin(),
                     references.end(),
                     [](const FunctionReference& ref) { return ref.is_definition; });
}

str_itr_range get_function_bounds(std::string& code_str, const std::string& func_name) {
  std::string regex_str = R"(\h*\w+\h+)" + func_name + R"(\h*\([\w\h\v,]*\)\h*\v*\{)";
  boost::regex func_signature_regex(regex_str);

  auto signature_range = boost::find_regex(code_str, func_signature_regex);

  if (signature_range.empty()) {
    return signature_range;
  }

  auto last_itr = signature_range.end();
  std::vector<str_itr> scope_stack = {last_itr - 1};

  size_t curr_pos = signature_range.end() - code_str.begin();
  while ((curr_pos = code_str.find_first_of("{}", curr_pos)) != std::string::npos) {
    if (code_str[curr_pos] == '{') {
      scope_stack.push_back(code_str.begin() + curr_pos);
    } else {
      // found a '}'
      scope_stack.pop_back();
      if (scope_stack.empty()) {
        last_itr = code_str.begin() + curr_pos + 1;
        break;
      }
    }
    curr_pos += 1;
  }

  if (!scope_stack.empty()) {
    // return an empty range
    return str_itr_range();
  }

  return str_itr_range(signature_range.begin(), last_itr);
}

// Finds every call to `func_name`, leaving its own definition alone so that
// retargeting the calls leaves the definition behind. glslang drops it during dead
// code elimination once nothing reaches it, which is what the AST pass this replaced
// relied on too.
std::vector<size_t> find_call_sites(const std::string& code_str,
                                    const std::string& func_name) {
  std::vector<size_t> call_positions;
  for (auto const& reference : find_function_references(code_str, func_name)) {
    if (!reference.is_definition) {
      call_positions.push_back(reference.position);
    }
  }
  return call_positions;
}

void replace_call_sites(std::string& code_str,
                        const std::vector<size_t>& call_positions,
                        size_t call_name_length,
                        const std::string& target_name) {
  // Back to front, so each replacement leaves the earlier offsets valid
  for (auto itr = call_positions.rbegin(); itr != call_positions.rend(); ++itr) {
    code_str.replace(*itr, call_name_length, target_name);
  }
}

// TODO: GLSL version constant. See also spirv-opt and spirv-cross usages
// TODO: string_view all the things
constexpr char spirv_shader_header[] = R"(#version 460 core
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require
)";

constexpr char type_defines_string[] = R"(
#define BOOL 35670
#define BOOL_VEC2 35671
#define BOOL_VEC3 35672
#define BOOL_VEC4 35673
#define INT 5124
#define INT_VEC2 35667
#define INT_VEC3 35668
#define INT_VEC4 35669
#define FLOAT 5126
#define FLOAT_VEC2 35664
#define FLOAT_VEC3 35665
#define FLOAT_VEC4 35666
#define DOUBLE 5130
#define DOUBLE_VEC2 36860
#define DOUBLE_VEC3 36861
#define DOUBLE_VEC4 36862
#define UNSIGNED_INT 5125
#define UNSIGNED_INT_VEC2 36294
#define UNSIGNED_INT_VEC3 36295
#define UNSIGNED_INT_VEC4 36296
#define UNSIGNED_INT64_ARB 5135
#define UNSIGNED_INT64_VEC2_ARB 36853
#define UNSIGNED_INT64_VEC3_ARB 36854
#define UNSIGNED_INT64_VEC4_ARB 36855
#define INT64_ARB 5134
#define INT64_VEC2_ARB 36841
#define INT64_VEC3_ARB 36842
#define INT64_VEC4_ARB 36843
)";

std::string library_item_to_filename(const Library::Item& item) {
  std::string template_name(item.internal_name);
  auto stage = template_type_to_shader_stage(item.template_type);
  // strip internal path from base template name
  std::size_t pos = template_name.find_last_of("/");
  std::string basename =
      pos != std::string::npos ? template_name.substr(pos + 1) : template_name;
  // pop off any extension and add one based on shader stage. These are likely
  // identical but some have ".glsl" which isn't helpful
  return basename.substr(0, basename.find_first_of(".")) + "." +
         stage_to_extension(stage);
}

#if USE_SPIRV_OPT
// WIP spirv-opt integration
spirv_t optimize_spirv(const spirv_t& spirv_in) {
  spirv_t opt_spv;
  LOG(INFO) << "Optimizing Spir-V";
  // Must track what GlslangWrapper asks glslang to emit. This whole function is
  // behind USE_SPIRV_OPT, which is false, so the OpenGL target it carried until
  // now had never been exercised against a Vulkan blob.
  spv_target_env env = SPV_ENV_VULKAN_1_3;
  auto log_spv_msg = [](spv_message_level_t level,
                        const char* source,
                        const spv_position_t&,
                        const char* m) {
    switch (level) {
      case SPV_MSG_FATAL:
      case SPV_MSG_INTERNAL_ERROR:
      case SPV_MSG_ERROR:
        LOG(ERROR) << "Spirv-tools error: " << m << " Source: " << source;
        break;
      case SPV_MSG_WARNING:
        LOG(WARNING) << "Spirv-tools warning: " << m << "Source: " << source;
        break;
      case SPV_MSG_INFO:
        // TODO(scb): separate logfile support?
        VLOG(2) << "Spirv-tools: " << m;
        break;
      default:
        // TODO(scb): separate logfile support?
        VLOG(2) << "Spirv-tools: " << m;
    }
  };
  ::spvtools::Optimizer opt(env);
  opt.RegisterPerformancePasses().SetMessageConsumer(log_spv_msg);
  opt.Run(spirv_in.data(), spirv_in.size(), &opt_spv);

  return opt_spv;
}
#endif  // USE_SPIRV_OPT

}  // namespace

//
// ShaderManager
//
ShaderManager::ShaderManager(LibraryUqPtr library)
    : library_{std::move(library)}
    , can_save_artifacts_{false}
    , always_save_artifacts_{ShaderArtifactTypeBits::kNone} {
  CHECK(library_);
  glslang_wrapper_ = std::make_unique<GlslangWrapper>(*library_);
  library_->add("Marks/typeDefines.glsl", "glsl", "glsl", type_defines_string);
  library_->updateDictionaries();

  if ((can_save_artifacts_ = shader_artifacts_enabled_in_build()) == true) {
    artifact_path_ = get_artifact_pathname();
    can_save_artifacts_ = !artifact_path_.empty();
    LOG_IF(INFO, can_save_artifacts_)
        << "Shader artifact path: \"" << artifact_path_ << "\"";

    // sanity check artifact setup
    always_save_artifacts_ = is_always_save_artifacts_enabled();
    if (always_save_artifacts_ != ShaderArtifactTypeBits::kNone) {
      LOG(INFO) << "Always save shader artifacts enabled";
      if (!can_save_artifacts_) {
        LOG(WARNING) << "Artifact path is invalid or empty!!!";
        always_save_artifacts_ = ShaderArtifactTypeBits::kNone;
      }
    }
  }
}

ShaderManager::~ShaderManager() {
  glslang_wrapper_ = nullptr;
  library_ = nullptr;
#if GENERATE_COMPILE_STATS
  g_stats_reporter.generateReport(std::cout);
  std::cout << std::endl;
#endif
}

gfx::Library& ShaderManager::getLibrary() const {
  CHECK(library_);
  return *library_.get();
}

void ShaderManager::replaceLibrary(LibraryUqPtr library) {
  library_ = std::move(library);
  glslang_wrapper_ = std::make_unique<GlslangWrapper>(*library_);
  library_->add("Marks/typeDefines.glsl", "glsl", "glsl", type_defines_string);
  library_->updateDictionaries();
}

const std::string& ShaderManager::getTemplate(const std::string& internal_path) const {
  CHECK(library_);
  return library_->get(internal_path).code;
}

ShaderManager::TemplateDependencies ShaderManager::collectTemplateDependencies(
    const Builder& builder) const {
  TemplateDependencies dependencies;
  std::set<uint32_t> seen_includes;  // used to maintain uniqueness

  // Recursive function to process a Library::Item
  std::function<void(uint32_t)> process_item = [&](uint32_t item_index) {
    auto const& item = library_->get(item_index);
    dependencies.extension_indices.insert(item.extension_indices.begin(),
                                          item.extension_indices.end());

    for (auto include_index : item.include_indices) {
      if (seen_includes.insert(include_index).second) {
        // Get include name from dictionary
        auto const& dict_entry = library_->getInclude(include_index);
        // Call self on new Item, so an include's own includes precede it
        process_item(dict_entry.item->index);
        dependencies.include_indices.push_back(include_index);
      }
    }
  };

  // Process all Library templates referenced by this Builder
  for (auto index : builder.item_indices_) {
    process_item(index);
  }
  return dependencies;
}

std::string ShaderManager::buildExtensionAndIncludesString(Builder& builder) const {
  auto const dependencies = collectTemplateDependencies(builder);

  std::string extension_and_include_str{spirv_shader_header};
  for (auto index : dependencies.extension_indices) {
    absl::StrAppend(&extension_and_include_str, library_->getExtension(index).str);
  }
  for (auto index : dependencies.include_indices) {
    absl::StrAppend(&extension_and_include_str, library_->getInclude(index).str);
  }
  return extension_and_include_str;
}

// Retargets subroutine and function-call bindings in the assembled GLSL, replacing
// the glslang AST pass that used to do this after parsing.
//
// A binding only takes effect if its target is actually defined, which is how an
// optional binding stays a no-op when its target was never generated. The assembled
// source holds #include directives rather than the included text, so targets living
// in an include, as the colour conversion subroutines do, have to be looked up
// through the Library instead.
void ShaderManager::rebindSubroutineCalls(Builder& builder,
                                          const SubroutineMap& rebind_map) const {
  if (rebind_map.empty()) {
    return;
  }

  auto& final_glsl = builder.processed_code_;

  std::vector<const std::string*> include_sources;
  for (auto index : collectTemplateDependencies(builder).include_indices) {
    include_sources.push_back(&library_->getInclude(index).item->code);
  }

  auto const is_defined = [&](const std::string& target_name) {
    return has_function_definition(final_glsl, target_name) ||
           std::any_of(include_sources.begin(),
                       include_sources.end(),
                       [&target_name](const std::string* source) {
                         return has_function_definition(*source, target_name);
                       });
  };

  for (auto const& [call_name, binding] : rebind_map) {
    auto const& [target_name, is_required] = binding;

    // The AST pass this replaced walked call sites rather than bindings, so a binding
    // nothing calls was never checked. Keep that: an unused one must not fail a build
    auto const call_positions = find_call_sites(final_glsl, call_name);
    if (call_positions.empty()) {
      continue;
    }

    // An empty target name would make the definition regex match any function at all,
    // so reject it here rather than rebinding every call to the first one found
    if (target_name.empty() || !is_defined(target_name)) {
      RUNTIME_EX_ASSERT(
          !is_required,
          "Failed to rebind required shader function \'" + target_name + "\'");
      continue;
    }

    replace_call_sites(final_glsl, call_positions, call_name.size(), target_name);
  }
}

void ShaderManager::processOperators(Builder& builder,
                                     SubroutineMap* rebind_map,
                                     bool is_sub_builder) const {
  if (builder.is_processed_) {
    return;
  }
  // Copy the root template
  std::string output = builder.root_item_.code;
  CHECK(!output.empty());

  std::string local_preamble_str;
  for (const auto& op : builder.op_list_) {
    switch (op.type) {
      case OpType::kAddPreamble:
        absl::StrAppend(&local_preamble_str, op.str1);
        break;
      case OpType::kReplaceFirstTag:
      case OpType::kReplaceAllTags:
        output = absl::StrReplaceAll(output, {{make_tag_string(op.str1), op.str2}});
        break;
      case OpType::kReplaceAll:
        output = absl::StrReplaceAll(output, {{op.str1, op.str2}});
        break;
      case OpType::kReplaceAllMultiple:
        output = absl::StrReplaceAll(output, op.name_map);
        break;
      case OpType::kPrependTemplate:
        output = absl::StrCat(getTemplate(op.str1), output);
        break;
      case OpType::kAppendTemplate:
        absl::StrAppend(&output, getTemplate(op.str1));
        break;
      case OpType::kReplaceTemplate:
        boost::replace_first(output, make_tag_string(op.str1), getTemplate(op.str2));
        break;
      case OpType::kInsertBeforeFunc: {
        const std::string& func_name = op.str1;
        auto range = get_function_bounds(output, func_name);
        if (!range.empty()) {
          const auto pos = range.begin() - output.begin();
          output.insert(pos, op.str2);
        }
        break;
      }
      case OpType::kReplaceFuncCall: {
        if (rebind_map) {  // generating spir-v
          // push onto the call rebind map for IR fixup
          rebind_map->insert({op.str1, {op.str2, op.is_required}});
        } else {  // building substring
          output = absl::StrReplaceAll(output, {{op.str1, op.str2}});
        }
        break;
      }
      case OpType::kReplaceFuncDef: {
        // TODO(scb): This replaces a function definition with a whole chunk of new code,
        // usually a Scale. For example, getx() and gety() are typically replaced with
        // unique versions of quantitative scale, including UBO and sub functions. This
        // would be much better handled by simply adding these as a separate string that
        // we feed to glslang, leaving the original function in place. We can then simply
        // retarget the function calls to point to the new IntermAggregate node.
        // The tricky part will be dealing with decimal or project wrapped parameters.
        // We'll need to either restructure the shaders slightly, or explore more advanced
        // IR merging.
        const std::string& name = op.str1;
        auto range = get_function_bounds(output, name);
        if (!range.empty()) {
          boost::replace_range(output, range, op.str2);
        } else if (op.is_required) {
          // TODO(scb): Originally this was prefixed by the Mark's string operator output.
          // We need to add some debug information to the builder for these cases.
          RUNTIME_EX_ASSERT(
              false,
              "Cannot find a properly defined \"" + name + "\" function in the shader.");
        }
        break;
      }
      case OpType::kReplaceWithSubBuilder: {
        auto const& name = op.str1;
        auto range = get_function_bounds(output, name);
        if (!range.empty()) {
#if GENERATE_COMPILE_STATS
          auto start_time = timer_start();
#endif
          auto& sub_builder = *op.sub_builder;
          processOperators(sub_builder, nullptr, true);
#if GENERATE_COMPILE_STATS
          g_stats_reporter.addSubBuilderStats(sub_builder.getTemplateName(),
                                              timer_stop_microseconds(start_time));
#endif
          boost::replace_range(output, range, sub_builder.processed_code_);
        } else if (op.is_required) {
          RUNTIME_EX_ASSERT(
              false,
              "Cannot find a properly defined \"" + name + "\" function in the shader.");
        }
      } break;
      case OpType::kAppendSubBuilder: {
        auto& sub_builder = *op.sub_builder;
#if GENERATE_COMPILE_STATS
        auto start_time = timer_start();
#endif
        processOperators(sub_builder, nullptr, true);
#if GENERATE_COMPILE_STATS
        g_stats_reporter.addSubBuilderStats(sub_builder.getTemplateName(),
                                            timer_stop_microseconds(start_time));
#endif
        absl::StrAppend(&output, sub_builder.processed_code_);
      } break;
    }
  }

  auto& final_glsl = builder.processed_code_;
  // Sub-builders cannot have extensions or includes, but this is NOT enforced at
  // recording time
  if (!is_sub_builder) {
    // Add version, extensions, and includes
    final_glsl = buildExtensionAndIncludesString(builder);
  }
  // Add preamble and final code string
  if (local_preamble_str.empty()) {
    absl::StrAppend(&final_glsl, output);
  } else {
    absl::StrAppend(&final_glsl, local_preamble_str, output);
  }
  builder.is_processed_ = true;
}

// For Spirv, we generate a Glsl string and a function call replace map. This
// encapsulates both the kReplaceFuncCall operator and subroutines. The map is applied
// textually to the assembled source, so glslang is handed GLSL that already calls the
// right functions.
//
// Every stage of the material is assembled first and then compiled in one call, so
// that glslang can assign bindings and locations consistently across the stages
// instead of each stage being resolved as though it were alone.
ShaderCacheShPtrVector ShaderManager::buildSpirv(BuilderUqPtrVector& builders,
                                                 ShaderRedecorator* shader_redecorator,
                                                 bool save_artifacts) const {
#if GENERATE_COMPILE_STATS
  auto start_time = timer_start();
  std::vector<uint64_t> operator_times;
  operator_times.reserve(builders.size());
#endif

  // Assemble the GLSL for every stage before compiling any of it, since glslang needs
  // to see the whole material at once to assign its I/O across the stages
  std::vector<GlslangWrapper::StageSource> stage_sources;
  stage_sources.reserve(builders.size());
  for (auto& builder_ptr : builders) {
    auto& builder = *builder_ptr;
#if GENERATE_COMPILE_STATS
    start_time = timer_start();
#endif
    // Seed a unified function call replacement map to handle both subroutine
    // call rebinding and plain replacement operators. This can all be cleaned up once
    // we commit to Spirv exclusively and ditch the pure GLSL path.
    SubroutineMap func_rebind_map(builder.subroutine_map_.begin(),
                                  builder.subroutine_map_.end());

    // Build GLSL string and function bindings
    // TODO(scb): multi-string support in glslangWrapper
    processOperators(builder, &func_rebind_map, false);

    // Apply the bindings to the assembled source, before glslang ever sees it
    rebindSubroutineCalls(builder, func_rebind_map);

#if GENERATE_COMPILE_STATS
    operator_times.push_back(timer_stop_microseconds(start_time));
#endif

    stage_sources.push_back({library_item_to_filename(builder.root_item_),
                             builder.processed_code_,
                             builder.entry_point_,
                             builder.shader_stage_});
  }

#if GENERATE_COMPILE_STATS
  start_time = timer_start();
#endif

  // Generate spirv
  auto compile_results = glslang_wrapper_->glslToSpirv(stage_sources);
  CHECK_EQ(compile_results.stages.size(), builders.size());

#if GENERATE_COMPILE_STATS
  // One compile now covers the whole material, so there is no per stage figure to
  // report and each is charged the same total
  auto glsl_time = timer_stop_microseconds(start_time);
#endif

  ShaderCacheShPtrVector caches;
  caches.reserve(builders.size());

  for (size_t stage_index = 0; stage_index < builders.size(); ++stage_index) {
    auto& builder = *builders[stage_index];
    auto& compile_result = compile_results.stages[stage_index];
    auto const& pretty_name = stage_sources[stage_index].pretty_name;
    // Whether to save this stage's artifacts, which a failure forces on regardless of
    // what the caller asked for. Per stage, so one failure does not alter the others.
    auto stage_save_artifacts = save_artifacts;

    // default reflection
    ShaderReflection reflection;

    // populate reflection here?
    if (shader_redecorator && !compile_result.first.empty()) {
      // build reflection
      shader_redecorator->redecorate(
          compile_result.first, reflection, builder.getTemplateName());

      // validate presence of specified external uniform buffers
      for (auto const& name : builder.external_uniform_buffer_names_) {
        CHECK(reflection.getUniformBufferBinding(name) >= 0)
            << "Shader '" << builder.getTemplateName()
            << "' does not contain specified external uniform buffer '" << name << "'";
      }
    }

#if GENERATE_COMPILE_STATS
    g_stats_reporter.addBuildSpirvStats(builder.getShaderStage(),
                                        builder.getTemplateName(),
                                        operator_times[stage_index],
                                        glsl_time);
#endif

    spirv_t opt_spirv;

    spirv_t spirv;
    // Check if the spirv blob is empty
    if (!compile_result.first.empty()) {
      // TODO(scb): formalize spirv-opt pass / reflection
#if USE_SPIRV_OPT
      // TODO(scb): decide if we want to keep both the vanilla and optimized spirv
      // around. For now we only keep one, and just return vanilla vs optimized based on
      // the build flag.
      // Will resolve this with formal support for spirv-opt [BE-2779]
      spirv = optimize_spirv(std::move(compile_result.first));
#else
      spirv = std::move(compile_result.first);
#endif
    } else {
      // Write all artifacts on error with debug builds
      stage_save_artifacts = true;
    }

    // Save artifacts
    if (can_save_artifacts_) {
      ShaderArtifactTypeBits artifacts_to_save;

      if (stage_save_artifacts ||
          any_bits_set(builder.requirements_ & Builder::Requirements::kSaveArtifacts)) {
        artifacts_to_save = ShaderArtifactTypeBits::kAll;
      } else {
        artifacts_to_save = always_save_artifacts_;
      }

      if (artifacts_to_save) {
#if GENERATE_COMPILE_STATS
        static bool did_warn = false;
        if (!did_warn) {
          LOG(ERROR) << "Shader artifacts and GENERATE_COMPILE_STATS logging enabled. "
                        "Skipping shader artifact save";
          did_warn = true;
        }
#else
        auto basename = library_item_to_filename(builder.root_item_);
        if (ShaderArtifactTypeBits::kBuilder & artifacts_to_save) {
          serializeBuilder(builder, basename);
        }

        write_spirv_artifacts(builder.processed_code_,
                              spirv,
                              opt_spirv,
                              reflection,
                              basename,
                              artifacts_to_save);
#endif
      }
    }

    // Check if the spirv is empty. We do this last so we can save any useful artifacts
    // for debugging first.
    if (spirv.empty()) {
      // CompileResult should contain useful error log info, but handle the empty case
      // too. A failure anywhere in the material stops every stage of it, so the reason
      // may belong to a stage other than this one.
      std::string error_message = compile_result.second;
      if (error_message.empty()) {
        error_message = compile_results.material_error;
      }
      if (error_message.empty()) {
        error_message = "Error generating Spir-V for \"" + pretty_name + "\"";
      }
      RUNTIME_EX_ASSERT(false, error_message);
    }

#if 0
  // print cache size
  std::cout << "Cache size:\n";
  std::cout << "  spirv:    " << (sizeof(uint32_t) * builder.cache_.spirv.size()) << "\n";
  std::cout << "  glsl:     " << builder.cache_.glsl.size() << "\n";
#endif

    caches.push_back(std::make_unique<ShaderCache>(
        std::move(spirv),
        std::move(builder.processed_code_),
        std::move(reflection),
        builder.shader_stage_,
        builder.entry_point_.empty() ? "main" : std::move(builder.entry_point_),
        library_item_to_filename(builder.root_item_),
        std::move(builder.external_uniform_buffer_names_),
        builder.raytracing_hit_group_index_));
  }

  return caches;
}

namespace {
// Creates any sub-directories for saving artifacts to, and returns the full path.
// Returns an empty string if there is a problem creating the path.
std::string create_artifact_basename_from_cache(const ShaderCache& artifact_shader_cache,
                                                const std::string& sub_directory) {
  auto basename = artifact_shader_cache.getLibraryItemFilename();

  // Handle sub_directory
  if (!sub_directory.empty()) {
    if (create_artifact_subdir(sub_directory).empty()) {
      return std::string();
    }
    basename = sub_directory + '/' + basename;
  }
  return basename;
}
}  // namespace

void ShaderManager::saveArtifacts(const Builder& builder,
                                  const std::string& sub_directory,
                                  ShaderArtifactTypeBits type) const {
  // Clone builder and make it a cache
  if (can_save_artifacts_) {
    auto artifact_shader_cache = createCache(cloneBuilder(builder));
    auto const basename =
        create_artifact_basename_from_cache(*artifact_shader_cache, sub_directory);

    if (!basename.empty()) {
      serializeBuilder(builder, basename);
      write_spirv_artifacts(artifact_shader_cache->getGlsl(),
                            artifact_shader_cache->getSpirv(),
                            spirv_t(),
                            artifact_shader_cache->getReflection(),
                            basename,
                            type);
    }
  }
}

void ShaderManager::saveArtifacts(const ShaderCache& artifact_shader_cache,
                                  const std::string& sub_directory,
                                  ShaderArtifactTypeBits type) const {
  if (can_save_artifacts_) {
    auto const basename =
        create_artifact_basename_from_cache(artifact_shader_cache, sub_directory);

    if (!basename.empty()) {
      write_spirv_artifacts(artifact_shader_cache.getGlsl(),
                            artifact_shader_cache.getSpirv(),
                            spirv_t(),
                            artifact_shader_cache.getReflection(),
                            basename,
                            type);
    }
  }
}

ShaderCacheShPtrVector ShaderManager::createCacheVector(BuilderUqPtrVector&& builders,
                                                        bool save_artifacts) const {
  // construct a shader string from all the builder template names
  std::string shader_name;
  for (auto const& builder : builders) {
    if (shader_name.size()) {
      shader_name += "+";
    }
    shader_name += builder->getTemplateName();
  }

  // decorate this vector of shaders for use by a single material
  ShaderRedecorator shader_redecorator(shader_name);

  std::unordered_map<std::string, std::pair<ShaderReflection::ItemInfo, std::string>>
      ubo_duplicates_map, ssbo_duplicates_map;

#if GENERATE_COMPILE_STATS
  auto start_time = timer_start();
#endif
  // now compile the builders to caches, all of the material's stages together
  auto caches = buildSpirv(builders, &shader_redecorator, save_artifacts);

  for (size_t stage_index = 0; stage_index < builders.size(); ++stage_index) {
    // reject duplicate UBO or SSBO attr names across shader stages
    // unless their buffer is shared (ItemInfo is identical)
    auto const& template_name = builders[stage_index]->getTemplateName();
    auto const& reflection = caches[stage_index]->getReflection();
    auto cache_ubo_attr_names = reflection.getAllUniformBufferAttrNames();
    auto cache_ssbo_attr_names = reflection.getAllShaderStorageBufferAttrNames();
    for (auto const& name : cache_ubo_attr_names) {
      auto const& item_info = reflection.getUniformBufferAttrItemInfo(name);
      auto const [itr, emplaced] = ubo_duplicates_map.try_emplace(
          std::string(name), std::make_pair(item_info, template_name));
      if (!emplaced) {
        auto const& existing_item_info = itr->second.first;
        CHECK(existing_item_info == item_info)
            << "Duplicate non-shared UBO attr name '" << name << "' in templates '"
            << template_name << "' and '" << itr->second.second << "'";
      }
    }
    for (auto const& name : cache_ssbo_attr_names) {
      auto const& item_info = reflection.getShaderStorageBufferAttrItemInfo(name);
      auto const [itr, emplaced] = ssbo_duplicates_map.try_emplace(
          std::string(name), std::make_pair(item_info, template_name));
      if (!emplaced) {
        auto const& existing_item_info = itr->second.first;
        CHECK(existing_item_info == item_info)
            << "Duplicate non-shared SSBO attr name '" << name << "' in templates '"
            << template_name << "' and '" << itr->second.second << "'";
      }
    }
  }
#if GENERATE_COMPILE_STATS
  auto cache_time = timer_stop_microseconds(start_time);
  g_stats_reporter.addBuildCacheStats(shader_name, cache_time);
#endif

  return caches;
}

ShaderCacheShPtr ShaderManager::createCache(BuilderUqPtr&& builder,
                                            bool save_artifacts) const {
#if GENERATE_COMPILE_STATS
  auto start_time = timer_start();
  auto shader_name = builder->getTemplateName();
#endif

  // now compile from a builder to a cache. A lone builder is still a whole material as
  // far as the compiler is concerned, so it goes through the same path.
  BuilderUqPtrVector builders;
  builders.push_back(std::move(builder));
  auto caches = buildSpirv(builders, nullptr, save_artifacts);
  CHECK_EQ(caches.size(), 1u);
  auto rtn = std::move(caches.front());

#if GENERATE_COMPILE_STATS
  auto cache_time = timer_stop_microseconds(start_time);
  g_stats_reporter.addBuildCacheStats(shader_name, cache_time);
#endif
  return rtn;
}

ShaderCacheShPtrVector ShaderManager::createCacheVectorFromTemplate(
    const std::vector<Builder::ConstructionArgs>& args_vector,
    bool save_artifacts) const {
  return createCacheVector(createBuilderVector(args_vector), save_artifacts);
}

ShaderManager::BuilderUqPtr ShaderManager::createBuilder(
    const std::string& template_name,
    const Builder::Requirements requirements,
    const std::string& entry_point) const {
  return std::make_unique<Builder>(
      *library_, library_->get(template_name), requirements, entry_point);
}

ShaderManager::BuilderUqPtrVector ShaderManager::createBuilderVector(
    const std::vector<Builder::ConstructionArgs>& builderArgs) const {
  BuilderUqPtrVector rtn;
  for (const auto& args : builderArgs) {
    rtn.push_back(createBuilder(args.template_name, args.requirements, args.entry_point));
  }
  return rtn;
}

ShaderManager::BuilderUqPtr ShaderManager::cloneBuilder(const Builder& src) const {
  auto clone = std::make_unique<Builder>(*library_,
                                         library_->get(src.item_indices_[0]),
                                         src.requirements_,
                                         src.entry_point_);
  clone->item_indices_ = src.item_indices_;
  clone->op_list_ = src.op_list_;
  clone->subroutine_map_ = src.subroutine_map_;
  // Save two string copies if they'd just be ignored anyway
  if (src.is_processed_) {
    clone->is_processed_ = src.is_processed_;
    clone->processed_code_ = src.processed_code_;
  }
  clone->external_uniform_buffer_names_ = src.external_uniform_buffer_names_;
  clone->raytracing_hit_group_index_ = src.raytracing_hit_group_index_;

  return clone;
}

ShaderManager::BuilderUqPtrVector ShaderManager::cloneBuilderVector(
    const BuilderUqPtrVector& src_vector) const {
  BuilderUqPtrVector rtn;
  for (const auto& src : src_vector) {
    rtn.push_back(cloneBuilder(*(src.get())));
  }
  return rtn;
}

}  // namespace gfx
