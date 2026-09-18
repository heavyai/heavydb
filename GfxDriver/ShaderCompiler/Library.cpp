/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/Library.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/schema.h>
#include <rapidjson/stringbuffer.h>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/ShaderCompiler/JSONSchemas.h"

static constexpr bool KLogDictionaries = false;

namespace gfx {

namespace {
static std::unordered_map<std::string, Library::TemplateType> class_name_to_type_map = {
    {"vertex", Library::TemplateType::kVertex},
    {"fragment", Library::TemplateType::kFragment},
    {"geometry", Library::TemplateType::kGeometry},
    {"tess_control", Library::TemplateType::kTessControl},
    {"tess_eval", Library::TemplateType::kTessEval},
    {"compute", Library::TemplateType::kCompute},
    {"ray_gen", Library::TemplateType::kRayGen},
    {"ray_closest_hit", Library::TemplateType::kClosestHit},
    {"ray_miss", Library::TemplateType::kMiss},
    {"ray_intersection", Library::TemplateType::kIntersection},
    {"ray_callable", Library::TemplateType::kCallable},
    {"mesh", Library::TemplateType::kMesh},
    {"task", Library::TemplateType::kTask},
    {"glsl", Library::TemplateType::kGlsl},
};

static std::string read_file(const std::string& filename) {
  std::ifstream file(filename);

  RUNTIME_EX_ASSERT(file.is_open(), "Library failed to open file: " + filename + ".");

  std::string buffer;
  file.seekg(0, std::ios::end);
  buffer.reserve(file.tellg());
  file.seekg(0, std::ios::beg);
  buffer.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  return buffer;
}
}  // namespace

Library::Library() : are_dictionaries_valid_{false} {}

void Library::reset() {
  template_map_.clear();
}

void Library::addFromManifestFile(const std::string& filename, const std::string& path) {
  using namespace rapidjson;

  LOG(INFO) << "Initializing shader library using manifest: \'" << path + filename << ".";
  std::string json_string = read_file(path + filename);
  RUNTIME_EX_ASSERT(!json_string.empty(), "Shader manifest json string is empty.");

  // Initial parsing
  Document manifest;
  if (manifest.Parse(json_string.c_str()).HasParseError()) {
    // Not a valid JSON
    std::string err(GetParseError_En(manifest.GetParseError()));
    THROW_RUNTIME_EX("Error parsing shader manifest: " + err);
  }

  // Prepare schema validator
  Document sd;
  if (sd.Parse(get_shader_manifest_schema().c_str()).HasParseError()) {
    std::string err(GetParseError_En(sd.GetParseError()));
    LOG(ERROR) << "Invalid shader manifest schema: " + err;
  }
  SchemaDocument schema(sd);
  SchemaValidator validator(schema);

  Document doc;
  if (doc.Parse(json_string.c_str()).HasParseError()) {
    std::string err(GetParseError_En(doc.GetParseError()));
    // Not a valid JSON
    THROW_RUNTIME_EX("Error parsing builder serialization file: " + err);
  }

  // Validate against schema
  if (!doc.Accept(validator)) {
    // Invalid according to schema, build error string.
    StringBuffer sb;
    validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
    std::string err("Schema validation failed: " + std::string(sb.GetString()) +
                    "\nInvalid keyword: " + validator.GetInvalidSchemaKeyword());
    sb.Clear();
    validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
    err += std::string("\nDocument pointer: ") + sb.GetString();
    THROW_RUNTIME_EX(err);
  }

  Value& sources = manifest["sources"];
  if (sources.IsArray()) {
    for (auto& source : sources.GetArray()) {
      Value& name = source["filename"];
      Value& file_path = source["file_path"];
      Value& internal_path = source["internal_path"];
      Value& class_name = source["class"];
      Value& language = source["language"];

      std::string internal_name(std::string(internal_path.GetString()) +
                                name.GetString());
      RUNTIME_EX_ASSERT(!contains(internal_name),
                        "Shader already exists in library: " + internal_name);

      std::string external_file(path + file_path.GetString() + "/" + name.GetString());
      std::string code = read_file(external_file);

      RUNTIME_EX_ASSERT(!code.empty(),
                        "Error reading shader file: \"" + external_file + "\".");

      add(internal_name, class_name.GetString(), language.GetString(), code);
    }
  }
}

namespace {
// return pair of
//  substring of s between delimiters
//  string offset to just after stop_delim (to elide stop_delim)
std::pair<std::string, std::string::size_type> get_substring(
    const std::string& s,
    const std::string& start_delim,
    const std::string& stop_delim,
    std::string::size_type start_offset = 0) {
  auto first_delim_pos = s.find(start_delim, start_offset);
  if (first_delim_pos == std::string::npos) {
    return {std::string(), std::string::npos};
  }
  auto end_pos_of_first_delim = first_delim_pos + start_delim.length();
  auto last_delim_pos = s.find(stop_delim, end_pos_of_first_delim);
  CHECK_NE(last_delim_pos, std::string::npos)
      << "Failed to find stop delimiter \"" << stop_delim << "\"";

  return {s.substr(end_pos_of_first_delim, last_delim_pos - end_pos_of_first_delim),
          last_delim_pos + stop_delim.length()};
}

// return vector of all strings in s between delimiters
std::vector<std::string> get_all_substrings(const std::string& s,
                                            const std::string& start_delim,
                                            const std::string& end_delim) {
  std::string::size_type start_pos = 0;
  std::vector<std::string> strings;
  while (start_pos != std::string::npos) {
    auto [new_str, new_pos] = get_substring(s, start_delim, end_delim, start_pos);
    if (!new_str.empty()) {
      strings.push_back(std::move(new_str));
    }
    start_pos = new_pos;
  }
  return strings;
}
}  // namespace

void Library::Dictionary::update(const Library* library,
                                 const std::vector<Item*>& items,
                                 std::vector<string_size_t>& code_offsets,
                                 const std::string& block_begin_delim,
                                 const std::string& block_end_delim,
                                 const std::string& element_begin_delim,
                                 const std::string& element_end_delim,
                                 TransformCB transform_cb,
                                 WriteIndexCB write_cb) {
  // Map string to dictionary index
  std::unordered_map<std::string, uint32_t> dict_map;
  // Seed dict_map with the existing dictionary vector
  for (size_t i = 0; i < dictionary_.size(); ++i) {
    dict_map[dictionary_[i].str] = i;
  }

  // Loop over library items extracting blocks and elements
  for (auto* item : items) {
    if (!item->is_initialized) {
      // Extract the block substring containing the elements to scan
      auto [block_string, delim_end] =
          get_substring(item->code, block_begin_delim, block_end_delim);
      if (!block_string.empty()) {
        // Track the offset into the Item::code string to elide the extracted
        // substring from the template
        code_offsets[item->index] = std::max(delim_end, code_offsets[item->index]);

        // Extract individual strings (single extension or include names in a vector)
        auto dict_strings =
            get_all_substrings(block_string, element_begin_delim, element_end_delim);
        CHECK(!dict_strings.empty())
            << "Empty block encountered building Library::Dictionary";
        // Generate dictionary_, moving strings into Entry
        for (auto& str : dict_strings) {
          // Get an iterator to the entry in the temporary dict map
          auto [itr, did_insert] = dict_map.try_emplace(str, dictionary_.size());
          if (did_insert) {
            // New string found, add it to the dictionary
            auto& entry = dictionary_.emplace_back(
                std::move(str), library ? &library->get(str) : nullptr);
            transform_cb(entry);
          }
          // Add the index for the Entry to the Library::Item
          write_cb(item, itr->second);
        }
      }
    }
  }
}

const Library::Dictionary::Entry& Library::Dictionary::get(uint32_t index) const {
  CHECK_LT(index, dictionary_.size());
  return dictionary_[index];
}

bool Library::Dictionary::isEmpty() const {
  return dictionary_.empty();
}

std::ostream& operator<<(std::ostream& os, const gfx::Library::Dictionary& d) {
  int i = 0;
  for (auto const& entry : d.dictionary_) {
    os << i++ << "  " << entry.str;
  }
  return os;
}

void Library::updateDictionaries() {
  // Build extension and include dictionaries
  std::vector<string_size_t> offsets(item_vector_.size(), 0);
  extension_dict_.update(
      nullptr,
      item_vector_,
      offsets,
      "//<extensions>",
      "//</extensions>",
      "#extension ",
      ":",
      [](Dictionary::Entry& entry) {
        entry.str = std::string("#extension " + entry.str + " : require\n");
      },
      [](Item* item, uint32_t index) { item->extension_indices.push_back(index); });

  include_dict_.update(
      this,
      item_vector_,
      offsets,
      "//<includes>",
      "//</includes>",
      "\"",
      "\"",
      [](Dictionary::Entry& entry) {
        entry.str = std::string("#include \"" + entry.str + "\"\n");
      },
      [](Item* item, uint32_t index) { item->include_indices.push_back(index); });

  // Trim extensions and includes from the code string
  for (auto* item : item_vector_) {
    if (!item->is_initialized && offsets[item->index] > 0) {
      item->code = item->code.substr(offsets[item->index]);
    }
    item->is_initialized = true;
  }

  are_dictionaries_valid_ = true;

  if constexpr (KLogDictionaries) {
    if (!extension_dict_.isEmpty()) {
      std::cout << "Library extensions dictonary -----\n" << extension_dict_ << std::endl;
    }
    if (!include_dict_.isEmpty()) {
      std::cout << "Library includes dictonary -----\n" << include_dict_ << std::endl;
    }
  }
}

void Library::add(const std::string& internal_name,
                  const std::string& class_name,
                  const std::string& language,
                  const std::string& code) {
  RUNTIME_EX_ASSERT(!contains(internal_name),
                    "shader already exists in library: " + internal_name);

  // embed the internal name in the item (useful for serialization of Builders)
  auto& item = template_map_[internal_name] = {
      internal_name, class_name_to_type_map[class_name], language, code};

  // Assign vector lookup index
  item.index = item_vector_.size();
  item_vector_.emplace_back(&item);
  are_dictionaries_valid_ = false;
}

bool Library::contains(const std::string& filename) const {
  return template_map_.find(filename) != template_map_.end();
}

const Library::Item& Library::get(const std::string& filename) const {
  auto itr = template_map_.find(filename);
  RUNTIME_EX_ASSERT(itr != template_map_.cend(),
                    "Failed to find shader " + filename + " in library.");
  return itr->second;
}

const Library::Item& Library::get(uint32_t index) const {
  CHECK_LT(index, item_vector_.size());
  return *item_vector_[index];
}

const Library::Dictionary::Entry& Library::getExtension(uint32_t index) {
  CHECK(are_dictionaries_valid_);
  return extension_dict_.get(index);
}

const Library::Dictionary::Entry& Library::getInclude(uint32_t index) {
  CHECK(are_dictionaries_valid_);
  return include_dict_.get(index);
}

}  // namespace gfx
