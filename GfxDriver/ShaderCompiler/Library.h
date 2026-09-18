/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

// TODO(scb): distinguish template strings (which we want to copy and manipulate)
// versus explicit strings like passthru shaders which we can just compile and stash

class Library {
 public:
  enum TemplateType {
    kVertex,
    kFragment,
    kGeometry,
    kTessControl,
    kTessEval,
    kCompute,
    kRayGen,
    kAnyHit,
    kClosestHit,
    kMiss,
    kIntersection,
    kCallable,
    kMesh,
    kTask,
    kGlsl
  };

  using LanguageType = std::string;
  using DictionaryIndices = std::vector<uint32_t>;
  using string_size_t = std::string::size_type;

  struct Item {
    std::string internal_name;
    TemplateType template_type{};
    LanguageType language_type;
    std::string code;
    uint32_t index{0};
    DictionaryIndices extension_indices;
    DictionaryIndices include_indices;
    bool is_initialized{false};
  };

  // Dictionary class
  // Builds a look up table for delimited substrings for all Library::Items
  class Dictionary {
   public:
    struct Entry {
      std::string str;   // string to use
      const Item* item;  // Library::Item referenced for recursive lookups
      explicit Entry(std::string&& str, const Item* item) : str{str}, item{item} {}
    };

    // Callback function for transforming the entry into its complete form that can
    // inserted directly into a final shader string
    using TransformCB = std::function<void(Entry&)>;

    // Callback function for writing the index into the Library::Item
    // (Library::Item stores vectors of extension and include strings,
    // the callback handles writing to the correct vector)
    using WriteIndexCB = std::function<void(Item*, uint32_t)>;

    // Update the dictionary
    void update(const Library* library,           // library for looking up includes
                const std::vector<Item*>& items,  // items to process
                std::vector<string_size_t>& code_offsets,  // final code offsets per item
                const std::string& block_begin_delim,      // code block start delimiter
                const std::string& block_end_delim,        // code block end delimiter
                const std::string& element_begin_delim,  // single element start delimiter
                const std::string& element_end_delim,    // single element end delimiter
                TransformCB transform_cb,  // callback to modify entry on addition
                WriteIndexCB write_cb);    // callback to write dictionary index into Item

    // Get the Dictionary::Entry for index
    const Entry& get(uint32_t index) const;

    bool isEmpty() const;
    friend std::ostream& operator<<(std::ostream& os, const Dictionary& d);

   private:
    std::vector<Entry> dictionary_;
  };

  Library();
  ~Library() = default;

  // Lock the library and generate include and extension maps
  void updateDictionaries();

  // Clear the library
  void reset();

  // Load all the shaders referenced in a JSON manifest. Does not clear the map.
  void addFromManifestFile(const std::string& filename, const std::string& path);

  // Add a single shader source string and internal path key
  void add(const std::string& internal_path,
           const std::string& class_name,
           const std::string& language,
           const std::string& source);

  // Check if the library contains a particular shader string
  bool contains(const std::string& internal_path) const;

  // Retrieve item given an internal library path
  const Item& get(const std::string& internal_path) const;

  // Retrieve item with an item index
  const Item& get(uint32_t index) const;

  const Dictionary::Entry& getExtension(uint32_t index);
  const Dictionary::Entry& getInclude(uint32_t index);

  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;

 private:
  using TemplateMap = std::unordered_map<std::string, Item>;
  TemplateMap template_map_;
  std::vector<Item*> item_vector_;

  // Dictionaries
  Dictionary extension_dict_;
  Dictionary include_dict_;

  bool are_dictionaries_valid_;
};

}  // namespace gfx
