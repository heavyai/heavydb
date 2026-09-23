/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

#include <boost/noncopyable.hpp>

#include "GfxDriver/ShaderCompiler/Library.h"
#include "GfxDriver/ShaderCompiler/ShaderCache.h"
#include "GfxDriver/ShaderCompiler/ShaderRedecorator.h"
#include "GfxDriver/TypeGLSL.h"
#include "Shared/EnumBitmaskOps.h"

namespace gfx {

// TODO(scb): shader cache to speed startup. This will require hashing the input
// shader and possibly even higher level inputs storing the hash with the compiled shader
// blob.

class ShaderManager : boost::noncopyable {
  using ProcessResult = std::string;

 public:
  explicit ShaderManager(LibraryUqPtr library);
  ShaderManager() = delete;
  ~ShaderManager();

  //! Get the current library
  gfx::Library& getLibrary() const;

  //! Destroy current library and replace with a new one (hot loading)
  void replaceLibrary(LibraryUqPtr library);

  //! Retrieve a raw template string from the shader library
  const std::string& getTemplate(const std::string& internal_path) const;

  /**
   * Builder class records all operations internally, to be processed later by
   * ShaderManager to build a Spirv binary. Internally it maintains a cache of generated
   * results once the spirv has been built. This cache will be automatically invalidated
   * and regenerated if operations are added.
   *
   * The Builder does not retain any external state (it is self contained), so it is safe
   * to retain a Builder for reuse later, or clone one via ShaderManager::cloneBuilder for
   * additional operations.
   *
   * Operation recording is a forward only process, there is currently no way to reset
   * a Builder or remove operations. If operation removal is desired, this must be added
   * as a specific Operator internally so a serialized Builder accurately records the
   * process.
   * */
  class Builder {
   public:
    enum class Requirements { kNothing = 0, kSaveArtifacts = 1 << 0 };
    using NameMap = std::vector<std::pair<std::string, std::string>>;

    struct ConstructionArgs {
      std::string template_name;
      Requirements requirements;
      std::string entry_point;
      ConstructionArgs(std::string template_name,
                       Requirements requirements = Requirements::kNothing,
                       std::string entry_point = std::string())
          : template_name{std::move(template_name)}
          , requirements{requirements}
          , entry_point{std::move(entry_point)} {}
      ConstructionArgs() = delete;
    };

    enum class OpType {
      kAddPreamble,
      kReplaceFirstTag,
      kReplaceAllTags,
      kReplaceAll,
      kReplaceAllMultiple,
      kPrependTemplate,
      kAppendTemplate,
      kReplaceTemplate,
      kInsertBeforeFunc,
      kReplaceFuncCall,
      kReplaceFuncDef,
      kReplaceWithSubBuilder,
      kAppendSubBuilder
    };

    explicit Builder(const Library& library,
                     const Library::Item& library_item,
                     Requirements req,
                     const std::string& entry_point);

    ~Builder();

    // Get the template name this was created from
    const std::string& getTemplateName() const;

    // Get the pipeline stage being built
    ShaderStage getShaderStage() const;

    // Get the entry point. An empty string indicates 'main'
    const std::string& getEntryPoint() const;

    // Add tag delimiters to a string (utility function, does not retain string)
    // e.g. "useUniform" would become "<useUniform>"
    // DEPRECATED - Do not add to anything new!
    std::string makeTag(const std::string& name) const;

    // Preamble string control (included immediately after #version tag)
    void addPreambleString(std::string str);

    //
    // Exact substitutions
    // Currently used by property writers, which build their own tags.
    // TODO(scb): Tag delimiters should be set by Builder, so we'll need
    // to revisit interactions between the builder and the prop writers
    void replaceAll(std::string search_str, std::string new_str);

    void replaceAllMultiple(NameMap&& name_map);
    //
    // Tag marker substitutions.
    // Tags are built automatically from the input string so passing "foo" for tag_str
    // will cause a search for "<foo>"
    //
    // replace the first occurrence of the given `tag_str` with `new_str`
    void replaceFirstTag(std::string tag_str, std::string new_str);

    // replace all occurrences of the given `tag_str` with `new_str`
    void replaceAllTags(std::string tag_str, std::string new_str);

    // Add secondary template as standalone string
    // TODO(scb): Only used by AccumTx extents/stddev. Reorganize afer we drop GLSL
    void prependTemplate(std::string template_name);

    // Append a secondary template to the as standalone string
    void appendTemplate(std::string template_name);

    // Replace the given marker with the template. This is only used by the pure GLSL
    // path for color conversion code injection.
    // TODO(scb): Remove once we go full Spirv
    void replaceTagWithTemplate(std::string tag_str, std::string template_name);

    //
    // Higher level manipulations.
    // These methods perform more complex operations, either building search tags and
    // substitutions automatically, or performing parse tree changes such as replacing
    // function components. In some cases lexical assumptions are made about the
    // the shaders, such as "useU" + property name indicating a uniform value.
    //
    // Builds a tag "<useU'name'>" and replaces occurrences with 0 or 1 to indicate
    // to the shader if the value is a uniform or not
    void setAttributeType(const std::string& name, bool is_uniform);

    // Replace "Type" and "Enum" tags appropriately for both the input and output
    // properties of a Scale.
    void setPropertyInOutTypes(const std::string& prop_name,
                               const BaseTypeGLSL& in_type,
                               const BaseTypeGLSL& out_type);

    // Replace "<useU'name'>" tags with 0 or 1 to indicate if a the value is a uniform
    // Replace "Type" and "Enum" tags for input and output properties to in_out_type
    void setAllPropertyTypes(const std::string& name,
                             bool is_uniform,
                             BufferAttrType in_out_type);

    // Insert "new_str" before function declaration "func_decl". (eg) Used to inject the
    // projection UBO above the projection functions.
    void insertBeforeFunction(std::string search_str, std::string new_str);

    // Replace all instances of the function call (string includes parameters)
    // (eg) used by accumulation rendering to replace 'getDensityColor(pct)' with
    // a call into a Scale function.
    void replaceFunctionCall(std::string orig_sig, std::string new_sig);

    // Replace a function definition with an entirely new string. This is typically a new
    // function and other supporting components. This could include additional functions,
    // UBOs, etc. This is mainly used to inject a Scale into a Mark.
    void replaceFunctionDefinition(std::string func_name,
                                   std::string new_body,
                                   bool is_required);

    // Exactly the same as replaceFunctionDefinition, except takes a Builder instead of a
    // string. This builder is then processed to create the new function body, and the
    // specified function definition is replaced with it.
    void replaceFunctionWithSubBuilder(std::string func_name,
                                       std::shared_ptr<Builder> sub_builder,
                                       bool is_required);

    // Append a sub-builder to allow appending a dynamic template with internal operators
    void appendSubBuilder(std::shared_ptr<Builder> sub_builder);

    // Add a subroutine binding.
    // Required bindings will trigger an error if the subroutine is not found in the
    // shader. Subroutine bindings are resolved after the operator list is processed and
    // all substitutions and injections have been performed. Substitution happens after
    // parsing of the glsl but before creation of the Spirv binary (by manipulating the
    // call graph in the parse tree).
    void addSubroutineBinding(std::string call, std::string target, bool is_required);

    // Set external uniform buffers
    // The set of names will be stored in the Builder and passed through to the
    // Material which will then assume that uniform buffers of those names are
    // "external", the same as SSBOs, and not to be created and managed locally.

    // Replace any existing names with a new set
    void setExternalUniformBuffers(
        std::set<std::string_view>&& external_uniform_buffer_names);

    // Append a new external UBO name to the set, checking for duplicates
    void addExternalUniformBuffer(std::string_view external_uniform_buffer_name);

    // Set the hit-group index for use in RaytracingPipeline shader grouping
    void setRaytracingHitGroupIndex(uint32_t index);

   private:
    const Library& library_;
    // Be sure to update ShaderManager::clone() when adding stashes here
    const Library::Item& root_item_;
    std::vector<uint32_t> item_indices_;
    ShaderStage shader_stage_;
    Requirements requirements_;
    std::string entry_point_;
    std::set<std::string_view> external_uniform_buffer_names_;
    uint32_t raytracing_hit_group_index_;
    // Cached GLSL code after processing operators
    ShaderManager::ProcessResult processed_code_;
    // Is processed_code_ valid?
    bool is_processed_ = false;

    struct Operator {
      OpType type;
      std::string str1;
      std::string str2;
      std::shared_ptr<Builder> sub_builder;
      NameMap name_map;
      bool is_required;
      explicit Operator(OpType type,
                        std::string str1,
                        std::string str2,
                        std::shared_ptr<Builder> sub_builder,
                        bool is_required);

      explicit Operator(OpType type, NameMap&& name_map);
    };

    void addOperator(OpType type,
                     std::string str1 = std::string(),
                     std::string str2 = std::string(),
                     std::shared_ptr<Builder> sub_builder = nullptr,
                     bool is_required = false);

    void addOperator(OpType type, NameMap&& name_mapping);

    std::vector<Operator> op_list_;
    SubroutineMap subroutine_map_;

    friend class ::gfx::ShaderManager;
  };

  using BuilderUqPtr = std::unique_ptr<ShaderManager::Builder>;
  using BuilderUqPtrVector = std::vector<BuilderUqPtr>;
  using BuilderShPtr = std::shared_ptr<ShaderManager::Builder>;

  BuilderUqPtr createBuilder(
      const std::string& template_name,
      const Builder::Requirements requirements = Builder::Requirements::kNothing,
      const std::string& entry_point = std::string()) const;

  BuilderUqPtrVector createBuilderVector(
      const std::vector<Builder::ConstructionArgs>& args_vector) const;

  BuilderUqPtr cloneBuilder(const Builder& src) const;
  BuilderUqPtrVector cloneBuilderVector(const BuilderUqPtrVector& src) const;

  // Process operators, invoke glslang, and return a spirv binary suitable for final
  // specialization.
  ShaderCacheShPtrVector createCacheVector(BuilderUqPtrVector&& builders,
                                           bool save_artifacts = false) const;

  // Like createCacheVector, but operates on only one builder, and returns one cache.
  // NOTE: this explicitly does _not_ redecorate the shaders.
  ShaderCacheShPtr createCache(BuilderUqPtr&& builder, bool save_artifacts = false) const;

  // Takes the same args as `createBuilderVector()`, but immediately turns the Builder
  // into a ShaderCache vector and returns that, rather than returning the builder.
  ShaderCacheShPtrVector createCacheVectorFromTemplate(
      const std::vector<Builder::ConstructionArgs>& args_vector,
      bool save_artifacts = false) const;

  // Saves all available artifacts for a processed Builder.
  // If shader compilation fails, this will be called to save all available artifacts
  // This can also be called by QueryRendererContext if the Vega usermeta property
  // specifies saving artifact data (usually for tests).
  // sub_directory will be created within the global artifact path location (if it exists)
  void saveArtifacts(const Builder& builder,
                     const std::string& sub_directory = std::string(),
                     ShaderArtifactTypeBits type = ShaderArtifactTypeBits::kAll) const;
  void saveArtifacts(const ShaderCache& artifact_shader_cache,
                     const std::string& sub_directory = std::string(),
                     ShaderArtifactTypeBits type = ShaderArtifactTypeBits::kAll) const;

  // Deserialization (used by test fixtures).
  BuilderUqPtr deserializeBuilder(const std::string& serialized_json);

 private:
  LibraryUqPtr library_;

  bool can_save_artifacts_;
  ShaderArtifactTypeBits always_save_artifacts_;
  std::string artifact_path_;

  GlslangWrapperUqPtr glslang_wrapper_;

  void processOperators(Builder& builder,
                        SubroutineMap* rebind_map,
                        bool is_sub_builder) const;

  std::string buildExtensionAndIncludesString(Builder& builder) const;

  // Process operators, invoke glslang, and return a spirv binary suitable for final
  // specialization
  std::unique_ptr<ShaderCache> buildSpirv(Builder& builder,
                                          ShaderRedecorator* shader_redecorator,
                                          bool save_artifacts = false) const;

  struct Serializer;
  void serializeBuilder(const Builder& builder, const std::string& filename) const;
};

}  // namespace gfx

ENABLE_BITMASK_OPS(gfx::ShaderManager::Builder::Requirements);
