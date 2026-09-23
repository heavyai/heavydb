/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/SpirvArtifacts.h"

#include <fstream>
#include <functional>

#include <glslang/SPIRV/disassemble.h>

#include <spirv_cross/spirv_glsl.hpp>
#include <spirv_cross/spirv_parser.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "GfxDriver/ShaderCompiler/SpirvCrossUtils.h"
#include "Logger/Logger.h"

namespace filesys = boost::filesystem;

namespace gfx {

constexpr char artifact_path_env_var[] = "OMNISCI_shader_artifact_path";
constexpr char always_save_artifacts_env_var[] = "OMNISCI_always_save_shader_artifacts";

std::string get_artifact_pathname() {
  auto env = getenv(artifact_path_env_var);
  if (env != nullptr) {
    filesys::path p(env);
    p.remove_trailing_separator();
    // check existence and make sure it's a directory
    if (filesys::exists(p) && filesys::is_directory(p)) {
      // Add a preferred separator so string is ready for filename appends
      return p.string() + filesys::path::preferred_separator;
    } else if (shader_artifacts_enabled_in_build()) {
      LOG(WARNING) << "\"" << artifact_path_env_var << "\""
                   << " environment variable is set but points to an invalid location.";
    }
  }
  return std::string();
}

std::string create_artifact_subdir(const std::string& subdir) {
  std::string pname = get_artifact_pathname();
  if (pname.empty()) {
    LOG(WARNING)
        << "Unable to create shader artifact subdirectory, artifact path is invalid";
  } else {
    filesys::path p(pname);
    p /= subdir;

    if (!filesys::exists(p)) {
      // only allow single directory level creation
      filesys::create_directory(p);
    }

    // final sanity check
    if (filesys::exists(p) && filesys::is_directory(p)) {
      return p.string() + filesys::path::preferred_separator;
    }
  }
  return std::string();
}

static std::unordered_map<std::string, ShaderArtifactTypeBits> string_to_artifact_map = {
    {"none", ShaderArtifactTypeBits::kNone},
    {"1", ShaderArtifactTypeBits::kAll},
    {"true", ShaderArtifactTypeBits::kAll},
    {"all", ShaderArtifactTypeBits::kAll},
    {"builder", ShaderArtifactTypeBits::kBuilder},
    {"glsl", ShaderArtifactTypeBits::kGlsl},
    {"spirv", ShaderArtifactTypeBits::kSpvBin},
    {"spirv-dis", ShaderArtifactTypeBits::kSpvDis},
    {"spirv-glsl", ShaderArtifactTypeBits::kSpvGlsl},
    {"reflect", ShaderArtifactTypeBits::kSpvReflect}};

ShaderArtifactTypeBits string_to_shader_artifact_type(std::string s) {
  boost::algorithm::to_lower(s);
  auto const itr = string_to_artifact_map.find(s);
  if (itr != string_to_artifact_map.end()) {
    return itr->second;
  }
  return ShaderArtifactTypeBits::kNone;
}

ShaderArtifactTypeBits is_always_save_artifacts_enabled() {
  auto env = getenv(always_save_artifacts_env_var);
  if (env != nullptr) {
    return string_to_shader_artifact_type(env);
  }
  return ShaderArtifactTypeBits::kNone;
}

void write_to_file(const std::string& filename,
                   bool is_binary,
                   std::function<void(std::ostream&)> write) {
  std::ios_base::openmode mode =
      std::ios::out |
      (is_binary ? std::ios::binary : static_cast<std::ios_base::openmode>(0));
  std::fstream file(filename, mode);
  if (file) {
    write(file);
  } else {
    LOG(WARNING) << "Failed to open shader artifact file \'" + filename + "\'";
  }
}

void write_spirv_artifacts(const std::string& glsl_string,
                           const spirv_t& spv,
                           const spirv_t& opt_spv,
                           const std::string& base_name,
                           ShaderArtifactTypeBits artifacts) {
  if (shader_artifacts_enabled_in_build()) {
    std::string path = get_artifact_pathname();
    if (path.empty()) {
      LOG(WARNING) << "Unable to write spir-v artifacts, path is empty";
      return;
    }

    std::string pathed_name = path + base_name;

    if ((ShaderArtifactTypeBits::kGlsl & artifacts) && !glsl_string.empty()) {
      std::string filename(pathed_name + ".glsl");
      write_to_file(filename, false, [&](auto& file) { file << glsl_string; });
    }

    if (!spv.empty()) {
      if (ShaderArtifactTypeBits::kSpvBin & artifacts) {
        // unoptimized spirv
        std::string filename = pathed_name + ".spv";
        write_to_file(filename, true, [&](auto& file) {
          for (const auto word : spv) {
            file.write((const char*)&word, 4);
          }
        });

        // optimized spirv
        if (!opt_spv.empty()) {
          std::string filename = pathed_name + ".opt.spv";
          write_to_file(filename, true, [&](auto& file) {
            for (const auto word : spv) {
              file.write((const char*)&word, 4);
            }
          });
        }
      }

      if (ShaderArtifactTypeBits::kSpvDis & artifacts) {
        // unoptimized spirv
        std::string filename = pathed_name + ".spvdis";
        write_to_file(filename, false, [&](auto& file) { spv::Disassemble(file, spv); });

        // optimized spirv
        if (!opt_spv.empty()) {
          std::string filename = pathed_name + ".opt.spvdis";
          write_to_file(
              filename, false, [&](auto& file) { spv::Disassemble(file, opt_spv); });
        }
      }

      if ((ShaderArtifactTypeBits::kSpvGlsl | ShaderArtifactTypeBits::kSpvReflect) &
          artifacts) {
        // Use the parser interface (demonstrate separate Parser invocation,
        // CompilerGLSL can take spirv directly as well).
        std::string filename = pathed_name;
        try {
          spirv_cross::Parser spirv_parser{spirv_t(spv)};
          spirv_parser.parse();
          spirv_cross::CompilerGLSL glsl(spirv_parser.get_parsed_ir());

          // Cross compile back to GLSL
          if (ShaderArtifactTypeBits::kSpvGlsl & artifacts) {
            filename = pathed_name + ".spv.glsl";
            write_to_file(filename, false, [&](auto& file) {
              spirv_cross::CompilerGLSL::Options options;
              // These happen to be the default options. Set them anyway in case
              // that changes
              options.version = 450;
              options.es = false;
              options.vulkan_semantics = true;
              glsl.set_common_options(options);
              file << glsl.compile() << std::endl;
            });
          }

          // reflection
          if (ShaderArtifactTypeBits::kSpvReflect & artifacts) {
            filename = pathed_name + ".spv.reflection";
            write_to_file(
                filename, false, [&](auto& file) { write_spirv_reflection(file, glsl); });
          }
        } catch (spirv_cross::CompilerError& e) {
          LOG(ERROR) << "SPIRV-Cross exception writing artifacts for '" << filename
                     << "' (" << e.what() << ")";
        } catch (std::exception& e) {
          LOG(ERROR) << "Exception writing artifacts for '" << filename << "' ("
                     << e.what() << ")";
        }
      }
    }
  }
}

}  // namespace gfx
