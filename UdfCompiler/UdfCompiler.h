/*
 * Copyright 2021 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UDF_COMPILER_H
#define UDF_COMPILER_H

#include <string>
#include <vector>

#include "CudaMgr/CudaMgr.h"

/**
 * Driver for calling clang/clang++ to compile C++ programs to LLVM IR for use as a UDF.
 * Default initialization will find Clang using the clang library invocations. An optional
 * clang override and additional arguments to the clang binary can be added. Once
 * initialized the class holds the state for calling clang until destruction.
 */
class UdfCompiler {
 public:
  UdfCompiler(CudaMgr_Namespace::NvidiaDeviceArch target_arch,
              const std::string& clang_path_override = "");
  UdfCompiler(CudaMgr_Namespace::NvidiaDeviceArch target_arch,
              const std::string& clang_path_override,
              const std::vector<std::string> clang_options);

  /**
   * Compile a C++ file to LLVM IR, and generate an AST file. Both artifacts exist as
   * files on disk. Three artifacts will be generated; the AST file, the CPU LLVM IR, and
   * GPU LLVM IR (if CUDA is enabled and compilation succeeds). These LLVM IR files can be
   * loaded by the Executor. The AST will be processed by Calcite.
   */
  std::pair<std::string, std::string> compileUdf(const std::string& udf_file_name) const;

  static std::string getAstFileName(const std::string& udf_file_name);

 private:
  /**
   * Call clang binary to generate abstract syntax tree file for registration in Calcite.
   */
  void generateAST(const std::string& file_name) const;

  static std::string genLLVMIRFilename(const std::string& udf_file_name);
  static std::string genCUDAIRFilename(const std::string& udf_file_name);

  /**
   * Formulate Clang command line command and call clang binary to generate LLVM IR for
   * the C/C++ file.
   */
#ifdef HAVE_CUDA
  std::string compileToCudaIR(const std::string& udf_file_name) const;
#endif
  std::string compileToLLVMIR(const std::string& udf_file_name) const;

  /**
   * Formulate the full compile command and call the compiler.
   */
  int compileFromCommandLine(const std::vector<std::string>& command_line) const;

  std::string clang_path_;
  std::vector<std::string> clang_options_;
#ifdef HAVE_CUDA
  CudaMgr_Namespace::NvidiaDeviceArch target_arch_;
#endif
};

#endif
