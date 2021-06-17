/*
 * Copyright 2019 OmniSci, Inc.
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

/**
 * @file    UDFCompiler.h
 * @author  Michael Collison <michael.collison@omnisci.com>
 * @brief   External interface for parsing AST and bitcode files
 *
 * Copyright (c) 2018 OmniSci, Inc.
 */

#ifndef UDF_COMPILER_H
#define UDF_COMPILER_H

#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <string>
#include <vector>

#include "CudaMgr/CudaMgr.h"

class UdfClangDriver {
 public:
  UdfClangDriver(const std::string&);
  clang::driver::Driver* getClangDriver() { return &the_driver; }

 private:
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diag_options;
  clang::DiagnosticConsumer* diag_client;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_id;
  clang::DiagnosticsEngine diags;
  std::unique_ptr<clang::DiagnosticConsumer> diag_client_owner;
  clang::driver::Driver the_driver;
};

class UdfCompiler {
 public:
  UdfCompiler(const std::string& udf_file_name,
              CudaMgr_Namespace::NvidiaDeviceArch target_arch,
              const std::string& clang_path = "");
  UdfCompiler(const std::string& udf_file_name,
              CudaMgr_Namespace::NvidiaDeviceArch target_arch,
              const std::string& clang_path,
              const std::vector<std::string> clang_options);
  int compileUdf();
  const std::string& getAstFileName() const;

 private:
  void init(const std::string& clang_path);
  std::string removeFileExtension(const std::string& path);
  std::string getFileExt(std::string& s);
  int parseToAst(const char* file_name);
  std::string genGpuIrFilename(const char* udf_file_name);
  std::string genCpuIrFilename(const char* udf_file_name);
  int compileToGpuByteCode(const char* udf_file_name, bool cpu_mode);
  int compileToCpuByteCode(const char* udf_file_name);
  void replaceExtn(std::string& s, const std::string& new_ext);
  int compileFromCommandLine(const std::vector<std::string>& command_line);
  void readCompiledModules();
  void readGpuCompiledModule();
  void readCpuCompiledModule();
  int compileForGpu();

 private:
  std::string udf_file_name_;
  std::string udf_ast_file_name_;
#ifdef HAVE_CUDA
  CudaMgr_Namespace::NvidiaDeviceArch target_arch_;
#endif
  std::string clang_path_;
  std::vector<std::string> clang_options_;
};
#endif
