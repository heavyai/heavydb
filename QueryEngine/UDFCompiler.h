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

#include <string>
#include <vector>

class UdfCompiler {
 public:
  UdfCompiler(const std::string&);
  int compileUdf();
  const std::string& getAstFileName() const;

 private:
  std::string removeFileExtension(const std::string& path);
  std::string getFileExt(std::string& s);
  int parseToAst(const char* file_name);
  std::string genGpuIrFilename(const char* udf_file_name);
  std::string genCpuIrFilename(const char* udf_file_name);
  int compileToGpuByteCode(const char* udf_file_name, bool cpu_mode);
  int compileToCpuByteCode(const char* udf_file_name);
  void replaceExtn(std::string& s, const std::string& new_ext);
  int compileFromCommandLine(std::vector<const char*>& command_line);
  void readCompiledModules();
  int compileForGpu();

 private:
  std::string udf_file_name_;
  std::string udf_ast_file_name_;
};
#endif
