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

extern int parseToAst(const char* fileName);
extern std::string gen_gpu_ir_filename(const char* udf_fileName);
extern std::string gen_cpu_ir_filename(const char* udf_fileName);
extern int compileToGpuByteCode(const char* udf_fileNamem, bool cpu_mode);
extern int compileToCpuByteCode(const char* udf_fileName);
extern void replaceExtn(std::string& s, const std::string& newExt);

#endif
