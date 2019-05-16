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

#include "UDFCompiler.h"
#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <glog/logging.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <boost/process/search_path.hpp>
#include <memory>
#include "Execute.h"

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory ToolingSampleCategory("UDF Tooling");

namespace {

// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.

class FunctionDeclVisitor : public RecursiveASTVisitor<FunctionDeclVisitor> {
 public:
  FunctionDeclVisitor(llvm::raw_fd_ostream& ast_file, SourceManager& s_manager)
      : ast_file_(ast_file), source_manager_(s_manager) {
    source_manager_.getDiagnostics().setShowColors();
  }

  bool VisitFunctionDecl(FunctionDecl* f) {
    // Only function definitions (with bodies), not declarations.
    if (f->hasBody()) {
      if (getMainFileName() == getFuncDeclFileName(f)) {
        f->dump(ast_file_);
      }
    }

    return true;
  }

 private:
  std::string getMainFileName() const {
    auto f_entry = source_manager_.getFileEntryForID(source_manager_.getMainFileID());
    return f_entry->getName().str();
  }

  std::string getFuncDeclFileName(FunctionDecl* f) const {
    SourceLocation spell_loc = source_manager_.getSpellingLoc(f->getLocation());
    PresumedLoc p_loc = source_manager_.getPresumedLoc(spell_loc);

    return std::string(p_loc.getFilename());
  }

 private:
  llvm::raw_fd_ostream& ast_file_;
  SourceManager& source_manager_;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class DeclASTConsumer : public ASTConsumer {
 public:
  DeclASTConsumer(llvm::raw_fd_ostream& ast_file, SourceManager& s_manager)
      : visitor_(ast_file, s_manager) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef decl_reference) override {
    for (DeclGroupRef::iterator b = decl_reference.begin(), e = decl_reference.end();
         b != e;
         ++b) {
      // Traverse the declaration using our AST visitor.
      visitor_.TraverseDecl(*b);
    }
    return true;
  }

 private:
  FunctionDeclVisitor visitor_;
};

// For each source file provided to the tool, a new FrontendAction is created.
class HandleDeclAction : public ASTFrontendAction {
 public:
  HandleDeclAction(llvm::raw_fd_ostream& ast_file) : ast_file_(ast_file) {}

  ~HandleDeclAction() override {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& instance,
                                                 StringRef file) override {
    return llvm::make_unique<DeclASTConsumer>(ast_file_, instance.getSourceManager());
  }

 private:
  llvm::raw_fd_ostream& ast_file_;
};

class ToolFactory : public FrontendActionFactory {
 public:
  ToolFactory(llvm::raw_fd_ostream& ast_file) : ast_file_(ast_file) {}

  clang::FrontendAction* create() override { return new HandleDeclAction(ast_file_); }

 private:
  llvm::raw_fd_ostream& ast_file_;
};

bool on_search_path(const std::string file) {
  boost::filesystem::path p = boost::process::search_path(file);
  return boost::filesystem::exists(p);
}
}  // namespace

std::string UdfCompiler::removeFileExtension(const std::string& path) {
  if (path == "." || path == "..") {
    return path;
  }

  size_t pos = path.find_last_of("\\/.");
  if (pos != std::string::npos && path[pos] == '.') {
    return path.substr(0, pos);
  }

  return path;
}

std::string UdfCompiler::getFileExt(std::string& s) {
  size_t i = s.rfind('.', s.length());
  if (1 != std::string::npos) {
    return (s.substr(i + 1, s.length() - i));
  }
}

void UdfCompiler::replaceExtn(std::string& s, const std::string& new_ext) {
  std::string::size_type i = s.rfind('.', s.length());

  if (i != std::string::npos) {
    s.replace(i + 1, getFileExt(s).length(), new_ext);
  }
}

std::string UdfCompiler::genGpuIrFilename(const char* udf_file_name) {
  std::string gpu_file_name(removeFileExtension(udf_file_name));

  gpu_file_name += "_gpu.bc";
  return gpu_file_name;
}

std::string UdfCompiler::genCpuIrFilename(const char* udf_fileName) {
  std::string cpu_file_name(removeFileExtension(udf_fileName));

  cpu_file_name += "_cpu.bc";
  return cpu_file_name;
}

int UdfCompiler::compileFromCommandLine(std::vector<const char*>& command_line) {
  auto a_path = llvm::sys::findProgramByName("clang++");
  auto clang_path = a_path.get();

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diag_options(
      new DiagnosticOptions());

  clang::DiagnosticConsumer* diag_client =
      new TextDiagnosticPrinter(llvm::errs(), diag_options.get());

  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_id(new clang::DiagnosticIDs());
  clang::DiagnosticsEngine diags(diag_id, diag_options.get(), diag_client);

  std::unique_ptr<clang::DiagnosticConsumer> diag_client_owner(diags.takeClient());

  clang::driver::Driver the_driver(
      clang_path.c_str(), llvm::sys::getDefaultTargetTriple(), diags);

  the_driver.CCPrintOptions = 0;
  std::unique_ptr<driver::Compilation> compilation(
      the_driver.BuildCompilation(command_line));

  if (!compilation) {
    LOG(FATAL) << "failed to build compilation object!\n";
  }

  llvm::SmallVector<std::pair<int, const driver::Command*>, 10> failing_commands;
  int res = the_driver.ExecuteCompilation(*compilation, failing_commands);

  if (res < 0) {
    for (const std::pair<int, const driver::Command*>& p : failing_commands) {
      if (p.first) {
        the_driver.generateCompilationDiagnostics(*compilation, *p.second);
      }
    }
  }

  return res;
}

int UdfCompiler::compileToGpuByteCode(const char* udf_file_name, bool cpu_mode) {
  auto a_path = llvm::sys::findProgramByName("clang++");
  auto clang_path = a_path.get();

  std::string gpu_outName(genGpuIrFilename(udf_file_name));

  std::vector<const char*> command_line{clang_path.c_str(),
                                        "-c",
                                        "-O2",
                                        "-emit-llvm",
                                        "-o",
                                        gpu_outName.c_str(),
                                        "-std=c++14"};

  // If we are not compiling for cpu mode, then target the gpu
  // Otherwise assume we can generic ir that will
  // be translated to gpu code during target code generation
  if (!cpu_mode) {
    command_line.emplace_back("--cuda-gpu-arch=sm_30");
    command_line.emplace_back("--cuda-device-only");
    command_line.emplace_back("-xcuda");
  }

  command_line.emplace_back(udf_file_name);

  return compileFromCommandLine(command_line);
}

int UdfCompiler::compileToCpuByteCode(const char* udf_file_name) {
  auto a_path = llvm::sys::findProgramByName("clang++");
  auto clang_path = a_path.get();

  std::string cpu_outName(genCpuIrFilename(udf_file_name));

  std::vector<const char*> command_line{clang_path.c_str(),
                                        "-c",
                                        "-O2",
                                        "-emit-llvm",
                                        "-o",
                                        cpu_outName.c_str(),
                                        "-std=c++14",
                                        udf_file_name};

  return compileFromCommandLine(command_line);
}

int UdfCompiler::parseToAst(const char* file_name) {
  int num_args = 3;
  const char arg0[] = "astparser";
  const char* arg1 = file_name;
  const char arg2[] = "--";
  const char* arg_vector[3] = {arg0, arg1, arg2};

  CommonOptionsParser op(num_args, arg_vector, ToolingSampleCategory);
  ClangTool tool(op.getCompilations(), op.getSourcePathList());

  std::string out_name(file_name);
  std::string file_ext("ast");
  replaceExtn(out_name, file_ext);

  std::error_code out_error_info;
  llvm::raw_fd_ostream out_file(
      llvm::StringRef(out_name), out_error_info, llvm::sys::fs::F_None);

  auto factory = llvm::make_unique<ToolFactory>(out_file);
  return tool.run(factory.get());
}

const std::string& UdfCompiler::getAstFileName() const {
  return udf_ast_file_name_;
}

UdfCompiler::UdfCompiler(const std::string& file_name)
    : udf_file_name_(file_name), udf_ast_file_name_(file_name) {
  replaceExtn(udf_ast_file_name_, "ast");
}

void UdfCompiler::readCompiledModules() {
  std::string cpu_ir_file(genCpuIrFilename(udf_file_name_.c_str()));
  std::string gpu_ir_file(genGpuIrFilename(udf_file_name_.c_str()));

  VLOG(1) << "UDFCompiler cpu bc file = " << cpu_ir_file << std::endl;
  VLOG(1) << "UDFCompiler gpu bc file = " << gpu_ir_file << std::endl;

  read_udf_cpu_module(cpu_ir_file);
  read_udf_gpu_module(gpu_ir_file);
}

int UdfCompiler::compileForGpu() {
  int gpu_compile_result = 1;

  if (on_search_path("nvcc")) {
    gpu_compile_result = compileToGpuByteCode(udf_file_name_.c_str(), false);
  }

  // If gpu compilation fails but cpu compilation has succeeded, try compiling
  // for the cpu with the assumption the user does not have the CUDA toolkit
  // installed
  if (gpu_compile_result != 0) {
    gpu_compile_result = compileToGpuByteCode(udf_file_name_.c_str(), true);
  }

  return gpu_compile_result;
}

int UdfCompiler::compileUdf() {
  if (on_search_path("clang++")) {
    LOG(INFO) << "UDFCompiler filename to compiler: " << udf_file_name_ << std::endl;
    if (!boost::filesystem::exists(udf_file_name_)) {
      LOG(FATAL) << "User defined function file " << udf_file_name_ << " does not exist.";
      return 1;
    }

    auto ast_result = parseToAst(udf_file_name_.c_str());

    if (ast_result == 0) {
      // Compile udf file to generate cpu and gpu bytecode files

      int cpu_compile_result = compileToCpuByteCode(udf_file_name_.c_str());
      int gpu_compile_result = 1;

      if (cpu_compile_result == 0) {
        gpu_compile_result = compileForGpu();

        if (gpu_compile_result == 0) {
          readCompiledModules();
        } else {
          LOG(FATAL) << "Unable to compile UDF file for gpu" << std::endl;
          return 1;
        }
      } else {
        LOG(FATAL) << "Unable to compile UDF file for cpu" << std::endl;
        return 1;
      }
    } else {
      LOG(FATAL) << "Unable to create AST file for udf compilation" << std::endl;
      return 1;
    }
  } else {
    LOG(FATAL) << "Unable to compile udfs due to absence of clang++" << std::endl;
    return 1;
  }

  return 0;
}
