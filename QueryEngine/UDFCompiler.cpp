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
#include <memory>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory ToolingSampleCategory("UDF Tooling");

std::string remove_extension(const std::string& path) {
  if (path == "." || path == "..") {
    return path;
  }

  size_t pos = path.find_last_of("\\/.");
  if (pos != std::string::npos && path[pos] == '.') {
    return path.substr(0, pos);
  }

  return path;
}

static std::string getFileExt(std::string& s) {
  size_t i = s.rfind('.', s.length());
  if (1 != std::string::npos) {
    return (s.substr(i + 1, s.length() - i));
  }
}

void replaceExtn(std::string& s, const std::string& newExt) {
  std::string::size_type i = s.rfind('.', s.length());

  if (i != std::string::npos) {
    s.replace(i + 1, getFileExt(s).length(), newExt);
  }
}

// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.

class FunctionDeclVisitor : public RecursiveASTVisitor<FunctionDeclVisitor> {
 public:
  FunctionDeclVisitor(llvm::raw_fd_ostream& astFile, SourceManager& sManager)
      : mAstFile(astFile), mSourceManager(sManager) {
    mSourceManager.getDiagnostics().setShowColors();
  }

  bool VisitFunctionDecl(FunctionDecl* f) {
    // Only function definitions (with bodies), not declarations.
    if (f->hasBody()) {
      if (getMainFileName() == getFuncDeclFileName(f)) {
        f->dump(mAstFile);
      }
    }

    return true;
  }

 private:
  std::string getMainFileName() const {
    auto fEntry = mSourceManager.getFileEntryForID(mSourceManager.getMainFileID());
    return fEntry->getName().str();
  }

  std::string getFuncDeclFileName(FunctionDecl* f) const {
    SourceLocation spellLoc = mSourceManager.getSpellingLoc(f->getLocation());
    PresumedLoc pLoc = mSourceManager.getPresumedLoc(spellLoc);

    return std::string(pLoc.getFilename());
  }

 private:
  llvm::raw_fd_ostream& mAstFile;
  SourceManager& mSourceManager;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class DeclASTConsumer : public ASTConsumer {
 public:
  DeclASTConsumer(llvm::raw_fd_ostream& astFile, SourceManager& sManager)
      : Visitor(astFile, sManager) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      // Traverse the declaration using our AST visitor.
      Visitor.TraverseDecl(*b);
    }
    return true;
  }

 private:
  FunctionDeclVisitor Visitor;
};

// For each source file provided to the tool, a new FrontendAction is created.
class HandleDeclAction : public ASTFrontendAction {
 public:
  HandleDeclAction(llvm::raw_fd_ostream& astFile) : mAstFile(astFile) {}

  ~HandleDeclAction() override {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& instance,
                                                 StringRef file) override {
    return llvm::make_unique<DeclASTConsumer>(mAstFile, instance.getSourceManager());
  }

 private:
  llvm::raw_fd_ostream& mAstFile;
};

class ToolFactory : public FrontendActionFactory {
 public:
  ToolFactory(llvm::raw_fd_ostream& astFile) : mAstFile(astFile) {}

  clang::FrontendAction* create() override { return new HandleDeclAction(mAstFile); }

 private:
  llvm::raw_fd_ostream& mAstFile;
};

std::string gen_gpu_ir_filename(const char* udf_fileName) {
  std::string gpu_fileName(remove_extension(udf_fileName));

  gpu_fileName += "_gpu.bc";
  return gpu_fileName;
}

std::string gen_cpu_ir_filename(const char* udf_fileName) {
  std::string cpu_fileName(remove_extension(udf_fileName));

  cpu_fileName += "_cpu.bc";
  return cpu_fileName;
}

int compileFromCommandLine(std::vector<const char*>& commandLine) {
  auto aPath = llvm::sys::findProgramByName("clang++");
  auto clangPath = aPath.get();

  clang::DiagnosticOptions* diagOptions = new DiagnosticOptions();
  clang::DiagnosticConsumer* diagClient =
      new TextDiagnosticPrinter(llvm::errs(), diagOptions);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(new clang::DiagnosticIDs());
  clang::DiagnosticsEngine diags(diagID, diagOptions, diagClient);

  clang::driver::Driver theDriver(
      clangPath.c_str(), llvm::sys::getDefaultTargetTriple(), diags);

  theDriver.CCPrintOptions = 0;
  std::unique_ptr<driver::Compilation> compilation(
      theDriver.BuildCompilation(commandLine));

  if (!compilation) {
    LOG(FATAL) << "failed to build compilation object!\n";
  }

  llvm::SmallVector<std::pair<int, const driver::Command*>, 10> failingCommands;
  int res = theDriver.ExecuteCompilation(*compilation, failingCommands);

  if (res < 0) {
    for (const std::pair<int, const driver::Command*>& p : failingCommands) {
      if (p.first) {
        theDriver.generateCompilationDiagnostics(*compilation, *p.second);
      }
    }
  }

  return res;
}

int compileToGpuByteCode(const char* udf_fileName, bool cpu_mode) {
  auto aPath = llvm::sys::findProgramByName("clang++");
  auto clangPath = aPath.get();

  std::string gpu_outName(gen_gpu_ir_filename(udf_fileName));

  std::vector<const char*> commandLine;

  commandLine.push_back(clangPath.c_str());
  commandLine.push_back("-c");
  commandLine.push_back("-O2");
  commandLine.push_back("-emit-llvm");
  commandLine.push_back("-o");
  commandLine.push_back(gpu_outName.c_str());
  commandLine.push_back("-std=c++14");

  // If we are not compiling for cpu mode, then target the gpu
  // Otherwise assume we can generic ir that will
  // be translated to gpu code during target code generation
  if (!cpu_mode) {
    commandLine.push_back("--cuda-gpu-arch=sm_30");
    commandLine.push_back("--cuda-device-only");
    commandLine.push_back("-xcuda");
  }

  commandLine.push_back(udf_fileName);

  return compileFromCommandLine(commandLine);
}

int compileToCpuByteCode(const char* udf_fileName) {
  auto aPath = llvm::sys::findProgramByName("clang++");
  auto clangPath = aPath.get();

  std::string cpu_outName(gen_cpu_ir_filename(udf_fileName));

  std::vector<const char*> commandLine;

  commandLine.push_back(clangPath.c_str());
  commandLine.push_back("-c");
  commandLine.push_back("-O2");
  commandLine.push_back("-emit-llvm");

  commandLine.push_back("-o");
  commandLine.push_back(cpu_outName.c_str());
  commandLine.push_back("-std=c++14");
  commandLine.push_back(udf_fileName);

  return compileFromCommandLine(commandLine);
}

int parseToAst(const char* fileName) {
  int numArgs = 3;
  const char arg0[] = "astparser";
  const char* arg1 = fileName;
  const char arg2[] = "--";
  const char* argVector[3] = {arg0, arg1, arg2};

  CommonOptionsParser op(numArgs, argVector, ToolingSampleCategory);
  ClangTool tool(op.getCompilations(), op.getSourcePathList());

  std::string outName(fileName);
  std::string fileExt("ast");
  replaceExtn(outName, fileExt);

  std::error_code OutErrorInfo;
  llvm::raw_fd_ostream outFile(
      llvm::StringRef(outName), OutErrorInfo, llvm::sys::fs::F_None);

  auto factory = llvm::make_unique<ToolFactory>(outFile);
  return tool.run(factory.get());
}
