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

#include <llvm/IR/Function.h>

#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/CallGraphSCCPass.h>

#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>

#include "Logger/Logger.h"

/**
 * Annotates internal functions with function attributes designating the function as one
 * which does not modify memory, does not throw, does not synchronize state with other
 * functions/parts of the program, and is guaranteed to return. This allows the LLVM
 * optimizer to more aggressively remove / reorder these functions and is particularly
 * important for dead code elimination.
 */
class AnnotateInternalFunctionsPass : public llvm::CallGraphSCCPass {
 public:
  static char ID;
  AnnotateInternalFunctionsPass() : CallGraphSCCPass(ID) {}

  bool runOnSCC(llvm::CallGraphSCC& SCC) override {
    bool updated_function_defs = false;

    // iterate the call graph
    for (auto& node : SCC) {
      CHECK(node);
      auto fcn = node->getFunction();
      if (!fcn) {
        continue;
      }
      if (isInternalStatelessFunction(fcn->getName()) ||
          isInternalMathFunction(fcn->getName())) {
        updated_function_defs = true;
        std::vector<llvm::Attribute::AttrKind> attrs{llvm::Attribute::NoFree,
                                                     llvm::Attribute::NoSync,
                                                     llvm::Attribute::NoUnwind,
                                                     llvm::Attribute::WillReturn,
                                                     llvm::Attribute::ReadNone,
                                                     llvm::Attribute::Speculatable};
        for (const auto& attr : attrs) {
          fcn->addFnAttr(attr);
        }
      } else if (isReadOnlyFunction(fcn->getName())) {
        updated_function_defs = true;
        fcn->addFnAttr(llvm::Attribute::ReadOnly);
      }
    }

    return updated_function_defs;
  }

  llvm::StringRef getPassName() const override { return "AnnotateInternalFunctionsPass"; }

 private:
  static const std::set<std::string> extension_functions;

  static bool isInternalStatelessFunction(const llvm::StringRef& func_name) {
    // extension functions or non-inlined builtins which do not modify any state
    return extension_functions.count(func_name.str()) > 0;
  }

  static const std::set<std::string> math_builtins;

  static bool isInternalMathFunction(const llvm::StringRef& func_name) {
    // include all math functions from ExtensionFunctions.hpp
    return math_builtins.count(func_name.str()) > 0;
  }

  static const std::set<std::string> readonly_functions;

  static bool isReadOnlyFunction(const llvm::StringRef& func_name) {
    // functions which do not write through any pointer arguments or modify any state
    // visible to caller
    return readonly_functions.count(func_name.str()) > 0;
  }
};

char AnnotateInternalFunctionsPass::ID = 0;

const std::set<std::string> AnnotateInternalFunctionsPass::extension_functions =
    std::set<std::string>{"transform_4326_900913_x",
                          "transform_4326_900913_y",
                          "transform_900913_4326_x",
                          "transform_900913_4326_y",
                          // ExtensionFunctions.hpp
                          "conv_4326_900913_x",
                          "conv_4326_900913_y",
                          "distance_in_meters",
                          "approx_distance_in_meters",
                          "rect_pixel_bin_x",
                          "rect_pixel_bin_y",
                          "rect_pixel_bin_packed",
                          "reg_hex_horiz_pixel_bin_x",
                          "reg_hex_horiz_pixel_bin_y",
                          "reg_hex_horiz_pixel_bin_packed",
                          "reg_hex_vert_pixel_bin_x",
                          "reg_hex_vert_pixel_bin_y",
                          "reg_hex_vert_pixel_bin_packed",
                          "convert_meters_to_merc_pixel_width",
                          "convert_meters_to_merc_pixel_height",
                          "is_point_in_merc_view",
                          "is_point_size_in_merc_view"};

// TODO: consider either adding specializations here for the `__X` versions (for different
// types), or just truncate the function name removing the underscores in
// `isInternalMathFunction`.
const std::set<std::string> AnnotateInternalFunctionsPass::math_builtins =
    std::set<std::string>{"Acos",  "Asin",    "Atan", "Atan2",    "Ceil",    "Cos",
                          "Cot",   "degrees", "Exp",  "Floor",    "ln",      "Log",
                          "Log10", "log",     "pi",   "power",    "radians", "Round",
                          "Sin",   "Tan",     "tan",  "Truncate", "isNan"};

const std::set<std::string> AnnotateInternalFunctionsPass::readonly_functions =
    std::set<std::string>{"check_interrupt"};
