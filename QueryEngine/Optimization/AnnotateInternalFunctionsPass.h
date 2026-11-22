/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

#include <set>
#include <string>
#include <vector>

#include "Logger/Logger.h"

/**
 * Annotates internal functions with function attributes designating the function as one
 * which does not modify memory, does not throw, does not synchronize state with other
 * functions/parts of the program, and is guaranteed to return. This allows the LLVM
 * optimizer to more aggressively remove / reorder these functions and is particularly
 * important for dead code elimination.
 */
class AnnotateInternalFunctionsPass
    : public llvm::PassInfoMixin<AnnotateInternalFunctionsPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module& module, llvm::ModuleAnalysisManager& /*am*/) {
    bool updated_function_defs = false;
    for (llvm::Function& fcn : module) {
      if (fcn.isDeclaration()) {
        continue;
      }
      if (isInternalStatelessFunction(fcn.getName()) ||
          isInternalMathFunction(fcn.getName())) {
        applyInternalFunctionAnnotations(fcn);
        updated_function_defs = true;
      }
    }
    return updated_function_defs ? llvm::PreservedAnalyses::none()
                                 : llvm::PreservedAnalyses::all();
  }

 private:
  static void applyInternalFunctionAnnotations(llvm::Function& fcn) {
    // WriteOnly is automatically added to all math functions in llvm 14.0
    // https://reviews.llvm.org/D116426 which is incompatible with ReadNone.
    fcn.removeFnAttr(llvm::Attribute::WriteOnly);

    // LLVM 17+ only allows readnone/readonly on arguments; use memory(none)
    // for function-level "does not access memory" semantics.
    fcn.addFnAttr(llvm::Attribute::getWithMemoryEffects(fcn.getContext(),
                                                        llvm::MemoryEffects::none()));

    const std::vector<llvm::Attribute::AttrKind> attrs{llvm::Attribute::NoFree,
                                                       llvm::Attribute::NoSync,
                                                       llvm::Attribute::NoUnwind,
                                                       llvm::Attribute::WillReturn,
                                                       llvm::Attribute::Speculatable};
    for (const auto& attr : attrs) {
      fcn.addFnAttr(attr);
    }
  }

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
};

inline const std::set<std::string> AnnotateInternalFunctionsPass::extension_functions =
    std::set<std::string>{"point_coord_array_is_null",
                          "decompress_x_coord_geoint",
                          "decompress_y_coord_geoint",
                          "compress_x_coord_geoint",
                          "compress_y_coord_geoint",
                          // GeoOpsRuntime.cpp
                          "transform_4326_900913_x",
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
inline const std::set<std::string> AnnotateInternalFunctionsPass::math_builtins =
    std::set<std::string>{"Acos",  "Asin",    "Atan", "Atan2",    "Ceil",    "Cos",
                          "Cot",   "degrees", "Exp",  "Floor",    "ln",      "Log",
                          "Log10", "log",     "pi",   "power",    "radians", "Round",
                          "Sin",   "Tan",     "tan",  "Truncate", "is_nan",  "is_inf"};
