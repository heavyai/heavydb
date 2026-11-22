/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include "IRCodegenUtils.h"

#include <llvm/IR/Module.h>

inline void check_udf_param_is_pointer(llvm::Module* module,
                                       const std::string& func_name,
                                       size_t param_num) {
  llvm::Function* udf_func = module->getFunction(func_name);
  if (!udf_func) {
    return;
  }
  llvm::FunctionType* udf_func_type = udf_func->getFunctionType();
  CHECK(param_num < udf_func_type->getNumParams());
  CHECK(udf_func_type->getParamType(param_num)->isPointerTy());
}

// Buffer / Array abstraction: { T* ptr, i64 size, i8 is_null }
inline llvm::StructType* buffer_struct_ty(llvm::LLVMContext& ctx,
                                          llvm::Type* elem_ptr_type) {
  CHECK(elem_ptr_type);
  CHECK(elem_ptr_type->isPointerTy());
  return llvm::StructType::get(
      ctx, {elem_ptr_type, get_int_type(64, ctx), get_int_type(8, ctx)}, false);
}

// TextEncodingNone: { i8* ptr, i64 size, i8 is_null }
inline llvm::StructType* text_encoding_none_struct_ty(llvm::LLVMContext& ctx) {
  return llvm::StructType::get(ctx,
                               {typed_ptr_ty(get_int_type(8, ctx), 0),
                                get_int_type(64, ctx),
                                get_int_type(8, ctx)},
                               false);
}

// GeoPoint / GeoMultiPoint / GeoLineString: { i8* ptr, i32 x4 metadata }
inline llvm::StructType* geo_point_like_struct_ty(llvm::LLVMContext& ctx) {
  return llvm::StructType::get(ctx,
                               {typed_ptr_ty(get_int_type(8, ctx), 0),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx)},
                               false);
}

inline llvm::StructType* geo_point_struct_ty(llvm::LLVMContext& ctx) {
  return geo_point_like_struct_ty(ctx);
}

inline llvm::StructType* geo_multipoint_struct_ty(llvm::LLVMContext& ctx) {
  return geo_point_like_struct_ty(ctx);
}

inline llvm::StructType* geo_linestring_struct_ty(llvm::LLVMContext& ctx) {
  return geo_point_like_struct_ty(ctx);
}

// GeoMultiLineString / GeoPolygon: { i8* coords, i32, i8* sizes, i32 x4 metadata }
inline llvm::StructType* geo_multi_linestring_struct_ty(llvm::LLVMContext& ctx) {
  return llvm::StructType::get(ctx,
                               {typed_ptr_ty(get_int_type(8, ctx), 0),
                                get_int_type(32, ctx),
                                typed_ptr_ty(get_int_type(8, ctx), 0),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx)},
                               false);
}

inline llvm::StructType* geo_polygon_struct_ty(llvm::LLVMContext& ctx) {
  return geo_multi_linestring_struct_ty(ctx);
}

// GeoMultiPolygon: 9 fields with coords, ring sizes, and polygon bounds
inline llvm::StructType* geo_multi_polygon_struct_ty(llvm::LLVMContext& ctx) {
  return llvm::StructType::get(ctx,
                               {typed_ptr_ty(get_int_type(8, ctx), 0),
                                get_int_type(32, ctx),
                                typed_ptr_ty(get_int_type(8, ctx), 0),
                                get_int_type(32, ctx),
                                typed_ptr_ty(get_int_type(8, ctx), 0),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx),
                                get_int_type(32, ctx)},
                               false);
}
