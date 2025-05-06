/*
 * Copyright 2023 HEAVY.AI, Inc.
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
 * @file    enums.h
 * @brief   QueryEngine enum classes with minimal #include files.
 */

#pragma once

#include "Shared/define_enum_class.h"

namespace heavyai {

// clang-format off
HEAVYAI_DEFINE_ENUM_CLASS_WITH_DESCRIPTIONS(ErrorCode
  , (NO_ERROR, "No Error")
  , (DIV_BY_ZERO, "Division by zero")
  , (OUT_OF_GPU_MEM, "Query couldn't keep the entire working set of columns in GPU memory")
  , (OUT_OF_SLOTS, "Out of Slots")
  , (UNSUPPORTED_SELF_JOIN, "Self joins not supported yet")
  , (OUT_OF_RENDER_MEM, "Insufficient GPU memory for query results in render output buffer sized by render-mem-bytes")
  , (OUT_OF_CPU_MEM, "Not enough host memory to execute the query")
  , (OVERFLOW_OR_UNDERFLOW, "Overflow or underflow")
  , (OUT_OF_TIME, "Query execution has exceeded the time limit")
  , (INTERRUPTED, "Query execution has been interrupted")
  , (COLUMNAR_CONVERSION_NOT_SUPPORTED, "Columnar conversion not supported for variable length types")
  , (TOO_MANY_LITERALS, "Too many literals in the query")
  , (STRING_CONST_IN_RESULTSET, "NONE ENCODED String types are not supported as input result set.")
  , (STREAMING_TOP_N_NOT_SUPPORTED_IN_RENDER_QUERY, "Streaming-Top-N not supported in Render Query")
  , (SINGLE_VALUE_FOUND_MULTIPLE_VALUES, "Multiple distinct values encountered")
  , (GEOS_OR_H3, "GEOS or H3 error")
  , (WIDTH_BUCKET_INVALID_ARGUMENT, "Arguments of WIDTH_BUCKET function does not satisfy the condition")
  , (BBOX_OVERLAPS_LIMIT_EXCEEDED, "Maximum supported number of bounding box overlaps exceeded")
  , (PROBING_LENGTH_EXCEEDED, "Hash table probing length exceeded on insert.")
  )
// clang-format on

HEAVYAI_DEFINE_ENUM_CLASS(QueryDescriptionType,
                          GroupByPerfectHash,
                          GroupByBaselineHash,
                          Projection,
                          TableFunction,
                          NonGroupedAggregate,
                          Estimator)

// Enumerate the pointer parameters passed to the kernel.
// LITERALS are included iff literals are hoisted.
HEAVYAI_DEFINE_ENUM_CLASS(KernelParam,
                          ERROR_CODE,
                          TOTAL_MATCHED,
                          GROUPBY_BUF,
                          NUM_FRAGMENTS,
                          NUM_TABLES,
                          ROW_INDEX_RESUME,
                          COL_BUFFERS,
                          LITERALS,
                          NUM_ROWS,
                          FRAG_ROW_OFFSETS,
                          FRAG_IDS,
                          MAX_MATCHED,
                          INIT_AGG_VALS,
                          JOIN_HASH_TABLES,
                          ROW_FUNC_MGR)

}  // namespace heavyai
