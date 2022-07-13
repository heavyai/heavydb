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

#ifndef QUERYENGINE_ARROW_UTIL_H
#define QUERYENGINE_ARROW_UTIL_H

#include <arrow/status.h>
#include <arrow/util/macros.h>
#include <stdexcept>
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Shared/likely.h"

inline void arrow_status_throw(const ::arrow::Status& s) {
  std::string message = s.ToString();
  switch (s.code()) {
    case ::arrow::StatusCode::OutOfMemory:
      throw OutOfMemory(message);
    default:
      throw std::runtime_error(message);
  }
}

#define ARROW_THROW_NOT_OK(s) \
  do {                        \
    ::arrow::Status _s = (s); \
    if (UNLIKELY(!_s.ok())) { \
      arrow_status_throw(_s); \
    }                         \
  } while (0)

// Based on ARROW_ASSIGN_OR_RAISE from arrow/result.h

#define ARROW_THROW_IF(condition, status) \
  do {                                    \
    if (ARROW_PREDICT_FALSE(condition)) { \
      ARROW_THROW_NOT_OK(status);         \
    }                                     \
  } while (0)

#define ARROW_ASSIGN_OR_THROW_IMPL(result_name, lhs, rexpr) \
  auto result_name = (rexpr);                               \
  ARROW_THROW_NOT_OK((result_name).status());               \
  lhs = std::move(result_name).MoveValueUnsafe();

#define ARROW_ASSIGN_OR_THROW_NAME(x, y) ARROW_CONCAT(x, y)

#define ARROW_ASSIGN_OR_THROW(lhs, rexpr) \
  ARROW_ASSIGN_OR_THROW_IMPL(             \
      ARROW_ASSIGN_OR_THROW_NAME(_error_or_value, __COUNTER__), lhs, rexpr);

#endif  // QUERYENGINE_ARROW_UTIL_H
