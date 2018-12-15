/*
 * Copyright 2018 OmniSci, Inc.
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

#include "sqldefs.h"

#include <glog/logging.h>
#include <string>

inline std::string sql_window_function_to_str(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::ROW_NUMBER: {
      return "ROW_NUMBER";
    }
    case SqlWindowFunctionKind::LAG: {
      return "LAG";
    }
    case SqlWindowFunctionKind::LEAD: {
      return "LEAD";
    }
    default: {
      LOG(FATAL) << "Invalid window function kind";
      return "";
    }
  }
}
