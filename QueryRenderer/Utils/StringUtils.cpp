/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/StringUtils.h"

#include <algorithm>

namespace QueryRenderer {

std::string makeLowerCase(const std::string& str) {
  std::string rtn(str);
  std::transform(rtn.begin(), rtn.end(), rtn.begin(), ::tolower);

  return rtn;
}

std::string makeUpperCase(const std::string& str) {
  std::string rtn(str);
  std::transform(rtn.begin(), rtn.end(), rtn.begin(), ::toupper);

  return rtn;
}

}  // namespace QueryRenderer
