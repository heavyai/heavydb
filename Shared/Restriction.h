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

/**
 * @file		Restriction.cpp
 * @brief		Structure to hold details of restrictions a given session has.
 *              Column: name of the column the restriction is on
 *              Values: vector of strings the restriction allows access to
 *
 * Copyright (c) 2021 OmniSci, Inc.  All rights reserved.
 **/
#pragma once

#include <cstdint>
#include "Logger/Logger.h"

struct Restriction {
  std::string column;
  std::vector<std::string> values;

  Restriction(const std::string& c, const std::vector<std::string> v) {
    column = c;
    values = v;
  };
};
