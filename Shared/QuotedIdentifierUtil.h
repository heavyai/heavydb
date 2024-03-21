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

#pragma once

#include <string>
#include <vector>

namespace shared {

/**
 * Concatenate a vector of identifiers into a composite identifier represented as a single
 * string.
 */
std::string concatenate_identifiers(const std::vector<std::string>& identifiers,
                                    const char delimiter = '.');

/**
 * Split a composite identifier.
 *
 * NOTE: This function is intended as the inverse of `concatenate_identifiers`.
 *
 */
std::vector<std::string> split_identifiers(const std::string& composite_identifier,
                                           const char delimiter = '.',
                                           const char quote = '\"');

}  // namespace shared
