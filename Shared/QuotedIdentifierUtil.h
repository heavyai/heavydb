/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
