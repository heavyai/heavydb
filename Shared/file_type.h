/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    file_type.h
 * @brief   shared utility for mime-types
 *
 */

#include <string>

namespace shared {

bool is_compressed_mime_type(const std::string& mime_type);
bool is_compressed_file_extension(const std::string& location);

}  // namespace shared