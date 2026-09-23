/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    SourceType.h
 * @brief   Shared Enum
 *
 */

#pragma once

namespace import_export {

enum class SourceType {
  kUnknown,
  kUnsupported,
  kDelimitedFile,
  kGeoFile,
  kRasterFile,
  kParquetFile,
  kOdbc,
  kRegexParsedFile,
};

}  // namespace import_export
