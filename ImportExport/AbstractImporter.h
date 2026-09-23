/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Catalog/SessionInfo.h>

namespace import_export {

struct ImportStatus;

class AbstractImporter {
 public:
  virtual ~AbstractImporter() = default;

  /*
   * Import data returning the status of the import.
   */
  virtual ImportStatus import(const Catalog_Namespace::SessionInfo* session_info) = 0;
};
}  // namespace import_export
