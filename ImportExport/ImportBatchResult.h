/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Fragmenter/Fragmenter.h"

namespace import_export {

struct ImportStatus;

class ImportBatchResult {
 public:
  virtual ~ImportBatchResult() = default;

  virtual std::optional<Fragmenter_Namespace::InsertData> getInsertData() const = 0;

  virtual ImportStatus getImportStatus() const = 0;
};

};  // namespace import_export
