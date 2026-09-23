/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>

#include "Catalog/CatalogFwd.h"
#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/Chunk/Chunk.h"

namespace foreign_storage {

struct IterativeFileScanParameters {
  IterativeFileScanParameters(
      std::map<std::pair<int, int>, Chunk_NS::Chunk>& column_id_to_chunk_map,
      int32_t fragment_id,
      const std::vector<AbstractBuffer*>& delete_buffers)
      : column_id_and_batch_id_to_chunk_map(column_id_to_chunk_map)
      , fragment_id(fragment_id)
      , delete_buffers(delete_buffers) {}

  std::map<std::pair<int, int>, Chunk_NS::Chunk>& column_id_and_batch_id_to_chunk_map;

  int32_t fragment_id;
  const std::vector<AbstractBuffer*>& delete_buffers;
};

}  // namespace foreign_storage
