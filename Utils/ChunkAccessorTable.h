/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file ChunkAccessorTable.h
 * @brief
 *
 */

#pragma once

#include "Catalog/CatalogFwd.h"
#include "DataMgr/Chunk/Chunk.h"

#include <tuple>
#include <vector>

// convenience functions for multi-fragment support in multi-threaded worker functions
// (poly rendering, importer)

using ChunkIterVector = std::vector<ChunkIter>;
using ChunkAccessorTable = std::vector<
    std::tuple<size_t, std::vector<std::shared_ptr<Chunk_NS::Chunk>>, ChunkIterVector>>;

ChunkAccessorTable getChunkAccessorTable(const Catalog_Namespace::Catalog& cat,
                                         const TableDescriptor* td,
                                         const std::vector<std::string>& columnNames);

ChunkIterVector& getChunkItersAndRowOffset(ChunkAccessorTable& table,
                                           size_t rowid,
                                           size_t& rowOffset);
