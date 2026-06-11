/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file CacheEvictionAlgorithm.h
 * @brief
 *
 * This file includes the class specification for the cache eviction algorithm interface
 * used by the Foreign Storage Interface (FSI).  This interface can be implemented to
 * quickly slot out different caching algorithms for the FSI cache.
 * A caching algorithm can be queried to determine which chunks should be evicted in what
 * order and needs to be updated with cache usage data.
 */

#pragma once

#include "DataMgr/AbstractBufferMgr.h"

class NoEntryFoundException : public std::runtime_error {
 public:
  NoEntryFoundException()
      : std::runtime_error("Cache attempting to evict from empty queue") {}
};

class CacheEvictionAlgorithm {
 public:
  virtual ~CacheEvictionAlgorithm() {}
  virtual const ChunkKey evictNextChunk() = 0;
  virtual void touchChunk(const ChunkKey&) = 0;
  virtual void removeChunk(const ChunkKey&) = 0;
};
