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
 * @file CacheEvictionAlgorithm.h
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
