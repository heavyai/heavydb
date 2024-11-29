/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include <list>
#include <shared_mutex>

#include "Shared/types.h"

namespace Buffer_Namespace {

class Buffer;  // forward declaration

// Memory Pages types in buffer pool
enum MemStatus { FREE, USED };

struct BufferSeg {
  int start_page;
  size_t num_pages;
  MemStatus mem_status;
  ChunkKey chunk_key;
  unsigned int pin_count;
  int slab_num;

  BufferSeg()
      : mem_status(FREE), pin_count(0), slab_num(-1), last_touched(0), buffer(nullptr) {}

  BufferSeg(const int start_page, const size_t num_pages)
      : start_page(start_page)
      , num_pages(num_pages)
      , mem_status(FREE)
      , pin_count(0)
      , slab_num(-1)
      , last_touched(0)
      , buffer(nullptr) {}

  BufferSeg(const int start_page, const size_t num_pages, const MemStatus mem_status)
      : start_page(start_page)
      , num_pages(num_pages)
      , mem_status(mem_status)
      , pin_count(0)
      , slab_num(-1)
      , last_touched(0)
      , buffer(nullptr) {}

  BufferSeg(const int start_page,
            const size_t num_pages,
            const MemStatus mem_status,
            const int last_touched)
      : start_page(start_page)
      , num_pages(num_pages)
      , mem_status(mem_status)
      , pin_count(0)
      , slab_num(-1)
      , last_touched(last_touched)
      , buffer(nullptr) {}

  BufferSeg(const BufferSeg& buffer_seg)
      : start_page(buffer_seg.start_page)
      , num_pages(buffer_seg.num_pages)
      , mem_status(buffer_seg.mem_status)
      , chunk_key(buffer_seg.chunk_key)
      , pin_count(buffer_seg.pin_count)
      , slab_num(buffer_seg.slab_num)
      , last_touched(buffer_seg.last_touched)
      , buffer(buffer_seg.buffer) {}

  void setBuffer(Buffer* buffer_ptr) {
    std::unique_lock<std::shared_mutex> buffer_lock(buffer_mutex);
    buffer = buffer_ptr;
  }

  Buffer* getBuffer() const {
    std::shared_lock<std::shared_mutex> buffer_lock(buffer_mutex);
    return buffer;
  }

  void setLastTouched(uint32_t last_touched_param) {
    std::unique_lock<std::shared_mutex> lock(last_touched_mutex);
    last_touched = last_touched_param;
  }

  uint32_t getLastTouched() const {
    std::shared_lock<std::shared_mutex> lock(last_touched_mutex);
    return last_touched;
  }

 private:
  uint32_t last_touched;
  Buffer* buffer;

  mutable std::shared_mutex last_touched_mutex;
  mutable std::shared_mutex buffer_mutex;
};

using BufferList = std::list<BufferSeg>;
}  // namespace Buffer_Namespace
