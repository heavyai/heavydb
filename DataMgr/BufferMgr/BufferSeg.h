/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "Shared/types.h"

namespace Buffer_Namespace {

class Buffer;  // forward declaration

// Memory Pages types in buffer pool
enum MemStatus { FREE, USED };

struct BufferSeg {
  int start_page;
  size_t num_pages;
  MemStatus mem_status;
  Buffer* buffer;
  ChunkKey chunk_key;
  unsigned int pin_count;
  int slab_num;
  unsigned int last_touched;

  BufferSeg()
      : mem_status(FREE), buffer(0), pin_count(0), slab_num(-1), last_touched(0) {}
  BufferSeg(const int start_page, const size_t num_pages)
      : start_page(start_page)
      , num_pages(num_pages)
      , mem_status(FREE)
      , buffer(0)
      , pin_count(0)
      , slab_num(-1)
      , last_touched(0) {}
  BufferSeg(const int start_page, const size_t num_pages, const MemStatus mem_status)
      : start_page(start_page)
      , num_pages(num_pages)
      , mem_status(mem_status)
      , buffer(0)
      , pin_count(0)
      , slab_num(-1)
      , last_touched(0) {}
  BufferSeg(const int start_page,
            const size_t num_pages,
            const MemStatus mem_status,
            const int last_touched)
      : start_page(start_page)
      , num_pages(num_pages)
      , mem_status(mem_status)
      , buffer(0)
      , pin_count(0)
      , slab_num(-1)
      , last_touched(last_touched) {}
};

using BufferList = std::list<BufferSeg>;
}  // namespace Buffer_Namespace
