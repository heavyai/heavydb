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

#ifndef BUFFERSEG_H
#define BUFFERSEG_H

#include <list>

namespace Buffer_Namespace {

class Buffer;  // forward declaration

// Memory Pages types in buffer pool
enum MemStatus { FREE, USED };

struct BufferSeg {
  int startPage;
  size_t numPages;
  MemStatus memStatus;
  Buffer* buffer;
  ChunkKey chunkKey;
  unsigned int pinCount;
  int slabNum;
  unsigned int lastTouched;

  BufferSeg() : memStatus(FREE), buffer(0), pinCount(0), slabNum(-1), lastTouched(0) {}
  BufferSeg(const int startPage, const size_t numPages)
      : startPage(startPage),
        numPages(numPages),
        memStatus(FREE),
        buffer(0),
        pinCount(0),
        slabNum(-1),
        lastTouched(0) {}
  BufferSeg(const int startPage, const size_t numPages, const MemStatus memStatus)
      : startPage(startPage),
        numPages(numPages),
        memStatus(memStatus),
        buffer(0),
        pinCount(0),
        slabNum(-1),
        lastTouched(0) {}
  BufferSeg(const int startPage, const size_t numPages, const MemStatus memStatus, const int lastTouched)
      : startPage(startPage),
        numPages(numPages),
        memStatus(memStatus),
        buffer(0),
        pinCount(0),
        slabNum(-1),
        lastTouched(lastTouched) {}
};

typedef std::list<BufferSeg> BufferList;
}

#endif
