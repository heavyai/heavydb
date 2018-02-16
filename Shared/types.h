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

/*
 * File:        types.h
 * Author(s):   steve@map-d.com
 *
 * Created on June 19, 2014, 4:29 PM
 */

#ifndef _TYPES_H
#define _TYPES_H

#include <sstream>
#include <string>
#include <vector>
#include <thread>

// The ChunkKey is a unique identifier for chunks in the database file.
// The first element of the underlying vector for ChunkKey indicates the type of
// ChunkKey (also referred to as the keyspace id)
typedef std::vector<int> ChunkKey;

inline std::string showChunk(const ChunkKey& key) {
  std::ostringstream tss;
  for (auto vecIt = key.begin(); vecIt != key.end(); ++vecIt) {
    tss << *vecIt << ",";
  }
  return tss.str();
}

#ifndef NO_OOM_TRACE
void oom_trace_push(const std::string&);
void oom_trace_pop();
void oom_trace_dump();

struct OomStub {
  ~OomStub() { oom_trace_pop(); }
};

#define OOM_TRACE_PUSH(...)     \
  OomStub oomStub##__COUNTER__; \
  oom_trace_push(std::string(__func__) + ":" + std::to_string(__LINE__) + " " __VA_ARGS__)
#define OOM_TRACE_DUMP oom_trace_dump()
#else
#define OOM_TRACE_PUSH(...)
#define OOM_TRACE_DUMP
#endif  // OOM_TRACE

#endif /* _TYPES_H */
