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

#include "MurmurHash.h"
#include "MurmurHash1Inl.h"

extern "C" NEVER_INLINE DEVICE uint32_t MurmurHash1(const void* key, int len, const uint32_t seed) {
  return MurmurHash1Impl(key, len, seed);
}

extern "C" NEVER_INLINE DEVICE uint64_t MurmurHash64A(const void* key, int len, uint64_t seed) {
  const uint64_t m = 0xc6a4a7935bd1e995LLU;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t* data = (const uint64_t*)key;
  const uint64_t* end = data + (len / 8);

  while (data != end) {
    uint64_t k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  const unsigned char* data2 = (const unsigned char*)data;

  switch (len & 7) {
    case 7:
      h ^= ((uint64_t)data2[6]) << 48;
    case 6:
      h ^= ((uint64_t)data2[5]) << 40;
    case 5:
      h ^= ((uint64_t)data2[4]) << 32;
    case 4:
      h ^= ((uint64_t)data2[3]) << 24;
    case 3:
      h ^= ((uint64_t)data2[2]) << 16;
    case 2:
      h ^= ((uint64_t)data2[1]) << 8;
    case 1:
      h ^= ((uint64_t)data2[0]);
      h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}
