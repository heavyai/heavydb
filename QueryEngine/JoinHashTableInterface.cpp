/*
 * Copyright 2019 MapD Technologies, Inc.
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

#include "JoinHashTableInterface.h"
#include "QueryEngine/RuntimeFunctions.h"

namespace {

template <typename T>
void innerDecodeJoinHashBufferToString(const int8_t* ptr1,
                                       size_t entry_count,
                                       size_t key_component_count,
                                       bool raw,
                                       std::string& txt) {
  auto empty = get_empty_key<T>();
  auto ptr = reinterpret_cast<const T*>(ptr1);
  for (size_t e = 0; e < entry_count; ++e, ptr += key_component_count) {
    if (e != 0) {
      txt += " ";
    }
    if (*ptr == empty && !raw) {
      txt += "*";  // null hash table entry
    } else if (*ptr == empty - 1 && !raw) {
      txt += "?";  // write_pending (should never happen here)
    } else {
      txt += "(";
      for (size_t j = 0; j < key_component_count; ++j) {
        if (j != 0) {
          txt += ",";
        }
        txt += std::to_string(ptr[j]);
      }
      txt += ")";
    }
  }
}

}  // anonymous namespace

std::string decodeJoinHashBufferToString(
    size_t key_component_count,  // number of key parts
    size_t key_component_width,  // width of a key part
    const int8_t* ptr1,          // keys
    const int8_t* ptr2,          // offsets
    const int8_t* ptr3,          // counts
    const int8_t* ptr4,          // payloads (rowids)
    size_t buffer_size,          // total memory size
    bool raw) {
  std::string txt;

  CHECK(key_component_width == 8 || key_component_width == 4);

  auto i64ptr1 = reinterpret_cast<const int64_t*>(ptr1);
  auto i64ptr2 = reinterpret_cast<const int64_t*>(ptr2);
  auto i32ptr2 = reinterpret_cast<const int32_t*>(ptr2);
  auto i32ptr3 = reinterpret_cast<const int32_t*>(ptr3);
  auto i32ptr4 = reinterpret_cast<const int32_t*>(ptr4);
  auto i32ptr5 = reinterpret_cast<const int32_t*>(ptr1 + buffer_size);
  auto i64ptr5 = reinterpret_cast<const int64_t*>(i32ptr5);

  CHECK_LE(i64ptr1, i64ptr2);
  CHECK_LT(i64ptr2, i64ptr5);
  CHECK_LT(i32ptr2, i32ptr3);
  CHECK_LT(i32ptr3, i32ptr4);
  CHECK_LT(i32ptr4, i32ptr5);

  size_t entry_count = (ptr3 - ptr2) / sizeof(int32_t);

  // first section: keys
  if (i64ptr1 < i64ptr2) {
    if (key_component_width == 8) {
      innerDecodeJoinHashBufferToString<int64_t>(
          ptr1, entry_count, key_component_count, raw, txt);
    } else if (key_component_width == 4) {
      innerDecodeJoinHashBufferToString<int32_t>(
          ptr1, entry_count, key_component_count, raw, txt);
    }

    txt += " | ";
  }

  // second section: offsets
  for (size_t i = 0; &i32ptr2[i] < i32ptr3; ++i) {
    if (i != 0) {
      txt += " ";
    }
    if (i32ptr2[i] == -1) {
      txt += "*";  // null
    } else {
      txt += std::to_string(i32ptr2[i]);
    }
  }

  txt += " | ";

  // third section: counts
  for (size_t i = 0; &i32ptr3[i] < i32ptr4; ++i) {
    if (i != 0) {
      txt += " ";
    }
    if (i32ptr3[i] == 0) {
      txt += "*";  // null
    } else {
      txt += std::to_string(i32ptr3[i]);
    }
  }

  txt += " | ";

  // fourth section: payloads (rowids)
  for (size_t i = 0; &i32ptr4[i] < i32ptr5; ++i) {
    if (i != 0) {
      txt += " ";
    }
    if (i32ptr4[i] == -1) {
      txt += "*";  // null
    } else {
      txt += std::to_string(i32ptr4[i]);
    }
  }

  return txt;
}

namespace {

template <typename T>
std::string decodeJoinHashBufferToStringFlat(const JoinHashTableInterface* hash_table,
                                             const ExecutorDeviceType device_type,
                                             const int device_id) noexcept {
  auto mem =
      reinterpret_cast<const T*>(hash_table->getJoinHashBuffer(device_type, device_id));
  auto memsz = hash_table->getJoinHashBufferSize(device_type, device_id) / sizeof(T);
  std::string txt;
  for (size_t i = 0; i < memsz; ++i) {
    if (i > 0) {
      txt += ", ";
    }
    txt += std::to_string(mem[i]);
  }
  return txt;
}

}  // anonymous namespace

std::string JoinHashTableInterface::toStringFlat64(const ExecutorDeviceType device_type,
                                                   const int device_id) const noexcept {
  return decodeJoinHashBufferToStringFlat<int64_t>(this, device_type, device_id);
}

std::string JoinHashTableInterface::toStringFlat32(const ExecutorDeviceType device_type,
                                                   const int device_id) const noexcept {
  return decodeJoinHashBufferToStringFlat<int32_t>(this, device_type, device_id);
}

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferEntry& e) {
  os << "(";
  bool first = true;
  for (auto k : e.key) {
    if (!first) {
      os << ",";
    } else {
      first = false;
    }
    os << k;
  }
  os << ")";
  os << ": ";
  first = true;
  for (auto p : e.payload) {
    if (!first) {
      os << " ";
    } else {
      first = false;
    }
    os << p;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const std::set<DecodedJoinHashBufferEntry>& s) {
  for (auto e : s) {
    os << e << "\n";
  }
  return os;
}

namespace {

template <typename T>
void innerDecodeJoinHashBuffer(const int8_t* ptr1,
                               const int32_t* ptr2,
                               const int32_t* ptr3,
                               const int32_t* ptr4,
                               size_t entry_count,
                               size_t key_component_count,
                               std::set<DecodedJoinHashBufferEntry>& s) {
  auto empty = get_empty_key<T>();
  auto ptr = reinterpret_cast<const T*>(ptr1);
  for (size_t e = 0; e < entry_count; ++e, ptr += key_component_count) {
    if (*ptr == empty) {
      continue;
    }

    std::vector<int64_t> key;
    for (size_t j = 0; j < key_component_count; ++j) {
      key.push_back(ptr[j]);
    }

    int32_t offset = ptr2[e];

    int32_t count = ptr3[e];

    std::set<int32_t> payload;
    for (size_t j = 0; j < static_cast<size_t>(count); ++j) {
      payload.insert(ptr4[offset + j]);
    }

    s.insert({std::move(key), std::move(payload)});
  }
}

template <typename T>
void innerDecodeJoinHashBuffer(const int32_t* ptr2,
                               const int32_t* ptr3,
                               const int32_t* ptr4,
                               size_t entry_count,
                               std::set<DecodedJoinHashBufferEntry>& s) {
  auto empty = -1;
  auto ptr = reinterpret_cast<const T*>(ptr2);
  for (size_t e = 0; e < entry_count; ++e, ++ptr) {
    if (*ptr == empty) {
      continue;
    }

    std::vector<int64_t> key;
    key.push_back(e);

    int32_t offset = ptr2[e];

    int32_t count = ptr3[e];

    std::set<int32_t> payload;
    for (size_t j = 0; j < static_cast<size_t>(count); ++j) {
      payload.insert(ptr4[offset + j]);
    }

    s.insert({std::move(key), std::move(payload)});
  }
}

}  // anonymous namespace

std::set<DecodedJoinHashBufferEntry> decodeJoinHashBuffer(
    size_t key_component_count,  // number of key parts
    size_t key_component_width,  // width of a key part
    const int8_t* ptr1,          // keys
    const int8_t* ptr2,          // offsets
    const int8_t* ptr3,          // counts
    const int8_t* ptr4,          // payloads (rowids)
    size_t buffer_size) {        // total memory size
  std::set<DecodedJoinHashBufferEntry> s;

  CHECK(key_component_width == 8 || key_component_width == 4);

  auto i64ptr1 = reinterpret_cast<const int64_t*>(ptr1);
  auto i64ptr2 = reinterpret_cast<const int64_t*>(ptr2);
  auto i32ptr2 = reinterpret_cast<const int32_t*>(ptr2);
  auto i32ptr3 = reinterpret_cast<const int32_t*>(ptr3);
  auto i32ptr4 = reinterpret_cast<const int32_t*>(ptr4);
  auto i32ptr5 = reinterpret_cast<const int32_t*>(ptr1 + buffer_size);
  auto i64ptr5 = reinterpret_cast<const int64_t*>(i32ptr5);

  CHECK_LE(i64ptr1, i64ptr2);
  CHECK_LT(i64ptr2, i64ptr5);
  CHECK_LT(i32ptr2, i32ptr3);
  CHECK_LT(i32ptr3, i32ptr4);
  CHECK_LT(i32ptr4, i32ptr5);

  size_t entry_count = (ptr3 - ptr2) / sizeof(int32_t);

  if (i64ptr1 < i64ptr2) {  // BaselineJoinHashTable or OverlapsJoinHashTable
    if (key_component_width == 8) {
      innerDecodeJoinHashBuffer<int64_t>(
          ptr1, i32ptr2, i32ptr3, i32ptr4, entry_count, key_component_count, s);
    } else if (key_component_width == 4) {
      innerDecodeJoinHashBuffer<int32_t>(
          ptr1, i32ptr2, i32ptr3, i32ptr4, entry_count, key_component_count, s);
    }
  } else {  // JoinHashTable
    if (key_component_width == 8) {
      innerDecodeJoinHashBuffer<int64_t>(i32ptr2, i32ptr3, i32ptr4, entry_count, s);
    } else if (key_component_width == 4) {
      innerDecodeJoinHashBuffer<int32_t>(i32ptr2, i32ptr3, i32ptr4, entry_count, s);
    }
  }

  return s;
}
