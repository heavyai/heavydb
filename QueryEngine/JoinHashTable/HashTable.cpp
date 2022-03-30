/*
 * Copyright 2020 OmniSci, Inc.
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

#include "QueryEngine/JoinHashTable/HashTable.h"

#include "Logger/Logger.h"
#include "QueryEngine/RuntimeFunctions.h"

namespace {

namespace perfect_hash {

void to_set_one_to_one(const int32_t* const ptr4,
                       size_t entry_count,
                       DecodedJoinHashBufferSet& s) {
  const auto empty = -1;
  auto ptr = ptr4;
  for (size_t e = 0; e < entry_count; ++e, ++ptr) {
    if (*ptr == empty) {
      continue;
    }

    decltype(DecodedJoinHashBufferEntry::key) key;
    key.push_back(e);

    decltype(DecodedJoinHashBufferEntry::payload) payload;
    payload.insert(*ptr);

    s.insert({std::move(key), std::move(payload)});
  }
}

void to_set_one_to_many(const int32_t* const ptr2,
                        const int32_t* const ptr3,
                        const int32_t* const ptr4,
                        size_t entry_count,
                        DecodedJoinHashBufferSet& s) {
  auto empty = -1;
  auto ptr = ptr2;
  for (size_t e = 0; e < entry_count; ++e, ++ptr) {
    if (*ptr == empty) {
      continue;
    }

    decltype(DecodedJoinHashBufferEntry::key) key;
    key.push_back(e);

    int32_t offset = ptr2[e];

    int32_t count = ptr3[e];

    decltype(DecodedJoinHashBufferEntry::payload) payload;
    for (size_t j = 0; j < static_cast<size_t>(count); ++j) {
      payload.insert(ptr4[offset + j]);
    }

    s.insert({std::move(key), std::move(payload)});
  }
}

}  // namespace perfect_hash

namespace keyed_hash {

template <typename T>
void to_set_one_to_one(const int8_t* ptr1,
                       size_t entry_count,
                       size_t key_component_count,
                       DecodedJoinHashBufferSet& s) {
  auto empty = get_empty_key<T>();
  auto ptr = reinterpret_cast<const T*>(ptr1);
  for (size_t e = 0; e < entry_count; ++e, ptr += key_component_count) {
    if (*ptr == empty) {
      continue;
    }

    std::vector<int64_t> key;
    size_t j = 0;
    for (; j < key_component_count - 1; ++j) {
      key.push_back(ptr[j]);
    }

    std::set<int32_t> payload;
    payload.insert(ptr[j]);

    s.insert({std::move(key), std::move(payload)});
  }
}

template <typename T>
void to_set_one_to_many(const int8_t* ptr1,
                        const int32_t* const ptr2,
                        const int32_t* const ptr3,
                        const int32_t* const ptr4,
                        size_t entry_count,
                        size_t key_component_count,
                        DecodedJoinHashBufferSet& s) {
  auto empty = get_empty_key<T>();
  auto ptr = reinterpret_cast<const T*>(ptr1);
  for (size_t e = 0; e < entry_count; ++e, ptr += key_component_count) {
    if (*ptr == empty) {
      continue;
    }

    std::vector<int64_t> key;
    size_t j = 0;
    for (; j < key_component_count - 1; ++j) {
      key.push_back(ptr[j]);
    }

    int32_t offset = ptr2[e];

    int32_t count = ptr3[e];

    decltype(DecodedJoinHashBufferEntry::payload) payload;
    for (size_t j = 0; j < static_cast<size_t>(count); ++j) {
      payload.insert(ptr4[offset + j]);
    }

    s.insert({std::move(key), std::move(payload)});
  }
}

}  // namespace keyed_hash

}  // anonymous namespace

//! Decode hash table into a std::set for easy inspection and validation.
DecodedJoinHashBufferSet HashTable::toSet(
    size_t key_component_count,  // number of key parts
    size_t key_component_width,  // width of a key part
    size_t entry_count,          // number of hashable entries
    const int8_t* ptr1,          // keys
    const int8_t* ptr2,          // offsets
    const int8_t* ptr3,          // counts
    const int8_t* ptr4,          // payloads (rowids)
    size_t buffer_size) {        // total memory size
  DecodedJoinHashBufferSet s;

  CHECK_LE(ptr1, ptr2);
  CHECK_LE(ptr2, ptr3);
  CHECK_LE(ptr3, ptr4);
  CHECK_LE(ptr4, ptr1 + buffer_size);

  bool have_keys = ptr2 > ptr1;
  bool have_offsets = ptr3 > ptr2;
  bool have_counts = ptr4 > ptr3;
  bool have_payloads = (ptr1 + buffer_size) > ptr4;

  auto i32ptr2 = reinterpret_cast<const int32_t*>(ptr2);
  auto i32ptr3 = reinterpret_cast<const int32_t*>(ptr3);
  auto i32ptr4 = reinterpret_cast<const int32_t*>(ptr4);

  if (have_keys) {  // BaselineJoinHashTable or OverlapsJoinHashTable
    CHECK(key_component_width == 8 || key_component_width == 4);
    if (key_component_width == 8) {
      if (!have_offsets && !have_counts) {
        keyed_hash::to_set_one_to_one<int64_t>(ptr1, entry_count, key_component_count, s);
      } else {
        keyed_hash::to_set_one_to_many<int64_t>(
            ptr1, i32ptr2, i32ptr3, i32ptr4, entry_count, key_component_count, s);
      }
    } else if (key_component_width == 4) {
      if (!have_offsets && !have_counts) {
        keyed_hash::to_set_one_to_one<int32_t>(ptr1, entry_count, key_component_count, s);
      } else {
        keyed_hash::to_set_one_to_many<int32_t>(
            ptr1, i32ptr2, i32ptr3, i32ptr4, entry_count, key_component_count, s);
      }
    }
  } else {  // JoinHashTable
    if (!have_offsets && !have_counts && have_payloads) {
      perfect_hash::to_set_one_to_one(i32ptr4, entry_count, s);
    } else {
      perfect_hash::to_set_one_to_many(i32ptr2, i32ptr3, i32ptr4, entry_count, s);
    }
  }

  return s;
}

namespace {

template <typename T>
void inner_to_string(const int8_t* ptr1,
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

//! Decode hash table into a human-readable string.
std::string HashTable::toString(
    const std::string& type,         // perfect, keyed
    const std::string& layout_type,  // one-to-one, one-to-many, many-to-many
    size_t key_component_count,      // number of key parts
    size_t key_component_width,      // width of a key part
    size_t entry_count,              // number of hashable entries
    const int8_t* ptr1,              // keys
    const int8_t* ptr2,              // offsets
    const int8_t* ptr3,              // counts
    const int8_t* ptr4,              // payloads (rowids)
    size_t buffer_size,              // total memory size
    bool raw) {
  std::string txt;

  CHECK(ptr1 <= ptr2);
  CHECK(ptr2 <= ptr3);
  CHECK(ptr3 <= ptr4);
  CHECK(ptr4 <= ptr1 + buffer_size);

  bool have_keys = ptr2 > ptr1;
  bool have_offsets = ptr3 > ptr2;
  bool have_counts = ptr4 > ptr3;
  bool have_payloads = (ptr1 + buffer_size) > ptr4;

  // table heading
  txt += "| " + type;
  if (!have_offsets && !have_counts) {
    txt += layout_type;
  } else if (have_offsets && have_counts) {
    txt += layout_type;
  } else {
    CHECK(false);
  }

  // first section: keys
  if (have_keys) {
    CHECK(key_component_width == 8 || key_component_width == 4);

    if (!txt.empty()) {
      txt += " ";
    }
    txt += "| keys ";

    if (key_component_width == 8) {
      inner_to_string<int64_t>(ptr1, entry_count, key_component_count, raw, txt);
    } else if (key_component_width == 4) {
      inner_to_string<int32_t>(ptr1, entry_count, key_component_count, raw, txt);
    }
  }

  // second section: offsets
  if (have_offsets) {
    if (!txt.empty()) {
      txt += " ";
    }
    txt += "| offsets ";

    auto i32ptr2 = reinterpret_cast<const int32_t*>(ptr2);
    auto i32ptr3 = reinterpret_cast<const int32_t*>(ptr3);
    for (size_t i = 0; &i32ptr2[i] < i32ptr3; ++i) {
      if (i != 0) {
        txt += " ";
      }
      if (i32ptr2[i] == -1 && !raw) {
        txt += "*";  // null
      } else {
        txt += std::to_string(i32ptr2[i]);
      }
    }
  }

  // third section: counts
  if (have_counts) {
    if (!txt.empty()) {
      txt += " ";
    }
    txt += "| counts ";

    auto i32ptr3 = reinterpret_cast<const int32_t*>(ptr3);
    auto i32ptr4 = reinterpret_cast<const int32_t*>(ptr4);
    for (size_t i = 0; &i32ptr3[i] < i32ptr4; ++i) {
      if (i != 0) {
        txt += " ";
      }
      if (i32ptr3[i] == 0 && !raw) {
        txt += "*";  // null
      } else {
        txt += std::to_string(i32ptr3[i]);
      }
    }
  }

  // fourth section: payloads (rowids)
  if (have_payloads) {
    if (!txt.empty()) {
      txt += " ";
    }
    txt += "| payloads ";

    auto i32ptr4 = reinterpret_cast<const int32_t*>(ptr4);
    auto i32ptr5 = reinterpret_cast<const int32_t*>(ptr1 + buffer_size);
    for (size_t i = 0; &i32ptr4[i] < i32ptr5; ++i) {
      if (i != 0) {
        txt += " ";
      }
      if (i32ptr4[i] == -1 && !raw) {
        txt += "*";  // null
      } else {
        txt += std::to_string(i32ptr4[i]);
      }
    }
  }

  if (!txt.empty()) {
    txt += " |";
  }
  return txt;
}
