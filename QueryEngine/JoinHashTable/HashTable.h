/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

enum class HashType : int { OneToOne, OneToMany, ManyToMany };

struct DecodedJoinHashBufferEntry {
  std::vector<int64_t> key;
  std::set<int32_t> payload;

  bool operator<(const DecodedJoinHashBufferEntry& other) const {
    return std::tie(key, payload) < std::tie(other.key, other.payload);
  }

  bool operator==(const DecodedJoinHashBufferEntry& other) const {
    return key == other.key && payload == other.payload;
  }
};

class HashTableEntryInfo {
 public:
  HashTableEntryInfo(size_t num_hash_entries,
                     size_t num_keys,
                     size_t rowid_size_in_bytes,
                     HashType layout,
                     bool for_window_framing = false)
      : num_hash_entries_(num_hash_entries)
      , num_keys_(num_keys)
      , rowid_size_in_bytes_(rowid_size_in_bytes)
      , layout_(layout)
      , for_window_framing_(for_window_framing) {}
  virtual ~HashTableEntryInfo() = default;
  virtual size_t computeTotalNumSlots() const = 0;
  virtual size_t computeHashTableSize() const = 0;

  size_t getNumHashEntries() const { return num_hash_entries_; }
  size_t getNumKeys() const { return num_keys_; }
  size_t getRowIdSizeInBytes() const { return rowid_size_in_bytes_; }
  HashType getHashTableLayout() const { return layout_; }
  void setNumHashEntries(size_t num_hash_entries) {
    num_hash_entries_ = num_hash_entries;
  }
  void setNumKeys(size_t num_keys) { num_keys_ = num_keys; }
  void setRowIdSizeInBytes(size_t rowid_size_in_bytes) {
    rowid_size_in_bytes_ = rowid_size_in_bytes;
  }
  void setHashTableLayout(HashType layout) { layout_ = layout; }
  bool forWindowFraming() const { return for_window_framing_; }

 protected:
  size_t num_hash_entries_;
  size_t num_keys_;
  size_t rowid_size_in_bytes_;
  HashType layout_;
  bool for_window_framing_;
};

using DecodedJoinHashBufferSet = std::set<DecodedJoinHashBufferEntry>;

class HashTable {
 public:
  virtual ~HashTable() {}

  virtual size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const = 0;

  virtual int8_t* getCpuBuffer() = 0;
  virtual int8_t* getGpuBuffer() const = 0;
  virtual HashType getLayout() const = 0;

  virtual size_t getEntryCount() const = 0;
  virtual size_t getEmittedKeysCount() const = 0;
  virtual size_t getRowIdSize() const = 0;

  //! Decode hash table into a std::set for easy inspection and validation.
  static DecodedJoinHashBufferSet toSet(
      size_t key_component_count,  // number of key parts
      size_t key_component_width,  // width of a key part
      size_t entry_count,          // number of hashable entries
      const int8_t* ptr1,          // hash entries
      const int8_t* ptr2,          // offsets
      const int8_t* ptr3,          // counts
      const int8_t* ptr4,          // payloads (rowids)
      size_t buffer_size);

  //! Decode hash table into a human-readable string.
  static std::string toString(
      const std::string& type,         // perfect, keyed, or geo
      const std::string& layout_type,  // one-to-one, one-to-many, many-to-many
      size_t key_component_count,      // number of key parts
      size_t key_component_width,      // width of a key part
      size_t entry_count,              // number of hashable entries
      const int8_t* ptr1,              // hash entries
      const int8_t* ptr2,              // offsets
      const int8_t* ptr3,              // counts
      const int8_t* ptr4,              // payloads (rowids)
      size_t buffer_size,
      bool raw = false);
};
