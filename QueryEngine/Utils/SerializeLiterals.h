/*
 * Copyright 2024 HEAVY.AI, Inc.
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
 * @file   SerializeLiterals.h
 * @brief  Helper classes to serialize literals used by Executor::serializeLiterals().
 */

#pragma once

#include <Logger/Logger.h>

#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

namespace heavyai {
namespace serialize_literals {

/// Calculate memory requirements and structure of serialized literals.
/// See Executor::serializeLiterals() for more info.
class MemoryRequirements {
  // One VarKey for each LiteralValue variant type (std::variant::index())
  // except std::pair<std::vector<int8_t>, int> uses two (32 and 64).
  using VarKey = size_t;
  using VarlenMap = std::unordered_map<VarKey, size_t>;

  using TypeSize = size_t;
  using VarlenTypeSizePair = std::pair<VarKey, TypeSize>;
  /// Order by (TypeSize DESC, VarKey ASC).
  struct ByTypeSizeDescVarKeyAsc {
    bool operator()(VarlenTypeSizePair const a, VarlenTypeSizePair const b) const {
      return a.second > b.second || (a.second == b.second && a.first < b.first);
    }
  };
  using VarlenTypeSizes = std::set<VarlenTypeSizePair, ByTypeSizeDescVarKeyAsc>;

 public:
  MemoryRequirements() = default;
  MemoryRequirements(MemoryRequirements const&) = delete;
  MemoryRequirements& operator=(MemoryRequirements const&) = delete;

  size_t getHeaderSize() const { return header_size_; }

  size_t& getOffset(VarKey const var_key) {
    auto itr = varlen_offsets_.find(var_key);
    CHECK(itr != varlen_offsets_.end()) << var_key;
    return itr->second;
  }

  // operator() called for each LiteralValue in order it will appear in the Header.
  // Accomplishes two things:
  //  * Increases header_size_ to calculate the final Header size.
  //  * The call to std::visit() updates varlen_sizes_ and varlen_type_sizes_ used to
  //    calculate memory requirements for the variable sized content.
  void operator()(CgenState::LiteralValue const& lit) {
    size_t const lit_bytes = std::visit(CgenState::LiteralBytes{}, lit);
    header_size_ = CgenState::align(header_size_, lit_bytes) + lit_bytes;
    variant_index_ = lit.index();
    std::visit(*this, lit);
  }

  // operator()s called by std::visit().
  void operator()(int8_t) {}
  void operator()(int16_t) {}
  void operator()(int32_t) {}
  void operator()(int64_t) {}
  void operator()(float) {}
  void operator()(double) {}
  void operator()(std::pair<std::string, shared::StringDictKey> const&) {}
  void operator()(std::string const& v) {
    varlen_sizes_[variant_index_] += v.size();
    varlen_type_sizes_.emplace(variant_index_, sizeof(char));
  }
  template <typename T>
  void operator()(std::vector<T> const& v) {
    varlen_sizes_[variant_index_] += v.size();
    varlen_type_sizes_.emplace(variant_index_, sizeof(T));
  }
  void operator()(std::pair<std::vector<int8_t>, int> const& v) {
    CHECK(v.second == 64 || v.second == 32) << v.second;
    size_t const type_size = v.second / 8u;  // convert number of bits to bytes
    // v.first should contain a whole number of 32 or 64-bit values.
    CHECK_EQ(0u, v.first.size() % type_size) << v.first.size() << ' ' << v.second;
    static_assert(std::variant_size_v<CgenState::LiteralValue> <= 32u);
    // We can safely use v.second as a VarKey.
    // See above descriptions of VarKey and varlen_sizes_ for more info.
    VarKey const var_key = static_cast<VarKey>(v.second);
    varlen_sizes_[var_key] += v.first.size() / type_size;  // Number of elements
    varlen_type_sizes_.emplace(var_key, type_size);
  }

  /// Set varlen_offsets_ based on varlen_sizes_ and varlen_type_sizes_.
  /// Return the final offset = size in bytes of the entire serialized vector.
  size_t setVarlenOffsets() {
    size_t offset = header_size_;  // Offset of Variable Content is at end of the Header.
    CHECK(varlen_offsets_.empty());
    CHECK_EQ(varlen_sizes_.size(), varlen_type_sizes_.size());
    // varlen_type_sizes_ is ordered by (type_size DESC, VarKey ASC). This ordering
    // eliminates all alignment gaps in the Variable Content section.
    for (auto const& [var_key, type_size] : varlen_type_sizes_) {
      offset = CgenState::align(offset, type_size);
      varlen_offsets_.emplace(var_key, offset);
      // Increase byte offset to the end of the current varlen content
      // and the beginning of the next.
      offset += type_size * varlen_sizes_[var_key];
    }
    CHECK_EQ(varlen_sizes_.size(), varlen_offsets_.size());
    return offset;  // Final value is size of Serializer::serialized_.
  }

 private:
  size_t header_size_;  // in bytes

  /// varlen_sizes_ tracks the total size (number of elements, not bytes) of all variable
  /// length content by VarKey. Example: two separate literal arrays of type double have
  /// lengths 100 and 200. Then varlen_sizes_[5] == 300 where 5=variant index of double.
  /// Calling setVarlenOffsets() uses varlen_sizes_ and varlen_type_sizes_ to
  /// calculate varlen_offsets_ which are byte offsets of the string/array in the
  /// serialized_ output vector.
  VarlenMap varlen_sizes_;    // value in units of the type size
  VarlenMap varlen_offsets_;  // value in units of bytes

  /// Set of (var_key, type_size) pairs.  Used by setVarlenOffsets() for
  /// ordering the variable-length content by (type_size DESC, var_key ASC).
  VarlenTypeSizes varlen_type_sizes_;

  /// Set for each call to std::visit().
  VarKey variant_index_;
};

/// Serialize all literals into one std::vector<int8_t>.
class Serializer {
 public:
  Serializer(Executor const& executor, MemoryRequirements& memory_requirements)
      : executor_(executor)
      , memory_requirements_(memory_requirements)
      , serialized_(memory_requirements_.setVarlenOffsets())
      , header_index_(0u)
      , max_offset_(memory_requirements_.getHeaderSize()) {
    constexpr size_t max_header_size = size_t(std::numeric_limits<int32_t>::max());
    if (max_header_size < memory_requirements_.getHeaderSize()) {
      throw TooManyLiterals();
    }
  }

  Serializer(Serializer const&) = delete;
  Serializer& operator=(Serializer const&) = delete;

  size_t getMaxOffset() const { return max_offset_; }

  std::vector<int8_t>& getSerializedVector() { return serialized_; }

  // operator() called for each LiteralValue in the same order as was invoked
  // for MemoryRequirements.
  void operator()(CgenState::LiteralValue const& lit) {
    variant_index_ = lit.index();
    lit_bytes = std::visit(CgenState::LiteralBytes{}, lit);
    header_index_ = CgenState::align(header_index_, lit_bytes);
    std::visit(*this, lit);
    header_index_ += lit_bytes;
  }

  // operator()s called by std::visit().
  void operator()(int8_t v) { serialized_[header_index_] = v; }
  void operator()(int16_t v) { memcpy(&serialized_[header_index_], &v, lit_bytes); }
  void operator()(int32_t v) { memcpy(&serialized_[header_index_], &v, lit_bytes); }
  void operator()(int64_t v) { memcpy(&serialized_[header_index_], &v, lit_bytes); }
  void operator()(float v) { memcpy(&serialized_[header_index_], &v, lit_bytes); }
  void operator()(double v) { memcpy(&serialized_[header_index_], &v, lit_bytes); }
  void operator()(std::pair<std::string, shared::StringDictKey> const& v) {
    constexpr bool with_generation = true;
    auto* const sdp = executor_.getStringDictionaryProxy(v.second, with_generation);
    auto const str_id = g_enable_string_functions ? sdp->getOrAddTransient(v.first)
                                                  : sdp->getIdOfString(v.first);
    memcpy(&serialized_[header_index_], &str_id, lit_bytes);
  }
  void operator()(std::string const& v) {
    size_t& offset = memory_requirements_.getOffset(variant_index_);
    int32_t const off_and_len = pack(offset, v.size());
    memcpy(&serialized_[header_index_], &off_and_len, lit_bytes);
    memcpy(&serialized_[offset], v.data(), v.size());
    offset += v.size();
    max_offset_ = std::max(max_offset_, offset);
  }
  template <typename T>
  void operator()(std::vector<T> const& v) {
    size_t& offset = memory_requirements_.getOffset(variant_index_);
    int32_t const off_and_len = pack(offset, v.size());
    size_t const size_bytes = v.size() * sizeof(T);
    memcpy(&serialized_[header_index_], &off_and_len, lit_bytes);
    memcpy(&serialized_[offset], v.data(), size_bytes);
    offset += size_bytes;
    max_offset_ = std::max(max_offset_, offset);
  }
  void operator()(std::pair<std::vector<int8_t>, int> const& v) {
    size_t& offset = memory_requirements_.getOffset(v.second);
    int32_t const off_and_len = pack(offset, v.first.size());
    memcpy(&serialized_[header_index_], &off_and_len, lit_bytes);
    memcpy(&serialized_[offset], v.first.data(), v.first.size());
    offset += v.first.size();
    max_offset_ = std::max(max_offset_, offset);
  }

 private:
  Executor const& executor_;
  MemoryRequirements& memory_requirements_;

  std::vector<int8_t> serialized_;  // This is the final result.
  size_t header_index_;             // Index into serialized_.
  size_t max_offset_;               // Used for error checking.

  // Variables set prior to each call to std::visit().
  size_t variant_index_;
  size_t lit_bytes;

  // Pack offset and length into a int32_t value.
  static int32_t pack(size_t const offset, size_t const length) {
    rangeCheck<uint16_t>(offset, "Offset");
    rangeCheck<uint16_t>(length, "Length");
    auto const off_and_len = int32_t(offset) << 16 | int32_t(length);
    CHECK(off_and_len);
    return off_and_len;
  }

  template <typename T>
  static void rangeCheck(size_t const n, std::string const name) {
    if (size_t(std::numeric_limits<T>::max()) < n) {
      throw TooManyLiterals(name + ' ' + std::to_string(n) +
                            " exceeds bounds on literal values.");
    }
  }
};

}  // namespace serialize_literals
}  // namespace heavyai
