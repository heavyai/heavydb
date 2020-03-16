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
#include "BaselineJoinHashTable.h"
#include "ColumnFetcher.h"
#include "EquiJoinCondition.h"
#include "JoinHashTable.h"
#include "OverlapsJoinHashTable.h"
#include "RuntimeFunctions.h"
#include "ScalarExprVisitor.h"

namespace {

template <typename T>
void innerToString(const int8_t* ptr1,
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
std::string JoinHashTableInterface::toString(
    const std::string& type,     // perfect, keyed, or geo
    size_t key_component_count,  // number of key parts
    size_t key_component_width,  // width of a key part
    size_t entry_count,          // number of hashable entries
    const int8_t* ptr1,          // keys
    const int8_t* ptr2,          // offsets
    const int8_t* ptr3,          // counts
    const int8_t* ptr4,          // payloads (rowids)
    size_t buffer_size,          // total memory size
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
    txt += " one-to-one";
  } else if (have_offsets && have_counts) {
    txt += " one-to-many";
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
      innerToString<int64_t>(ptr1, entry_count, key_component_count, raw, txt);
    } else if (key_component_width == 4) {
      innerToString<int32_t>(ptr1, entry_count, key_component_count, raw, txt);
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

namespace {

template <typename T>
std::string toStringFlat(const JoinHashTableInterface* hash_table,
                         const ExecutorDeviceType device_type,
                         const int device_id) {
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
                                                   const int device_id) const {
  return toStringFlat<int64_t>(this, device_type, device_id);
}

std::string JoinHashTableInterface::toStringFlat32(const ExecutorDeviceType device_type,
                                                   const int device_id) const {
  return toStringFlat<int32_t>(this, device_type, device_id);
}

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferEntry& e) {
  os << "  {{";
  bool first = true;
  for (auto k : e.key) {
    if (!first) {
      os << ",";
    } else {
      first = false;
    }
    os << k;
  }
  os << "}, ";
  os << "{";
  first = true;
  for (auto p : e.payload) {
    if (!first) {
      os << ", ";
    } else {
      first = false;
    }
    os << p;
  }
  os << "}}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferSet& s) {
  os << "{\n";
  bool first = true;
  for (auto e : s) {
    if (!first) {
      os << ",\n";
    } else {
      first = false;
    }
    os << e;
  }
  if (!s.empty()) {
    os << "\n";
  }
  os << "}\n";
  return os;
}

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
DecodedJoinHashBufferSet JoinHashTableInterface::toSet(
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

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<JoinHashTableInterface> JoinHashTableInterface::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<JoinHashTableInterface> join_hash_table;
  CHECK_GT(device_count, 0);
  if (!g_enable_overlaps_hashjoin && qual_bin_oper->is_overlaps_oper()) {
    throw std::runtime_error(
        "Overlaps hash join disabled, attempting to fall back to loop join");
  }
  if (qual_bin_oper->is_overlaps_oper()) {
    VLOG(1) << "Trying to build geo hash table:";
    join_hash_table = OverlapsJoinHashTable::getInstance(
        qual_bin_oper, query_infos, memory_level, device_count, column_cache, executor);
  } else if (dynamic_cast<const Analyzer::ExpressionTuple*>(
                 qual_bin_oper->get_left_operand())) {
    VLOG(1) << "Trying to build keyed hash table:";
    join_hash_table = BaselineJoinHashTable::getInstance(qual_bin_oper,
                                                         query_infos,
                                                         memory_level,
                                                         preferred_hash_type,
                                                         device_count,
                                                         column_cache,
                                                         executor);
  } else {
    try {
      VLOG(1) << "Trying to build perfect hash table:";
      join_hash_table = JoinHashTable::getInstance(qual_bin_oper,
                                                   query_infos,
                                                   memory_level,
                                                   preferred_hash_type,
                                                   device_count,
                                                   column_cache,
                                                   executor);
    } catch (TooManyHashEntries&) {
      const auto join_quals = coalesce_singleton_equi_join(qual_bin_oper);
      CHECK_EQ(join_quals.size(), size_t(1));
      const auto join_qual =
          std::dynamic_pointer_cast<Analyzer::BinOper>(join_quals.front());
      VLOG(1) << "Trying to build keyed hash table after perfect hash table:";
      join_hash_table = BaselineJoinHashTable::getInstance(join_qual,
                                                           query_infos,
                                                           memory_level,
                                                           preferred_hash_type,
                                                           device_count,
                                                           column_cache,
                                                           executor);
    }
  }
  CHECK(join_hash_table);
  if (VLOGGING(2)) {
    if (join_hash_table->getMemoryLevel() == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      for (int device_id = 0; device_id < device_count; ++device_id) {
        if (join_hash_table->getJoinHashBufferSize(ExecutorDeviceType::GPU, device_id) <=
            1000) {
          VLOG(2) << "Built GPU hash table: "
                  << join_hash_table->toString(ExecutorDeviceType::GPU, device_id);
        }
      }
    } else {
      if (join_hash_table->getJoinHashBufferSize(ExecutorDeviceType::CPU) <= 1000) {
        VLOG(2) << "Build CPU hash table: "
                << join_hash_table->toString(ExecutorDeviceType::CPU);
      }
    }
  }
  return join_hash_table;
}

std::shared_ptr<Analyzer::ColumnVar> getSyntheticColumnVar(std::string_view table,
                                                           std::string_view column,
                                                           int rte_idx,
                                                           Executor* executor) {
  auto catalog = executor->getCatalog();
  CHECK(catalog);

  auto tmeta = catalog->getMetadataForTable(std::string(table));
  CHECK(tmeta);

  auto cmeta = catalog->getMetadataForColumn(tmeta->tableId, std::string(column));
  CHECK(cmeta);

  auto ti = cmeta->columnType;

  if (ti.is_geometry() && ti.get_type() != kPOINT) {
    int geoColumnId{0};
    switch (ti.get_type()) {
      case kLINESTRING: {
        geoColumnId = cmeta->columnId + 2;
        break;
      }
      case kPOLYGON: {
        geoColumnId = cmeta->columnId + 3;
        break;
      }
      case kMULTIPOLYGON: {
        geoColumnId = cmeta->columnId + 4;
        break;
      }
      default:
        CHECK(false);
    }
    cmeta = catalog->getMetadataForColumn(tmeta->tableId, geoColumnId);
    CHECK(cmeta);
    ti = cmeta->columnType;
  }

  auto cv =
      std::make_shared<Analyzer::ColumnVar>(ti, tmeta->tableId, cmeta->columnId, rte_idx);
  return cv;
}

class AllColumnVarsVisitor
    : public ScalarExprVisitor<std::set<const Analyzer::ColumnVar*>> {
 protected:
  std::set<const Analyzer::ColumnVar*> visitColumnVar(
      const Analyzer::ColumnVar* column) const override {
    return {column};
  }

  std::set<const Analyzer::ColumnVar*> visitColumnVarTuple(
      const Analyzer::ExpressionTuple* expr_tuple) const override {
    AllColumnVarsVisitor visitor;
    std::set<const Analyzer::ColumnVar*> result;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      const auto component_rte_set = visitor.visit(expr_component.get());
      result.insert(component_rte_set.begin(), component_rte_set.end());
    }
    return result;
  }

  std::set<const Analyzer::ColumnVar*> aggregateResult(
      const std::set<const Analyzer::ColumnVar*>& aggregate,
      const std::set<const Analyzer::ColumnVar*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

void setupSyntheticCaching(std::set<const Analyzer::ColumnVar*> cvs, Executor* executor) {
  std::unordered_set<int> phys_table_ids;
  for (auto cv : cvs) {
    phys_table_ids.insert(cv->get_table_id());
  }

  std::unordered_set<PhysicalInput> phys_inputs;
  for (auto cv : cvs) {
    phys_inputs.emplace(PhysicalInput{cv->get_column_id(), cv->get_table_id()});
  }

  executor->setupCaching(phys_inputs, phys_table_ids);
}

std::vector<InputTableInfo> getSyntheticInputTableInfo(
    std::set<const Analyzer::ColumnVar*> cvs,
    Executor* executor) {
  auto catalog = executor->getCatalog();
  CHECK(catalog);

  std::unordered_set<int> phys_table_ids;
  for (auto cv : cvs) {
    phys_table_ids.insert(cv->get_table_id());
  }

  // NOTE(sy): This vector ordering seems to work for now, but maybe we need to
  // review how rte_idx is assigned for ColumnVars. See for example Analyzer.h
  // and RelAlgExecutor.cpp and rte_idx there.
  std::vector<InputTableInfo> query_infos(phys_table_ids.size());
  size_t i = 0;
  for (auto id : phys_table_ids) {
    auto tmeta = catalog->getMetadataForTable(id);
    query_infos[i].table_id = id;
    query_infos[i].info = tmeta->fragmenter->getFragmentsForQuery();
    ++i;
  }

  return query_infos;
}

//! Make hash table from named tables and columns (such as for testing).
std::shared_ptr<JoinHashTableInterface> JoinHashTableInterface::getSyntheticInstance(
    std::string_view table1,
    std::string_view column1,
    std::string_view table2,
    std::string_view column2,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  auto a1 = getSyntheticColumnVar(table1, column1, 0, executor);
  auto a2 = getSyntheticColumnVar(table2, column2, 1, executor);

  auto qual_bin_oper = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, a1, a2);

  std::set<const Analyzer::ColumnVar*> cvs =
      AllColumnVarsVisitor().visit(qual_bin_oper.get());
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(cvs, executor);

  auto hash_table = JoinHashTableInterface::getInstance(qual_bin_oper,
                                                        query_infos,
                                                        memory_level,
                                                        preferred_hash_type,
                                                        device_count,
                                                        column_cache,
                                                        executor);
  return hash_table;
}

//! Make hash table from named tables and columns (such as for testing).
std::shared_ptr<JoinHashTableInterface> JoinHashTableInterface::getSyntheticInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  std::set<const Analyzer::ColumnVar*> cvs =
      AllColumnVarsVisitor().visit(qual_bin_oper.get());
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(cvs, executor);

  auto hash_table = JoinHashTableInterface::getInstance(qual_bin_oper,
                                                        query_infos,
                                                        memory_level,
                                                        preferred_hash_type,
                                                        device_count,
                                                        column_cache,
                                                        executor);
  return hash_table;
}
