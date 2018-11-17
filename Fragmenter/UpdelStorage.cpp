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
#include <algorithm>
#include <boost/variant.hpp>
#include <boost/variant/get.hpp>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "Catalog/Catalog.h"
#include "DataMgr/DataMgr.h"
#include "DataMgr/Encoder.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/thread_count.h"

namespace Fragmenter_Namespace {

void InsertOrderFragmenter::updateColumn(const Catalog_Namespace::Catalog* catalog,
                                         const std::string& tab_name,
                                         const std::string& col_name,
                                         const int fragment_id,
                                         const std::vector<uint64_t>& frag_offsets,
                                         const std::vector<ScalarTargetValue>& rhs_values,
                                         const SQLTypeInfo& rhs_type,
                                         const Data_Namespace::MemoryLevel memory_level,
                                         UpdelRoll& updel_roll) {
  const auto td = catalog->getMetadataForTable(tab_name);
  CHECK(td);
  const auto cd = catalog->getMetadataForColumn(td->tableId, col_name);
  CHECK(cd);
  td->fragmenter->updateColumn(catalog,
                               td,
                               cd,
                               fragment_id,
                               frag_offsets,
                               rhs_values,
                               rhs_type,
                               memory_level,
                               updel_roll);
}

inline bool is_integral(const SQLTypeInfo& t) {
  return t.is_integer() || t.is_boolean() || t.is_time() || t.is_timeinterval();
}

void InsertOrderFragmenter::updateColumn(const Catalog_Namespace::Catalog* catalog,
                                         const TableDescriptor* td,
                                         const ColumnDescriptor* cd,
                                         const int fragment_id,
                                         const std::vector<uint64_t>& frag_offsets,
                                         const ScalarTargetValue& rhs_value,
                                         const SQLTypeInfo& rhs_type,
                                         const Data_Namespace::MemoryLevel memory_level,
                                         UpdelRoll& updel_roll) {
  updateColumn(catalog,
               td,
               cd,
               fragment_id,
               frag_offsets,
               std::vector<ScalarTargetValue>(1, rhs_value),
               rhs_type,
               memory_level,
               updel_roll);
}

void InsertOrderFragmenter::updateColumn(const Catalog_Namespace::Catalog* catalog,
                                         const TableDescriptor* td,
                                         const ColumnDescriptor* cd,
                                         const int fragment_id,
                                         const std::vector<uint64_t>& frag_offsets,
                                         const std::vector<ScalarTargetValue>& rhs_values,
                                         const SQLTypeInfo& rhs_type,
                                         const Data_Namespace::MemoryLevel memory_level,
                                         UpdelRoll& updel_roll) {
  updel_roll.catalog = catalog;
  updel_roll.logicalTableId = catalog->getLogicalTableId(td->tableId);
  updel_roll.memoryLevel = memory_level;

  const auto nrow = frag_offsets.size();
  const auto n_rhs_values = rhs_values.size();
  if (0 == nrow) {
    return;
  }
  CHECK(nrow == n_rhs_values || 1 == n_rhs_values);

  auto fragment_it = std::find_if(
      fragmentInfoVec_.begin(), fragmentInfoVec_.end(), [=](FragmentInfo& f) -> bool {
        return f.fragmentId == fragment_id;
      });
  CHECK(fragment_it != fragmentInfoVec_.end());
  auto& fragment = *fragment_it;
  auto chunk_meta_it = fragment.getChunkMetadataMapPhysical().find(cd->columnId);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMapPhysical().end());
  ChunkKey chunk_key{
      catalog->get_currentDB().dbId, td->tableId, cd->columnId, fragment.fragmentId};
  auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                         &catalog->get_dataMgr(),
                                         chunk_key,
                                         Data_Namespace::CPU_LEVEL,
                                         0,
                                         chunk_meta_it->second.numBytes,
                                         chunk_meta_it->second.numElements);

  const auto ncore = (size_t)cpu_threads();

  std::vector<bool> has_null_per_thread(ncore, false);
  std::vector<double> max_double_per_thread(ncore, std::numeric_limits<double>::min());
  std::vector<double> min_double_per_thread(ncore, std::numeric_limits<double>::max());
  std::vector<int64_t> max_int64t_per_thread(ncore, std::numeric_limits<int64_t>::min());
  std::vector<int64_t> min_int64t_per_thread(ncore, std::numeric_limits<int64_t>::max());

  // parallel update elements
  std::vector<std::future<void>> threads;
  std::exception_ptr failed_any_chunk;
  std::mutex mtx;
  auto wait_cleanup_threads = [&] {
    try {
      for (auto& t : threads) {
        t.wait();
      }
      for (auto& t : threads) {
        t.get();
      }
    } catch (...) {
      std::unique_lock<std::mutex> lck(mtx);
      failed_any_chunk = std::current_exception();
    }
    threads.clear();
  };

  const auto segsz = (nrow + ncore - 1) / ncore;
  auto dbuf = chunk->get_buffer();
  auto dbuf_addr = dbuf->getMemoryPtr();
  dbuf->setUpdated();
  {
    std::lock_guard<std::mutex> lck(updel_roll.mutex);
    if (updel_roll.dirtyChunks.count(chunk.get()) == 0) {
      updel_roll.dirtyChunks.emplace(chunk.get(), chunk);
    }

    ChunkKey chunkey{updel_roll.catalog->get_currentDB().dbId,
                     cd->tableId,
                     cd->columnId,
                     fragment.fragmentId};
    updel_roll.dirtyChunkeys.insert(chunkey);
  }
  for (size_t rbegin = 0, c = 0; rbegin < nrow; ++c, rbegin += segsz) {
    threads.emplace_back(std::async(
        std::launch::async,
        [=,
         &has_null_per_thread,
         &min_int64t_per_thread,
         &max_int64t_per_thread,
         &min_double_per_thread,
         &max_double_per_thread,
         &frag_offsets,
         &rhs_values] {
          SQLTypeInfo lhs_type = cd->columnType;

          // !! not sure if this is a undocumented convention or a bug, but for a sharded
          // table the dictionary id of a encoded string column is not specified by
          // comp_param in physical table but somehow in logical table :) comp_param in
          // physical table is always 0, so need to adapt accordingly...
          auto cdl = (shard_ < 0)
                         ? cd
                         : catalog->getMetadataForColumn(
                               catalog->getLogicalTableId(td->tableId), cd->columnId);
          CHECK(cdl);
          DecimalOverflowValidator decimalOverflowValidator(lhs_type);
          StringDictionary* stringDict{nullptr};
          if (lhs_type.is_string()) {
            CHECK(kENCODING_DICT == lhs_type.get_compression());
            auto dictDesc = const_cast<DictDescriptor*>(
                catalog->getMetadataForDict(cdl->columnType.get_comp_param()));
            CHECK(dictDesc);
            stringDict = dictDesc->stringDict.get();
            CHECK(stringDict);
          }

          for (size_t r = rbegin; r < std::min(rbegin + segsz, nrow); r++) {
            const auto roffs = frag_offsets[r];
            auto data_ptr = dbuf_addr + roffs * get_element_size(lhs_type);
            auto sv = &rhs_values[1 == n_rhs_values ? 0 : r];
            ScalarTargetValue sv2;

            // Subtle here is on the two cases of string-to-string assignments, when
            // upstream passes RHS string as a string index instead of a preferred "real
            // string".
            //   case #1. For "SET str_col = str_literal", it is hard to resolve temp str
            //   index
            //            in this layer, so if upstream passes a str idx here, an
            //            exception is thrown.
            //   case #2. For "SET str_col1 = str_col2", RHS str idx is converted to LHS
            //   str idx.
            if (rhs_type.is_string()) {
              if (const auto vp = boost::get<int64_t>(sv)) {
                auto dictDesc = const_cast<DictDescriptor*>(
                    catalog->getMetadataForDict(rhs_type.get_comp_param()));
                if (nullptr == dictDesc) {
                  throw std::runtime_error(
                      "UPDATE does not support cast from string literal to string "
                      "column.");
                }
                auto stringDict = dictDesc->stringDict.get();
                CHECK(stringDict);
                sv2 = NullableString(stringDict->getString(*vp));
                sv = &sv2;
              }
            }

            if (const auto vp = boost::get<int64_t>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
#ifdef ENABLE_STRING_CONVERSION_AT_STORAGE_LAYER
                v = stringDict->getOrAdd(DatumToString(
                    rhs_type.is_time() ? Datum{.timeval = v} : Datum{.bigintval = v},
                    rhs_type));
#else
                throw std::runtime_error("UPDATE does not support cast to string.");
#endif
              }
              decimalOverflowValidator.validate<int64_t>(v);
              put_scalar<int64_t>(data_ptr, lhs_type, v, cd->columnName, &rhs_type);
              if (lhs_type.is_decimal()) {
                int64_t decimal;
                get_scalar<int64_t>(data_ptr, lhs_type, decimal);
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], decimal);
                if (!((v >= 0) ^ (decimal < 0))) {
                  throw std::runtime_error(
                      "Data conversion overflow on " + std::to_string(v) +
                      " from DECIMAL(" + std::to_string(rhs_type.get_dimension()) + ", " +
                      std::to_string(rhs_type.get_scale()) + ") to (" +
                      std::to_string(lhs_type.get_dimension()) + ", " +
                      std::to_string(lhs_type.get_scale()) + ")");
                }
              } else if (is_integral(lhs_type)) {
                set_minmax<int64_t>(
                    min_int64t_per_thread[c],
                    max_int64t_per_thread[c],
                    rhs_type.is_decimal() ? round(decimal_to_double(rhs_type, v)) : v);
              } else {
                set_minmax<double>(
                    min_double_per_thread[c],
                    max_double_per_thread[c],
                    rhs_type.is_decimal() ? decimal_to_double(rhs_type, v) : v);
              }
            } else if (const auto vp = boost::get<double>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
#ifdef ENABLE_STRING_CONVERSION_AT_STORAGE_LAYER
                v = stringDict->getOrAdd(DatumToString(Datum{.doubleval = v}, rhs_type));
#else
                throw std::runtime_error("UPDATE does not support cast to string.");
#endif
              }
              put_scalar<double>(data_ptr, lhs_type, v, cd->columnName);
              if (lhs_type.is_integer()) {
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], v);
              } else {
                set_minmax<double>(min_double_per_thread[c], max_double_per_thread[c], v);
              }
            } else if (const auto vp = boost::get<float>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
#ifdef ENABLE_STRING_CONVERSION_AT_STORAGE_LAYER
                v = stringDict->getOrAdd(DatumToString(Datum{.floatval = v}, rhs_type));
#else
                throw std::runtime_error("UPDATE does not support cast to string.");
#endif
              }
              put_scalar<float>(data_ptr, lhs_type, v, cd->columnName);
              if (lhs_type.is_integer()) {
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], v);
              } else {
                set_minmax<double>(min_double_per_thread[c], max_double_per_thread[c], v);
              }
            } else if (const auto vp = boost::get<NullableString>(sv)) {
              const auto s = boost::get<std::string>(vp);
              const auto sval = s ? *s : std::string("");
              if (lhs_type.is_string()) {
                decltype(stringDict->getOrAdd(sval)) sidx;
                {
                  std::unique_lock<std::mutex> lock(temp_mutex_);
                  sidx = stringDict->getOrAdd(sval);
                }
                put_scalar<int32_t>(data_ptr, lhs_type, sidx, cd->columnName);
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], sidx);
              } else if (sval.size() > 0) {
                auto dval = std::atof(sval.data());
                if (lhs_type.is_boolean()) {
                  dval = sval == "t" || sval == "true" || sval == "T" || sval == "True";
                } else if (lhs_type.is_time()) {
                  dval = StringToDatum(sval, lhs_type).timeval;
                }
                if (lhs_type.is_fp() || lhs_type.is_decimal()) {
                  put_scalar<double>(data_ptr, lhs_type, dval, cd->columnName);
                  set_minmax<double>(
                      min_double_per_thread[c], max_double_per_thread[c], dval);
                } else {
                  put_scalar<int64_t>(data_ptr, lhs_type, dval, cd->columnName);
                  set_minmax<int64_t>(
                      min_int64t_per_thread[c], max_int64t_per_thread[c], dval);
                }
              } else {
                put_null(data_ptr, lhs_type, cd->columnName);
                has_null_per_thread[c] = true;
              }
            } else {
              CHECK(false);
            }
          }
        }));
    if (threads.size() >= (size_t)cpu_threads()) {
      wait_cleanup_threads();
    }
    if (failed_any_chunk) {
      break;
    }
  }
  wait_cleanup_threads();
  if (failed_any_chunk) {
    std::rethrow_exception(failed_any_chunk);
  }

  bool has_null_per_chunk{false};
  double max_double_per_chunk{std::numeric_limits<double>::min()};
  double min_double_per_chunk{std::numeric_limits<double>::max()};
  int64_t max_int64t_per_chunk{std::numeric_limits<int64_t>::min()};
  int64_t min_int64t_per_chunk{std::numeric_limits<int64_t>::max()};
  for (size_t c = 0; c < ncore; ++c) {
    has_null_per_chunk |= has_null_per_thread[c];
    max_double_per_chunk =
        std::max<double>(max_double_per_chunk, max_double_per_thread[c]);
    min_double_per_chunk =
        std::min<double>(min_double_per_chunk, min_double_per_thread[c]);
    max_int64t_per_chunk =
        std::max<int64_t>(max_int64t_per_chunk, max_int64t_per_thread[c]);
    min_int64t_per_chunk =
        std::min<int64_t>(min_int64t_per_chunk, min_int64t_per_thread[c]);
  }
  updateColumnMetadata(cd,
                       fragment,
                       chunk,
                       has_null_per_chunk,
                       max_double_per_chunk,
                       min_double_per_chunk,
                       max_int64t_per_chunk,
                       min_int64t_per_chunk,
                       cd->columnType,
                       updel_roll);
}

void InsertOrderFragmenter::updateColumnMetadata(const ColumnDescriptor* cd,
                                                 FragmentInfo& fragment,
                                                 std::shared_ptr<Chunk_NS::Chunk> chunk,
                                                 const bool has_null_per_chunk,
                                                 const double max_double_per_chunk,
                                                 const double min_double_per_chunk,
                                                 const int64_t max_int64t_per_chunk,
                                                 const int64_t min_int64t_per_chunk,
                                                 const SQLTypeInfo& rhs_type,
                                                 UpdelRoll& updel_roll) {
  auto td = updel_roll.catalog->getMetadataForTable(cd->tableId);
  auto key = std::make_pair(td, &fragment);
  std::lock_guard<std::mutex> lck(updel_roll.mutex);
  if (0 == updel_roll.chunkMetadata.count(key)) {
    updel_roll.chunkMetadata[key] = fragment.getChunkMetadataMapPhysical();
  }
  if (0 == updel_roll.numTuples.count(key)) {
    updel_roll.numTuples[key] = fragment.shadowNumTuples;
  }
  auto& chunkMetadata = updel_roll.chunkMetadata[key];

  auto buffer = chunk->get_buffer();
  const auto& lhs_type = cd->columnType;
  if (is_integral(lhs_type) || (lhs_type.is_decimal() && rhs_type.is_decimal())) {
    buffer->encoder->updateStats(max_int64t_per_chunk, has_null_per_chunk);
    buffer->encoder->updateStats(min_int64t_per_chunk, has_null_per_chunk);
  } else if (lhs_type.is_fp()) {
    buffer->encoder->updateStats(max_double_per_chunk, has_null_per_chunk);
    buffer->encoder->updateStats(min_double_per_chunk, has_null_per_chunk);
  } else if (lhs_type.is_decimal()) {
    buffer->encoder->updateStats(
        (int64_t)(max_double_per_chunk * pow(10, lhs_type.get_scale())),
        has_null_per_chunk);
    buffer->encoder->updateStats(
        (int64_t)(min_double_per_chunk * pow(10, lhs_type.get_scale())),
        has_null_per_chunk);
  } else if (!lhs_type.is_array() &&
             !(lhs_type.is_string() && kENCODING_DICT != lhs_type.get_compression())) {
    buffer->encoder->updateStats(max_int64t_per_chunk, has_null_per_chunk);
    buffer->encoder->updateStats(min_int64t_per_chunk, has_null_per_chunk);
  }
  buffer->encoder->getMetadata(chunkMetadata[cd->columnId]);

  // removed as @alex suggests. keep it commented in case of any chance to revisit
  // it once after vacuum code is introduced. fragment.invalidateChunkMetadataMap();
}

void InsertOrderFragmenter::updateMetadata(const Catalog_Namespace::Catalog* catalog,
                                           const MetaDataKey& key,
                                           UpdelRoll& updel_roll) {
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  if (updel_roll.chunkMetadata.count(key)) {
    auto& fragmentInfo = *key.second;
    const auto& chunkMetadata = updel_roll.chunkMetadata[key];
    fragmentInfo.shadowChunkMetadataMap = chunkMetadata;
    fragmentInfo.setChunkMetadataMap(chunkMetadata);
    fragmentInfo.shadowNumTuples = updel_roll.numTuples[key];
    fragmentInfo.setPhysicalNumTuples(fragmentInfo.shadowNumTuples);
    // TODO(ppan): When fragment-level compaction is enable, the following code
    // should suffice. When not (ie. existing code), we'll revert to update
    // InsertOrderFragmenter::varLenColInfo_
    /*
    for (const auto cit : chunkMetadata) {
      const auto& cd = *catalog->getMetadataForColumn(td->tableId, cit.first);
      if (cd.columnType.get_size() < 0)
        fragmentInfo.varLenColInfox[cd.columnId] = cit.second.numBytes;
    }
    */
  }
}

}  // namespace Fragmenter_Namespace

void UpdelRoll::commitUpdate() {
  if (nullptr == catalog) {
    return;
  }
  const auto td = catalog->getMetadataForTable(logicalTableId);
  CHECK(td);
  // checkpoint all shards regardless, or epoch becomes out of sync
  if (td->persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    catalog->checkpoint(logicalTableId);
  }
  // for each dirty fragment
  for (auto& cm : chunkMetadata) {
    cm.first.first->fragmenter->updateMetadata(catalog, cm.first, *this);
  }
  dirtyChunks.clear();
  // flush gpu dirty chunks if update was not on gpu
  if (memoryLevel != Data_Namespace::MemoryLevel::GPU_LEVEL) {
    for (const auto& chunkey : dirtyChunkeys) {
      catalog->get_dataMgr().deleteChunksWithPrefix(
          chunkey, Data_Namespace::MemoryLevel::GPU_LEVEL);
    }
  }
}

void UpdelRoll::cancelUpdate() {
  if (nullptr == catalog) {
    return;
  }
  const auto td = catalog->getMetadataForTable(logicalTableId);
  CHECK(td);
  if (td->persistenceLevel != memoryLevel) {
    for (auto dit : dirtyChunks) {
      catalog->get_dataMgr().free(dit.first->get_buffer());
      dit.first->set_buffer(nullptr);
    }
  }
}
