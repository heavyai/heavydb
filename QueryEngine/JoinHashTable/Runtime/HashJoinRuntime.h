/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file    HashJoinRuntime.h
 * @brief
 *
 */

#ifndef QUERYENGINE_HASHJOINRUNTIME_H
#define QUERYENGINE_HASHJOINRUNTIME_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "../../../Shared/SqlTypesLayout.h"
#include "../../../Shared/sqltypes.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

#ifdef __CUDACC__
#include "../../DecodersImpl.h"
#else
#include "../../RuntimeFunctions.h"
#endif
#include "../../../Shared/funcannotations.h"

struct GenericKeyHandler;
struct BoundingBoxIntersectKeyHandler;
struct RangeKeyHandler;

struct BucketizedHashEntryInfo {
  alignas(sizeof(int64_t)) size_t bucketized_hash_entry_count;
  alignas(sizeof(int64_t)) int64_t bucket_normalization;

  inline size_t getNormalizedHashEntryCount() const {
    return bucketized_hash_entry_count;
  }
};

const size_t g_maximum_conditions_to_coalesce{8};

void init_hash_join_buff(int32_t* buff,
                         const int64_t entry_count,
                         const int32_t invalid_slot_val,
                         const int32_t cpu_thread_idx,
                         const int32_t cpu_thread_count);

#ifndef __CUDACC__
#ifdef HAVE_TBB

void init_hash_join_buff_tbb(int32_t* buff,
                             const int64_t entry_count,
                             const int32_t invalid_slot_val);

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__

void init_hash_join_buff_on_device(int32_t* buff,
                                   const int64_t entry_count,
                                   const int32_t invalid_slot_val,
                                   CUstream cuda_stream);

void init_baseline_hash_join_buff_32(int8_t* hash_join_buff,
                                     const int64_t entry_count,
                                     const size_t key_component_count,
                                     const bool with_val_slot,
                                     const int32_t invalid_slot_val,
                                     const int32_t cpu_thread_idx,
                                     const int32_t cpu_thread_count);

void init_baseline_hash_join_buff_64(int8_t* hash_join_buff,
                                     const int64_t entry_count,
                                     const size_t key_component_count,
                                     const bool with_val_slot,
                                     const int32_t invalid_slot_val,
                                     const int32_t cpu_thread_idx,
                                     const int32_t cpu_thread_count);

#ifndef __CUDACC__
#ifdef HAVE_TBB

void init_baseline_hash_join_buff_tbb_32(int8_t* hash_join_buff,
                                         const int64_t entry_count,
                                         const size_t key_component_count,
                                         const bool with_val_slot,
                                         const int32_t invalid_slot_val);

void init_baseline_hash_join_buff_tbb_64(int8_t* hash_join_buff,
                                         const int64_t entry_count,
                                         const size_t key_component_count,
                                         const bool with_val_slot,
                                         const int32_t invalid_slot_val);

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__

void init_baseline_hash_join_buff_on_device_32(int8_t* hash_join_buff,
                                               const int64_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val,
                                               CUstream cuda_stream);

void init_baseline_hash_join_buff_on_device_64(int8_t* hash_join_buff,
                                               const int64_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val,
                                               CUstream cuda_stream);

enum ColumnType { SmallDate = 0, Signed = 1, Unsigned = 2, Double = 3 };

struct JoinChunk {
  const int8_t*
      col_buff;  // actually from AbstractBuffer::getMemoryPtr() via Chunk_NS::Chunk
  size_t num_elems;
};

struct JoinColumn {
  const int8_t*
      col_chunks_buff;  // actually a JoinChunk* from ColumnFetcher::makeJoinColumn()
  size_t col_chunks_buff_sz;
  size_t num_chunks;
  size_t num_elems;
  size_t elem_sz;
};

struct JoinColumnTypeInfo {
  const size_t elem_sz;
  const int64_t min_val;
  const int64_t max_val;
  const int64_t null_val;
  const bool uses_bw_eq;
  const int64_t translated_null_val;
  const ColumnType column_type;
};

inline ColumnType get_join_column_type_kind(const SQLTypeInfo& ti) {
  if (ti.is_date_in_days()) {
    return SmallDate;
  } else {
    return is_unsigned_type(ti) ? Unsigned : Signed;
  }
}

struct JoinBucketInfo {
  std::vector<double> inverse_bucket_sizes_for_dimension;
  bool is_double;  // TODO(adb): assume float otherwise (?)
};

struct ShardInfo {
  size_t shard;
  size_t entry_count_per_shard;
  size_t num_shards;
  int device_count;
};

struct OneToOnePerfectJoinHashTableFillFuncArgs {
  int32_t* buff;
  int32_t* dev_err_buff;
  const int32_t invalid_slot_val;
  const bool for_semi_join;
  const JoinColumn join_column;
  const JoinColumnTypeInfo type_info;
  const int32_t* sd_inner_to_outer_translation_map;
  const int32_t min_inner_elem;
  const int64_t bucket_normalization;  // used only if we call bucketized_hash_join
};

struct OneToManyPerfectJoinHashTableFillFuncArgs {
  int32_t* buff;
  const BucketizedHashEntryInfo hash_entry_info;
  const JoinColumn join_column;
  const JoinColumnTypeInfo type_info;
  const int32_t* sd_inner_to_outer_translation_map;
  const int32_t min_inner_elem;
  const int64_t bucket_normalization;  // used only if we call bucketized_hash_join
  const bool for_window_framing;
};

int fill_hash_join_buff_bucketized(OneToOnePerfectJoinHashTableFillFuncArgs const args,
                                   int32_t const cpu_thread_idx,
                                   int32_t const cpu_thread_count);

int fill_hash_join_buff(OneToOnePerfectJoinHashTableFillFuncArgs const args,
                        int32_t const cpu_thread_idx,
                        int32_t const cpu_thread_count);

int fill_hash_join_buff_bitwise_eq(OneToOnePerfectJoinHashTableFillFuncArgs const args,
                                   int32_t const cpu_thread_idx,
                                   int32_t const cpu_thread_count);

void fill_hash_join_buff_on_device(CUstream cuda_stream,
                                   OneToOnePerfectJoinHashTableFillFuncArgs const args);

void fill_hash_join_buff_on_device_bucketized(
    CUstream cuda_stream,
    OneToOnePerfectJoinHashTableFillFuncArgs const args);

void fill_hash_join_buff_on_device_sharded(
    CUstream cuda_stream,
    OneToOnePerfectJoinHashTableFillFuncArgs const args,
    ShardInfo const shard_info);

void fill_hash_join_buff_on_device_sharded_bucketized(
    CUstream cuda_stream,
    OneToOnePerfectJoinHashTableFillFuncArgs const args,
    ShardInfo const shard_info);

void fill_one_to_many_hash_table(OneToManyPerfectJoinHashTableFillFuncArgs const args,
                                 int32_t const cpu_thread_count);

void fill_one_to_many_hash_table_bucketized(
    OneToManyPerfectJoinHashTableFillFuncArgs const args,
    int32_t const cpu_thread_count);

void fill_one_to_many_hash_table_on_device(
    CUstream cuda_stream,
    OneToManyPerfectJoinHashTableFillFuncArgs const args);

void fill_one_to_many_hash_table_on_device_bucketized(
    CUstream cuda_stream,
    OneToManyPerfectJoinHashTableFillFuncArgs const args);

void fill_one_to_many_hash_table_on_device_sharded(
    CUstream cuda_stream,
    OneToManyPerfectJoinHashTableFillFuncArgs const args,
    ShardInfo const shard_info);

int fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                    const int64_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const bool for_semi_join,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const GenericKeyHandler* key_handler,
                                    const int64_t num_elems,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count);

int bbox_intersect_fill_baseline_hash_join_buff_32(
    int8_t* hash_buff,
    const int64_t entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const bool with_val_slot,
    const BoundingBoxIntersectKeyHandler* key_handler,
    const int64_t num_elems,
    const int32_t cpu_thread_idx,
    const int32_t cpu_thread_count);

int range_fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                          const size_t entry_count,
                                          const int32_t invalid_slot_val,
                                          const size_t key_component_count,
                                          const bool with_val_slot,
                                          const RangeKeyHandler* key_handler,
                                          const size_t num_elems,
                                          const int32_t cpu_thread_idx,
                                          const int32_t cpu_thread_count);

int fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                    const int64_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const bool for_semi_join,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const GenericKeyHandler* key_handler,
                                    const int64_t num_elems,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count);

int bbox_intersect_fill_baseline_hash_join_buff_64(
    int8_t* hash_buff,
    const int64_t entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const bool with_val_slot,
    const BoundingBoxIntersectKeyHandler* key_handler,
    const int64_t num_elems,
    const int32_t cpu_thread_idx,
    const int32_t cpu_thread_count);

int range_fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                          const size_t entry_count,
                                          const int32_t invalid_slot_val,
                                          const size_t key_component_count,
                                          const bool with_val_slot,
                                          const RangeKeyHandler* key_handler,
                                          const size_t num_elems,
                                          const int32_t cpu_thread_idx,
                                          const int32_t cpu_thread_count);

void fill_baseline_hash_join_buff_on_device_32(int8_t* hash_buff,
                                               const int64_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const bool for_semi_join,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const int64_t num_elems,
                                               CUstream cuda_stream);

void fill_baseline_hash_join_buff_on_device_64(int8_t* hash_buff,
                                               const int64_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const bool for_semi_join,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const int64_t num_elems,
                                               CUstream cuda_stream);

void bbox_intersect_fill_baseline_hash_join_buff_on_device_64(
    int8_t* hash_buff,
    const int64_t entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const bool with_val_slot,
    int* dev_err_buff,
    const BoundingBoxIntersectKeyHandler* key_handler,
    const int64_t num_elems,
    CUstream cuda_stream);

void range_fill_baseline_hash_join_buff_on_device_64(int8_t* hash_buff,
                                                     const int64_t entry_count,
                                                     const int32_t invalid_slot_val,
                                                     const size_t key_component_count,
                                                     const bool with_val_slot,
                                                     int* dev_err_buff,
                                                     const RangeKeyHandler* key_handler,
                                                     const size_t num_elems,
                                                     CUstream cuda_stream);

void fill_one_to_many_baseline_hash_table_32(
    int32_t* buff,
    const int32_t* composite_key_dict,
    const int64_t hash_entry_count,
    const size_t key_component_count,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const std::vector<const int32_t*>& sd_inner_to_outer_translation_maps,
    const std::vector<int32_t>& sd_min_inner_elems,
    const int32_t cpu_thread_count,
    const bool is_range_join = false,
    const bool is_geo_compressed = false,
    const bool for_window_framing = false);

void fill_one_to_many_baseline_hash_table_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const int64_t hash_entry_count,
    const size_t key_component_count,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const std::vector<const int32_t*>& sd_inner_to_outer_translation_maps,
    const std::vector<int32_t>& sd_min_inner_elems,
    const int32_t cpu_thread_count,
    const bool is_range_join = false,
    const bool is_geo_compressed = false,
    const bool for_window_framing = false);

void fill_one_to_many_baseline_hash_table_on_device_32(
    int32_t* buff,
    const int32_t* composite_key_dict,
    const int64_t hash_entry_count,
    const size_t key_component_count,
    const GenericKeyHandler* key_handler,
    const int64_t num_elems,
    const bool for_window_framing,
    CUstream cuda_stream);

void fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const int64_t hash_entry_count,
    const GenericKeyHandler* key_handler,
    const int64_t num_elems,
    const bool for_window_framing,
    CUstream cuda_stream);

void bbox_intersect_fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const int64_t hash_entry_count,
    const BoundingBoxIntersectKeyHandler* key_handler,
    const int64_t num_elems,
    CUstream cuda_stream);

void range_fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const size_t hash_entry_count,
    const RangeKeyHandler* key_handler,
    const size_t num_elems,
    CUstream cuda_stream);

void approximate_distinct_tuples(uint8_t* hll_buffer_all_cpus,
                                 const uint32_t b,
                                 const size_t padded_size_bytes,
                                 const std::vector<JoinColumn>& join_column_per_key,
                                 const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                 const int thread_count);

void approximate_distinct_tuples_bbox_intersect(
    uint8_t* hll_buffer_all_cpus,
    std::vector<int32_t>& row_counts,
    const uint32_t b,
    const size_t padded_size_bytes,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_buckets_per_key,
    const int thread_count);

void approximate_distinct_tuples_range(
    uint8_t* hll_buffer_all_cpus,
    std::vector<int32_t>& row_counts,
    const uint32_t b,
    const size_t padded_size_bytes,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_buckets_per_key,
    const bool is_compressed,
    const int thread_count);

void approximate_distinct_tuples_on_device(uint8_t* hll_buffer,
                                           const uint32_t b,
                                           const GenericKeyHandler* key_handler,
                                           const int64_t num_elems,
                                           CUstream cuda_stream);

void approximate_distinct_tuples_on_device_bbox_intersect(
    uint8_t* hll_buffer,
    const uint32_t b,
    int32_t* row_counts_buffer,
    const BoundingBoxIntersectKeyHandler* key_handler,
    const int64_t num_elems,
    CUstream cuda_stream);

void compute_bucket_sizes_on_cpu(std::vector<double>& bucket_sizes_for_dimension,
                                 const JoinColumn& join_column,
                                 const JoinColumnTypeInfo& type_info,
                                 const std::vector<double>& bucket_size_thresholds,
                                 const int thread_count);

void approximate_distinct_tuples_on_device_range(uint8_t* hll_buffer,
                                                 const uint32_t b,
                                                 int32_t* row_counts_buffer,
                                                 const RangeKeyHandler* key_handler,
                                                 const size_t num_elems,
                                                 const size_t block_size_x,
                                                 const size_t grid_size_x,
                                                 CUstream cuda_stream);

void compute_bucket_sizes_on_device(double* bucket_sizes_buffer,
                                    const JoinColumn* join_column,
                                    const JoinColumnTypeInfo* type_info,
                                    const double* bucket_size_thresholds,
                                    CUstream cuda_stream);

#endif  // QUERYENGINE_HASHJOINRUNTIME_H
