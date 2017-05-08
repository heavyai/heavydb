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

/**
 * @file    ProfileTest.cpp
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Unit tests for microbenchmark.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */
#include "ProfileTest.h"
#include "Shared/measure.h"
#include "../QueryEngine/ResultRows.h"
#include "../QueryEngine/ResultSet.h"

#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
#include <cuda_runtime.h>
#include <thrust/system_error.h>
#endif

#include <gtest/gtest.h>
#include <boost/make_unique.hpp>

#include <future>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <unordered_map>

bool g_gpus_present = false;

const float c_space_usage = 2.0f;

namespace {
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
void check_error(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char* errorString{nullptr};
    cuGetErrorString(status, &errorString);
    throw std::runtime_error(errorString ? errorString : "Unkown error");
  }
}
#endif

inline size_t get_gpu_count() {
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  int num_gpus = 0;
  try {
    check_error(cuInit(0));
    check_error(cuDeviceGetCount(&num_gpus));
  } catch (std::runtime_error&) {
    return 0;
  }
  return num_gpus;
#else
  return 0;
#endif
}

inline bool is_gpu_present() {
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  return (get_gpu_count() > 0);
#else
  return false;
#endif  // HAVE_CUDA
}

template <typename T = int64_t>
bool generate_numbers(int8_t* random_numbers,
                      const unsigned num_random_numbers,
                      const T min_number,
                      const T max_number,
                      const DIST_KIND dist,
                      const size_t stride = sizeof(T)) {
  if (random_numbers == nullptr) {
    return false;
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  switch (dist) {
    case NRM: {
      std::normal_distribution<> d((max_number + min_number) / 2, 1);
      for (unsigned i = 0; i < num_random_numbers; ++i) {
        *reinterpret_cast<T*>(random_numbers + i * stride) =
            std::max<T>(min_number, std::min<T>(max_number, std::round(d(gen))));
      }
    } break;
    case EXP1: {
      std::exponential_distribution<> d(1);
      for (unsigned i = 0; i < num_random_numbers; ++i) {
        *reinterpret_cast<T*>(random_numbers + i * stride) =
            std::max<T>(min_number, std::min<T>(max_number, std::round(d(gen))));
      }
    } break;
    case EXP2: {
      std::exponential_distribution<> d(2);
      for (unsigned i = 0; i < num_random_numbers; ++i) {
        *reinterpret_cast<T*>(random_numbers + i * stride) =
            std::max<T>(min_number, std::min<T>(max_number, std::round(d(gen))));
      }
    } break;
    case UNI: {
      std::uniform_int_distribution<T> d(min_number, max_number);
      for (unsigned i = 0; i < num_random_numbers; ++i) {
        *reinterpret_cast<T*>(random_numbers + i * stride) = d(gen);
      }
    } break;
    case POI: {
      std::poisson_distribution<T> d(4);
      for (unsigned i = 0; i < num_random_numbers; ++i) {
        *reinterpret_cast<T*>(random_numbers + i * stride) = std::max<T>(min_number, std::min(max_number, d(gen)));
      }
    } break;
    default:
      CHECK(false);
  }

  return true;
}

bool generate_columns_on_host(int8_t* buffers,
                              const size_t row_count,
                              const size_t col_count,
                              const std::vector<size_t>& col_widths,
                              const std::vector<std::pair<int64_t, int64_t>>& ranges,
                              const bool is_columnar,
                              const std::vector<DIST_KIND>& dists) {
  if (buffers == nullptr) {
    return false;
  }
  CHECK_EQ(col_widths.size(), col_count);
  CHECK_EQ(ranges.size(), col_count);
  size_t row_size = 0;
  for (auto& wid : col_widths) {
    row_size += wid;
  }
  std::vector<std::future<bool>> child_threads;
  for (size_t i = 0; i < col_count; buffers += (is_columnar ? row_count : 1) * col_widths[i++]) {
    if (dists[i] == DIST_KIND::INVALID) {
      continue;
    }
    CHECK_LE(ranges[i].first, ranges[i].second);
    switch (col_widths[i]) {
      case 4:
        child_threads.push_back(std::async(std::launch::async,
                                           generate_numbers<int32_t>,
                                           buffers,
                                           row_count,
                                           static_cast<int32_t>(ranges[i].first),
                                           static_cast<int32_t>(ranges[i].second),
                                           dists[i],
                                           (is_columnar ? 4 : row_size)));
        break;
      case 8:
        child_threads.push_back(std::async(std::launch::async,
                                           generate_numbers<int64_t>,
                                           buffers,
                                           row_count,
                                           ranges[i].first,
                                           ranges[i].second,
                                           dists[i],
                                           (is_columnar ? 8 : row_size)));
        break;
      default:
        CHECK(false);
    }
  }
  for (auto& child : child_threads) {
    child.get();
  }
  return true;
}

inline void init_groups_on_host(int8_t* groups,
                                const size_t group_count,
                                const size_t col_count,
                                const std::vector<size_t>& col_widths,
                                const std::vector<size_t>& init_vals,
                                const bool is_columnar) {
  CHECK_EQ(col_count, col_widths.size());
  CHECK_EQ(col_count, init_vals.size());
  std::vector<std::future<void>> child_threads;
  const size_t cpu_count = cpu_threads();
  const auto stride = (group_count + cpu_count - 1) / cpu_count;
  size_t row_size = 0;
  for (auto wid : col_widths) {
    row_size += wid;
  }

  for (size_t start_group = 0; start_group < group_count; start_group += stride) {
    const auto end_group = std::min(group_count, start_group + stride);
    if (is_columnar) {
      child_threads.push_back(std::async(std::launch::async, [&, start_group, end_group]() {
        auto col_base = groups;
        for (size_t j = 0; j < col_count; col_base += col_widths[j++] * group_count) {
          for (size_t i = start_group; i < end_group; ++i) {
            switch (col_widths[j]) {
              case 4: {
                auto col_ptr = reinterpret_cast<uint32_t*>(col_base);
                std::fill(col_ptr, col_ptr + group_count, static_cast<uint32_t>(init_vals[j]));
              } break;
              case 8: {
                auto col_ptr = reinterpret_cast<size_t*>(col_base);
                std::fill(col_ptr, col_ptr + group_count, init_vals[j]);
              } break;
              default:
                CHECK(false);
            }
          }
        }
      }));
    } else {
      child_threads.push_back(std::async(std::launch::async, [&, start_group, end_group]() {
        for (size_t i = start_group; i < end_group; ++i) {
          auto row_base = groups + i * row_size;
          for (size_t j = 0; j < col_count; row_base += col_widths[j++]) {
            switch (col_widths[j]) {
              case 4:
                *reinterpret_cast<uint32_t*>(row_base) = static_cast<uint32_t>(init_vals[j]);
                break;
              case 8:
                *reinterpret_cast<size_t*>(row_base) = init_vals[j];
                break;
              default:
                CHECK(false);
            }
          }
        }
      }));
    }
  }
  for (auto& child : child_threads) {
    child.get();
  }
}

#if defined(TRY_COLUMNAR) || defined(TRY_MASH_COLUMNAR)
void columnarize_groups_on_host(int8_t* columnar_buffer,
                                const int8_t* rowwise_buffer,
                                const size_t row_count,
                                const std::vector<size_t>& col_widths) {
  std::vector<std::future<void>> child_threads;
  const size_t cpu_count = cpu_threads();
  const auto stride = (row_count + cpu_count - 1) / cpu_count;
  size_t row_size = 0;
  for (auto wid : col_widths) {
    row_size += wid;
  }

  for (size_t start_row = 0; start_row < row_count; start_row += stride) {
    const auto end_row = std::min(row_count, start_row + stride);
    child_threads.push_back(std::async(std::launch::async, [&, start_row, end_row]() {
      for (size_t i = start_row; i < end_row; ++i) {
        auto read_ptr = rowwise_buffer + i * row_size;
        auto write_base = columnar_buffer;
        for (size_t j = 0; j < col_widths.size(); ++j) {
          auto write_ptr = write_base + i * col_widths[j];
          switch (col_widths[j]) {
            case 4:
              *reinterpret_cast<uint32_t*>(write_ptr) = *reinterpret_cast<const uint32_t*>(read_ptr);
              break;
            case 8:
              *reinterpret_cast<size_t*>(write_ptr) = *reinterpret_cast<const size_t*>(read_ptr);
              break;
            default:
              CHECK(false);
          }
          read_ptr += col_widths[j];
          write_base += row_count * col_widths[j];
        }
      }
    }));
  }
  for (auto& child : child_threads) {
    child.get();
  }
}
#endif

template <typename ValT = int64_t>
ValT get_default_value(OP_KIND op) {
  switch (op) {
    case OP_COUNT:
    case OP_SUM:
      return ValT(0);
    case OP_MIN:
      return std::numeric_limits<ValT>::max();
    case OP_MAX:
      return std::numeric_limits<ValT>::min();
    default:
      CHECK(false);
  }
  return ValT(0);
}

DIST_KIND get_default_dist(OP_KIND op) {
  switch (op) {
    case OP_COUNT:
      return DIST_KIND::INVALID;
    case OP_SUM:
    case OP_MIN:
    case OP_MAX:
      return DIST_KIND::UNI;
    default:
      CHECK(false);
  }
  return DIST_KIND::INVALID;
}

template <typename ValT = int64_t>
std::pair<ValT, ValT> get_default_range(OP_KIND op) {
  switch (op) {
    case OP_COUNT:
      return {ValT(0), ValT(0)};
    case OP_SUM:
    case OP_MIN:
    case OP_MAX:
      return {std::numeric_limits<ValT>::min(), std::numeric_limits<ValT>::max()};
    default:
      CHECK(false);
  }
  CHECK(false);
  return {ValT(0), ValT(0)};
}

template <class T>
inline void hash_combine(std::size_t& seed, T const& v) {
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace

namespace std {

template <typename T>
struct hash<vector<T>> {
  size_t operator()(const vector<T>& vint) const {
    size_t seed = 0;
    for (auto i : vint) {
      // Combine the hash of the current vector with the hashes of the previous
      // ones
      hash_combine(seed, i);
    }
    return seed;
  }
};

}  // namespace std

namespace {
template <typename KeyT = int64_t>
inline bool is_empty_slot(const KeyT k) {
  static_assert(std::is_same<KeyT, int64_t>::value, "Unsupported template parameter other than int64_t for now");
  return k == EMPTY_KEY_64;
}

template <typename KeyT = int64_t, typename ValT = int64_t>
class AggregateEmulator {
 public:
  typedef std::unordered_map<std::vector<KeyT>, std::vector<ValT>> ResultType;

  explicit AggregateEmulator(const std::vector<OP_KIND>& ops) : agg_ops_(ops) {}

  ResultType run(const int8_t* buffers,
                 const size_t key_count,
                 const size_t val_count,
                 const size_t row_count,
                 const bool is_columnar) {
    std::vector<std::future<void>> child_threads;
    const size_t cpu_count = cpu_threads();
    const size_t stride = (row_count + cpu_count - 1) / cpu_count;
    std::vector<ResultType> partial_results(cpu_count);
    for (size_t start_row = 0, i = 0; start_row < row_count; start_row += stride, ++i) {
      const auto end_row = std::min(row_count, start_row + stride);
      child_threads.push_back(std::async(std::launch::async,
                                         &AggregateEmulator::runDispatch,
                                         this,
                                         std::ref(partial_results[i]),
                                         buffers,
                                         key_count,
                                         val_count,
                                         row_count,
                                         start_row,
                                         end_row,
                                         is_columnar));
    }

    for (auto& child : child_threads) {
      child.get();
    }

    return reduce(partial_results);
  }

  bool compare(const int8_t* buffers,
               const size_t key_count,
               const size_t val_count,
               const size_t group_count,
               const bool is_columnar,
               const ResultType& ref_result) {
    std::vector<std::future<size_t>> child_threads;
    const size_t cpu_count = cpu_threads();
    const auto stride = (group_count + cpu_count - 1) / cpu_count;
    for (size_t start_group = 0; start_group < group_count; start_group += stride) {
      const auto end_group = std::min(group_count, start_group + stride);
      child_threads.push_back(std::async(std::launch::async,
                                         &AggregateEmulator::compareDispatch,
                                         this,
                                         buffers,
                                         key_count,
                                         val_count,
                                         group_count,
                                         start_group,
                                         end_group,
                                         is_columnar,
                                         ref_result));
    }
    size_t matches = 0;
    for (auto& child : child_threads) {
      matches += child.get();
    }

    return matches == ref_result.size();
  }

  ResultType reduce(const std::vector<ResultType>& partial_results) {
    ResultType final_result;
    if (partial_results.size() == 1) {
      final_result = partial_results[0];
      return final_result;
    }
    for (auto& groups : partial_results) {
      for (auto& grp : groups) {
        auto& keys = grp.first;
        if (is_empty_slot(keys[0])) {
          continue;
        }
        if (!final_result.count(keys)) {
          final_result.insert(std::make_pair(keys, grp.second));
          continue;
        }
        const auto val_count = agg_ops_.size();
        CHECK_EQ(val_count, final_result[keys].size());
        CHECK_EQ(val_count, grp.second.size());
        for (size_t v = 0; v < val_count; ++v) {
          const ValT value = grp.second[v];
          switch (agg_ops_[v]) {
            case OP_COUNT:
            case OP_SUM:
              final_result[keys][v] += value;
              break;
            case OP_MIN:
              final_result[keys][v] = std::min(final_result[keys][v], value);
              break;
            case OP_MAX:
              final_result[keys][v] = std::max(final_result[keys][v], value);
              break;
            default:
              CHECK(false);
          }
        }
      }
    }
    return final_result;
  }

 private:
  void runDispatch(ResultType& partial_res,
                   const int8_t* buffers,
                   const size_t key_count,
                   const size_t val_count,
                   const size_t row_count,
                   const size_t start_row,
                   const size_t end_row,
                   const bool is_columnar) {
    CHECK_EQ(agg_ops_.size(), val_count);
    const size_t row_size = sizeof(KeyT) * key_count + sizeof(ValT) * val_count;
    for (size_t i = start_row; i < end_row; ++i) {
      std::vector<KeyT> keys(key_count);
      auto key_buffers = reinterpret_cast<const KeyT*>(buffers);
      if (is_columnar) {
        for (size_t k = 0; k < key_count; ++k) {
          keys[k] = key_buffers[i + k * row_count];
        }
      } else {
        for (size_t k = 0; k < key_count; ++k) {
          keys[k] = reinterpret_cast<const KeyT*>(buffers + i * row_size)[k];
        }
      }
      CHECK_EQ(keys.size(), key_count);
      if (is_empty_slot(keys[0])) {
        continue;
      }

      const bool inserted = partial_res.count(keys) != 0;
      if (inserted) {
        CHECK_EQ(partial_res[keys].size(), val_count);
      } else {
        partial_res[keys] = std::vector<ValT>(val_count);
      }

      for (size_t v = 0; v < val_count; ++v) {
        ValT value;
        if (is_columnar) {
          auto val_buffer = reinterpret_cast<const ValT*>(key_buffers + key_count * row_count);
          value = val_buffer[i + v * row_count];
        } else {
          auto val_buffer = reinterpret_cast<const ValT*>(buffers + row_size * i + sizeof(KeyT) * key_count);
          value = val_buffer[v];
        }

        switch (agg_ops_[v]) {
          case OP_COUNT:
            if (inserted) {
              ++partial_res[keys][v];
            } else {
              partial_res[keys][v] = 1;
            }
            break;
          case OP_SUM:
            if (inserted) {
              partial_res[keys][v] += value;
            } else {
              partial_res[keys][v] = value;
            }
            break;
          case OP_MIN:
            if (inserted) {
              partial_res[keys][v] = std::min(partial_res[keys][v], value);
            } else {
              partial_res[keys][v] = value;
            }
            break;
          case OP_MAX:
            if (inserted) {
              partial_res[keys][v] = std::max(partial_res[keys][v], value);
            } else {
              partial_res[keys][v] = value;
            }
            break;
          default:
            CHECK(false);
        }
      }
    }
  }

  size_t compareDispatch(const int8_t* buffers,
                         const size_t key_count,
                         const size_t val_count,
                         const size_t group_count,
                         const size_t start_group,
                         const size_t end_group,
                         const bool is_columnar,
                         const ResultType& ref_result) {
    CHECK_LT(size_t(0), key_count);
    size_t matches = 0;
    const size_t row_size = sizeof(KeyT) * key_count + sizeof(ValT) * val_count;
    for (size_t i = start_group; i < end_group; ++i) {
      std::vector<KeyT> keys(key_count);
      const auto key_buffers = reinterpret_cast<const KeyT*>(buffers);
      if (is_columnar) {
        for (size_t k = 0; k < key_count; ++k) {
          keys[k] = key_buffers[i + k * group_count];
        }
      } else {
        for (size_t k = 0; k < key_count; ++k) {
          keys[k] = reinterpret_cast<const KeyT*>(buffers + i * row_size)[k];
        }
      }
      if (is_empty_slot(keys[0])) {
        continue;
      }
      auto row_it = ref_result.find(keys);
      if (row_it == ref_result.end()) {
        return 0;
      }
      auto& ref_vals = row_it->second;
      CHECK_EQ(val_count, ref_vals.size());
      std::vector<ValT> actual_vals(val_count);
      for (size_t v = 0; v < val_count; ++v) {
        if (is_columnar) {
          auto val_buffers = reinterpret_cast<const ValT*>(key_buffers + key_count * group_count);
          actual_vals[v] = val_buffers[i + v * group_count];
        } else {
          auto val_buffers = reinterpret_cast<const ValT*>(buffers + row_size * i + sizeof(KeyT) * key_count);
          actual_vals[v] = val_buffers[v];
        }
      }
      for (size_t v = 0; v < val_count; ++v) {
        if (actual_vals[v] != ref_vals[v]) {
          return 0;
        }
      }
      ++matches;
    }
    return matches;
  }

  std::vector<OP_KIND> agg_ops_;
};

#ifdef SAVE_MASH_BUF
template <bool isColumnar, typename KeyT = int64_t, typename ValT = int64_t>
void mash_restore_dispatch(int8_t* output_buffer,
                           const int8_t* groups_buffer,
                           const size_t group_count,
                           const size_t entry_size,
                           const int8_t* input_buffer,
                           const size_t row_count,
                           const size_t key_count,
                           const size_t row_size,
                           const std::vector<size_t>& col_widths,
                           const size_t empty_key,
                           const size_t start_group,
                           const size_t end_group) {
  const auto val_count = col_widths.size() - key_count;
  const auto read_step = isColumnar ? row_count : 1;
  const auto write_step = isColumnar ? group_count : 1;
  for (size_t i = start_group; i < end_group; ++i) {
    const auto group_ptr = groups_buffer + i * (isColumnar ? sizeof(int64_t) : entry_size);
    const auto key_idx = *reinterpret_cast<const int64_t*>(group_ptr);
    auto read_ptr = input_buffer + key_idx * (isColumnar ? sizeof(KeyT) : row_size);
    auto write_ptr = output_buffer + i * (isColumnar ? sizeof(KeyT) : row_size);
    if (is_empty_slot(key_idx)) {
      *reinterpret_cast<KeyT*>(write_ptr) = static_cast<KeyT>(empty_key);
      continue;
    }
    for (size_t k = 0; k < key_count;
         ++k, write_ptr += write_step * sizeof(KeyT), read_ptr += read_step * sizeof(KeyT)) {
      *reinterpret_cast<KeyT*>(write_ptr) = *reinterpret_cast<const KeyT*>(read_ptr);
    }
    if (isColumnar) {
      write_ptr = output_buffer + key_count * sizeof(KeyT) * group_count + sizeof(ValT) * i;
      read_ptr = groups_buffer + sizeof(int64_t) * group_count + sizeof(ValT) * i;
      for (size_t v = 0; v < val_count;
           ++v, write_ptr += write_step * sizeof(ValT), read_ptr += write_step * sizeof(ValT)) {
        *reinterpret_cast<ValT*>(write_ptr) = *reinterpret_cast<const ValT*>(read_ptr);
      }
    } else {
      memcpy(write_ptr, group_ptr + sizeof(int64_t), entry_size - sizeof(int64_t));
    }
  }
}

template <bool isColumnar = false, typename KeyT = int64_t, typename ValT = int64_t>
void mash_restore_keys(int8_t* output_buffer,
                       const int8_t* groups_buffer,
                       const size_t group_count,
                       const int8_t* input_buffer,
                       const size_t row_count,
                       const size_t key_count,
                       const std::vector<size_t>& col_widths,
                       const std::vector<size_t>& init_vals) {
  size_t entry_size = sizeof(int64_t);
  for (size_t i = key_count; i < col_widths.size(); ++i) {
    entry_size += col_widths[i];
  }
  size_t row_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    row_size += col_widths[i];
  }
  std::vector<std::future<void>> child_threads;
  const size_t cpu_count = cpu_threads();
  const auto stride = (group_count + cpu_count - 1) / cpu_count;
  for (size_t start_group = 0; start_group < group_count; start_group += stride) {
    const auto end_group = std::min(group_count, start_group + stride);
    child_threads.push_back(std::async(std::launch::async,
                                       mash_restore_dispatch<isColumnar, KeyT, ValT>,
                                       output_buffer,
                                       groups_buffer,
                                       group_count,
                                       entry_size,
                                       input_buffer,
                                       row_count,
                                       key_count,
                                       row_size,
                                       std::ref(col_widths),
                                       init_vals[0],
                                       start_group,
                                       end_group));
  }
  for (auto& child : child_threads) {
    child.get();
  }
}
#endif

#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
class CudaTimer {
 public:
  CudaTimer(size_t buf_sz) : used_size(buf_sz) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  CudaTimer() : used_size(size_t(-1)) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  ~CudaTimer() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_, stop_);
    if (used_size == size_t(-1)) {
      std::cout << "Current query took " << elapsedTime << " ms on device.\n";
    } else {
      std::cout << "Current query took " << elapsedTime << " ms on device using " << used_size / (1024 * 1024.f)
                << " MB VRAM.\n";
    }
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

 private:
  const size_t used_size;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};
#endif
}  // namespace

TEST(Hash, Baseline) {
  // Config
  const std::vector<OP_KIND> agg_ops{OP_COUNT, OP_MAX};
  const size_t key_count = 3;
  const size_t val_count = 2;
  const size_t row_count = 20000000;
  const bool is_columnar = false;

  CHECK_EQ(agg_ops.size(), val_count);
  std::vector<size_t> col_widths(key_count, sizeof(int64_t));
  std::vector<size_t> init_vals(key_count, EMPTY_KEY_64);
  for (size_t i = 0; i < val_count; ++i) {
    col_widths.push_back(sizeof(uint64_t));
    init_vals.push_back(get_default_value(agg_ops[i]));
  }

  std::vector<DIST_KIND> dist_tries{
      DIST_KIND::UNI, DIST_KIND::UNI, DIST_KIND::NRM, DIST_KIND::EXP1, DIST_KIND::EXP2, DIST_KIND::POI};
  std::vector<std::string> dist_names{
      "uniform(-100, 100)", "uniform(-100000, 100000)", "normal(0, 1)", "exp(1)", "exp(2)", "poisson(4)"};
  std::vector<std::pair<int64_t, int64_t>> range_tries{
      {-100, 100},
      {-100000, 100000},
      {-1000, 1000},
      {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()},
      {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()},
      {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}};
  CHECK_EQ(dist_tries.size(), dist_names.size());
  CHECK_EQ(dist_tries.size(), range_tries.size());

  std::vector<std::vector<size_t>> selected_tries;
  for (size_t i = 0; i < dist_tries.size(); ++i) {
    selected_tries.push_back(std::vector<size_t>(key_count, i));
  }
  selected_tries.push_back({3, 2, 4, 3, 4, 5, 1});
  for (size_t i = 0; i < selected_tries.size(); ++i) {
    std::cout << "Try distributions of " << key_count << " keys: ";
    std::vector<DIST_KIND> distributions;
    for (size_t j = 0; j < key_count; ++j) {
      std::cout << dist_names[selected_tries[i][j]] << (j == key_count - 1 ? "" : ", ");
      distributions.push_back(dist_tries[selected_tries[i][j]]);
    }
    std::cout << std::endl;
    for (size_t v = 0; v < val_count; ++v) {
      distributions.push_back(get_default_dist(agg_ops[v]));
    }

    const auto col_count = key_count + val_count;
    std::vector<std::pair<int64_t, int64_t>> ranges;
    for (size_t j = 0; j < key_count; ++j) {
      ranges.push_back(range_tries[selected_tries[i][j]]);
    }
    for (size_t v = 0; v < val_count; ++v) {
      ranges.push_back(get_default_range(agg_ops[v]));
    }

    // Generate random data.
    std::vector<int64_t> input_buffer(row_count * col_count);
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
    int8_t* dev_input_buffer = nullptr;
    cudaMalloc(&dev_input_buffer, input_buffer.size() * sizeof(int64_t));
    if (generate_columns_on_device(
            dev_input_buffer, row_count, col_count, col_widths, ranges, is_columnar, distributions)) {
      cudaMemcpy(&input_buffer[0], dev_input_buffer, input_buffer.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
      generate_columns_on_host(reinterpret_cast<int8_t*>(&input_buffer[0]),
                               row_count,
                               col_count,
                               col_widths,
                               ranges,
                               is_columnar,
                               distributions);
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
      cudaMemcpy(dev_input_buffer, &input_buffer[0], input_buffer.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
#endif
    }
    AggregateEmulator<int64_t, int64_t> emulator(agg_ops);
    auto ref_result =
        emulator.run(reinterpret_cast<int8_t*>(&input_buffer[0]), key_count, val_count, row_count, is_columnar);
    std::cout << "  Generated " << row_count / 1000000.f << "M rows aggregated into " << ref_result.size()
              << " groups.\n";
    const auto actual_group_count = static_cast<size_t>(ref_result.size() * c_space_usage);
    std::vector<int64_t> groups_buffer(actual_group_count * col_count, 0);
#ifdef TRY_COLUMNAR
    std::vector<int64_t> columnar_groups_buffer(actual_group_count * col_count, 0);
#endif
#if defined(TRY_MASH) || defined(TRY_MASH_COLUMNAR)
#ifdef SAVE_MASH_BUF
    const auto actual_col_count = 1 + val_count;
#else
    const auto actual_col_count = col_count;
#endif
#endif
#ifdef TRY_MASH
    std::vector<int64_t> mash_groups_buffer(actual_group_count * actual_col_count, 0);
#endif
#ifdef TRY_MASH_COLUMNAR
    std::vector<int64_t> mash_columnar_groups_buffer(actual_group_count * actual_col_count, 0);
#endif
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
    const auto device_type = DEV_KIND::GPU;
    if (device_type == DEV_KIND::GPU) {
      std::cout << "  Baseline: ";
      try {
        int8_t* dev_groups_buffer = nullptr;
        cudaMalloc(&dev_groups_buffer, groups_buffer.size() * sizeof(int64_t));
        init_groups_on_device(dev_groups_buffer, actual_group_count, col_count, col_widths, init_vals, is_columnar);
        {
          CudaTimer timer(groups_buffer.size() * sizeof(int64_t));
          run_query_on_device(dev_groups_buffer,
                              actual_group_count,
                              dev_input_buffer,
                              row_count,
                              key_count,
                              val_count,
                              col_widths,
                              agg_ops,
                              is_columnar);
        }
        cudaMemcpy(
            &groups_buffer[0], dev_groups_buffer, groups_buffer.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaFree(dev_groups_buffer);
      } catch (const thrust::system_error& e) {
        std::cout << e.what() << std::endl;
      }
#if defined(TRY_MASH) || defined(TRY_MASH_COLUMNAR)
#ifdef SAVE_MASH_BUF
      std::vector<size_t> actual_col_widths(1, sizeof(int64_t));
      std::vector<size_t> actual_init_vals(1, EMPTY_KEY_64);
      for (size_t i = 0; i < val_count; ++i) {
        actual_col_widths.push_back(col_widths[key_count + i]);
        actual_init_vals.push_back(init_vals[key_count + i]);
      }
#else
      const auto& actual_col_widths = col_widths;
      const auto& actual_init_vals = init_vals;
#endif
#endif  // TRY_MASH || TRY_MASH_COLUMNAR
#ifdef TRY_MASH
      std::cout << "  MASH: ";
      try {
        int8_t* dev_mash_groups_buffer = nullptr;
        cudaMalloc(&dev_mash_groups_buffer, mash_groups_buffer.size() * sizeof(int64_t));
        init_groups_on_device(dev_mash_groups_buffer,
                              actual_group_count,
                              actual_col_count,
                              actual_col_widths,
                              actual_init_vals,
                              is_columnar);
        {
          CudaTimer timer;
          mash_run_query_on_device(dev_mash_groups_buffer,
                                   actual_group_count,
                                   dev_input_buffer,
                                   row_count,
                                   key_count,
                                   val_count,
                                   col_widths,
                                   agg_ops,
                                   is_columnar);
        }
        cudaMemcpy(&mash_groups_buffer[0],
                   dev_mash_groups_buffer,
                   mash_groups_buffer.size() * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
#ifdef SAVE_MASH_BUF
        if (key_count > 1) {
          std::vector<int64_t> temp_groups_buffer(actual_group_count * col_count, 0);
          auto elapsedTime = measure<>::execution([&]() {
            mash_restore_keys(reinterpret_cast<int8_t*>(&temp_groups_buffer[0]),
                              reinterpret_cast<int8_t*>(&mash_groups_buffer[0]),
                              actual_group_count,
                              reinterpret_cast<int8_t*>(&input_buffer[0]),
                              row_count,
                              key_count,
                              col_widths,
                              init_vals);
          });
          std::cout << "  \tAnd optional " << elapsedTime << " ms on host if using "
                    << mash_groups_buffer.size() * sizeof(int64_t) / (1024 * 1024.f) << " MB VRAM instead.\n";
          mash_groups_buffer.swap(temp_groups_buffer);
        }
#endif
        cudaFree(dev_mash_groups_buffer);
      } catch (const thrust::system_error& e) {
        std::cout << e.what() << std::endl;
      }
#endif  // TRY_MASH
#ifdef TRY_COLUMNAR
      std::cout << "  Baseline Columnar: ";
      try {
        const bool is_columnar = true;
        int8_t* dev_groups_buffer = nullptr;
        cudaMalloc(&dev_groups_buffer, columnar_groups_buffer.size() * sizeof(int64_t));
#if 0
        int8_t* dev_columnar_input_buffer = nullptr;
        cudaMalloc(&dev_columnar_input_buffer, input_buffer.size() * sizeof(int64_t));
        columnarize_groups_on_device(dev_columnar_input_buffer, dev_input_buffer, row_count, col_widths);
#else
        std::vector<int64_t> columnar_input_buffer(input_buffer.size());
        columnarize_groups_on_host(reinterpret_cast<int8_t*>(&columnar_input_buffer[0]),
                                   reinterpret_cast<const int8_t*>(&input_buffer[0]),
                                   row_count,
                                   col_widths);
        cudaMemcpy(dev_input_buffer,
                   &columnar_input_buffer[0],
                   columnar_input_buffer.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
#endif
        init_groups_on_device(dev_groups_buffer, actual_group_count, col_count, col_widths, init_vals, is_columnar);
        {
          CudaTimer timer(columnar_groups_buffer.size() * sizeof(int64_t));
          run_query_on_device(dev_groups_buffer,
                              actual_group_count,
                              dev_input_buffer,
                              row_count,
                              key_count,
                              val_count,
                              col_widths,
                              agg_ops,
                              is_columnar);
        }
        cudaMemcpy(&columnar_groups_buffer[0],
                   dev_groups_buffer,
                   columnar_groups_buffer.size() * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(dev_groups_buffer);
      } catch (const thrust::system_error& e) {
        std::cout << e.what() << std::endl;
      }
#endif  // TRY_COLUMNAR
#ifdef TRY_MASH_COLUMNAR
      std::cout << "  MASH Columnar: ";
      try {
        const bool is_columnar = true;
        int8_t* dev_mash_groups_buffer = nullptr;
        cudaMalloc(&dev_mash_groups_buffer, mash_columnar_groups_buffer.size() * sizeof(int64_t));
        std::vector<int64_t> columnar_input_buffer(input_buffer.size());
        columnarize_groups_on_host(reinterpret_cast<int8_t*>(&columnar_input_buffer[0]),
                                   reinterpret_cast<const int8_t*>(&input_buffer[0]),
                                   row_count,
                                   col_widths);
        cudaMemcpy(dev_input_buffer,
                   &columnar_input_buffer[0],
                   columnar_input_buffer.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
        init_groups_on_device(dev_mash_groups_buffer,
                              actual_group_count,
                              actual_col_count,
                              actual_col_widths,
                              actual_init_vals,
                              is_columnar);
        {
          CudaTimer timer;
          mash_run_query_on_device(dev_mash_groups_buffer,
                                   actual_group_count,
                                   dev_input_buffer,
                                   row_count,
                                   key_count,
                                   val_count,
                                   col_widths,
                                   agg_ops,
                                   is_columnar);
        }
        cudaMemcpy(&mash_columnar_groups_buffer[0],
                   dev_mash_groups_buffer,
                   mash_columnar_groups_buffer.size() * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
#ifdef SAVE_MASH_BUF
        if (key_count > 1) {
          std::vector<int64_t> temp_groups_buffer(actual_group_count * col_count, 0);
          auto elapsedTime = measure<>::execution([&]() {
            mash_restore_keys<true>(reinterpret_cast<int8_t*>(&temp_groups_buffer[0]),
                                    reinterpret_cast<int8_t*>(&mash_columnar_groups_buffer[0]),
                                    actual_group_count,
                                    reinterpret_cast<int8_t*>(&columnar_input_buffer[0]),
                                    row_count,
                                    key_count,
                                    col_widths,
                                    init_vals);
          });
          std::cout << "  \t\t And optional " << elapsedTime << " ms on host if using "
                    << mash_columnar_groups_buffer.size() * sizeof(int64_t) / (1024 * 1024.f) << " MB VRAM instead.\n";
          mash_columnar_groups_buffer.swap(temp_groups_buffer);
        }
#endif
        cudaFree(dev_mash_groups_buffer);
      } catch (const thrust::system_error& e) {
        std::cout << e.what() << std::endl;
      }
#endif  // TRY_MASH_COLUMNAR
    } else
#endif  // HAVE_CUDA
    {
      init_groups_on_host(reinterpret_cast<int8_t*>(&groups_buffer[0]),
                          actual_group_count,
                          col_count,
                          col_widths,
                          init_vals,
                          is_columnar);
      auto elapsedTime = measure<>::execution([&]() {
        // Do calculation on host
      });
      std::cout << "  Current query took " << elapsedTime << " ms on host\n";
    }
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
    CHECK(dev_input_buffer);
    cudaFree(dev_input_buffer);
    // TODO(miyu): enable this after profiling aggregation on host is added.
    ASSERT_TRUE(emulator.compare(reinterpret_cast<int8_t*>(&groups_buffer[0]),
                                 key_count,
                                 val_count,
                                 actual_group_count,
                                 is_columnar,
                                 ref_result));
#endif
#ifdef TRY_COLUMNAR
    ASSERT_TRUE(emulator.compare(reinterpret_cast<int8_t*>(&columnar_groups_buffer[0]),
                                 key_count,
                                 val_count,
                                 actual_group_count,
                                 true,
                                 ref_result));
#endif
#ifdef TRY_MASH
    ASSERT_TRUE(emulator.compare(reinterpret_cast<int8_t*>(&mash_groups_buffer[0]),
                                 key_count,
                                 val_count,
                                 actual_group_count,
                                 is_columnar,
                                 ref_result));
#endif
#ifdef TRY_MASH_COLUMNAR
    ASSERT_TRUE(emulator.compare(reinterpret_cast<int8_t*>(&mash_columnar_groups_buffer[0]),
                                 key_count,
                                 val_count,
                                 actual_group_count,
                                 true,
                                 ref_result));
#endif
  }
}

namespace {

template <typename KeyT = int64_t>
void reset_entry(KeyT* entry_ptr) {
  static_assert(std::is_same<KeyT, int64_t>::value, "Unsupported template parameter other than int64_t for now");
  *entry_ptr = static_cast<KeyT>(EMPTY_KEY_64);
}

template <bool isColumnar, typename KeyT = int64_t>
class Deduplicater {
 public:
  Deduplicater(int8_t* row_buff, const size_t row_size, const size_t row_count, const size_t key_count)
      : buff_(row_buff), entry_sz_(row_size), entry_cnt_(row_count), key_cnt_(key_count) {}
  size_t run() {
    std::vector<std::future<void>> child_threads;
    const size_t cpu_count = cpu_threads();
    const size_t stride = (entry_cnt_ + cpu_count - 1) / cpu_count;

    std::vector<std::unordered_set<std::vector<KeyT>>> mask_set(cpu_count, std::unordered_set<std::vector<KeyT>>());
    std::vector<std::mutex> mutex_set(cpu_count);
    for (size_t start_entry = 0, i = 0; start_entry < entry_cnt_; start_entry += stride, ++i) {
      const auto end_entry = std::min(entry_cnt_, start_entry + stride);
      child_threads.push_back(std::async(std::launch::async,
                                         &Deduplicater::runDispatch,
                                         this,
                                         std::ref(mask_set),
                                         std::ref(mutex_set),
                                         start_entry,
                                         end_entry));
    }

    for (auto& child : child_threads) {
      child.get();
    }

    size_t row_count = 0;
    for (auto& mask : mask_set) {
      row_count += mask.size();
    }
    CHECK_GE(entry_cnt_, row_count);
    return row_count;
  }

 private:
  int8_t* buff_;
  const size_t entry_sz_;
  const size_t entry_cnt_;
  const size_t key_cnt_;

  void runDispatch(std::vector<std::unordered_set<std::vector<KeyT>>>& mask_set,
                   std::vector<std::mutex>& mutex_set,
                   const size_t start_entry,
                   const size_t end_entry) {
    CHECK_EQ(mask_set.size(), mutex_set.size());
    const size_t set_size = mask_set.size();
    for (size_t i = start_entry; i < end_entry; ++i) {
      std::vector<KeyT> keys(key_cnt_);
      auto key_buffers = reinterpret_cast<KeyT*>(buff_);
      if (isColumnar) {
        for (size_t k = 0; k < key_cnt_; ++k) {
          keys[k] = key_buffers[i + k * entry_cnt_];
        }
      } else {
        for (size_t k = 0; k < key_cnt_; ++k) {
          keys[k] = reinterpret_cast<const KeyT*>(buff_ + i * entry_sz_)[k];
        }
      }
      CHECK_EQ(keys.size(), key_cnt_);
      const size_t mask_idx = std::hash<decltype(keys)>()(keys) % set_size;
      const bool inserted = [&]() {
        std::lock_guard<std::mutex> mask_lock(mutex_set[mask_idx]);
        auto it_ok = mask_set[mask_idx].insert(keys);
        return it_ok.second;
      }();
      if (!inserted) {
        if (isColumnar) {
          for (size_t k = 0; k < key_cnt_; ++k) {
            reset_entry(key_buffers + i + k * entry_cnt_);
          }
        } else {
          for (size_t k = 0; k < key_cnt_; ++k) {
            reset_entry(reinterpret_cast<KeyT*>(buff_ + i * entry_sz_) + k);
          }
        }
      }
    }
  }
};

}  // namespace

TEST(Reduction, Baseline) {
  // Config
  std::vector<OP_KIND> agg_ops{OP_SUM, OP_MAX};
  const size_t key_count = 2;
  const size_t val_count = 2;
  const size_t entry_count = 20000000;
  const bool is_columnar = false;
  const size_t result_count = std::max(size_t(2), get_gpu_count());
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  const float fill_rate = 0.5f;
#endif

  const size_t col_count = key_count + val_count;
  const std::vector<size_t> col_widths(col_count, sizeof(int64_t));
  std::vector<size_t> init_vals(key_count, EMPTY_KEY_64);
  for (size_t i = 0; i < val_count; ++i) {
    init_vals.push_back(get_default_value(agg_ops[i]));
  }
  std::vector<TargetInfo> target_infos;
  const SQLTypeInfo bigint_ti(kBIGINT, true);
  switch (val_count) {
    case 3:
      target_infos.push_back(TargetInfo{true, kMIN, bigint_ti, bigint_ti, true, false});
    case 2:
      target_infos.push_back(TargetInfo{true, kMAX, bigint_ti, bigint_ti, true, false});
    case 1:
      target_infos.push_back(TargetInfo{true, kSUM, bigint_ti, bigint_ti, true, false});
      break;
    default:
      CHECK(false);
  }
  std::reverse(target_infos.begin(), target_infos.end());

  const auto device_type = ExecutorDeviceType::CPU;
  QueryMemoryDescriptor query_mem_desc{};
  query_mem_desc.keyless_hash = false;
  query_mem_desc.has_nulls = false;
  CHECK_GT(key_count, 1);
  query_mem_desc.hash_type = GroupByColRangeType::MultiCol;
  query_mem_desc.output_columnar = is_columnar;
  query_mem_desc.entry_count = entry_count;
  size_t row_size = 0;
  for (size_t k = 0; k < key_count; ++k) {
    query_mem_desc.group_col_widths.emplace_back(sizeof(int64_t));
    row_size += sizeof(int64_t);
  }
  for (const auto& target_info : target_infos) {
    const auto slot_bytes = std::max(int8_t(8), static_cast<int8_t>(target_info.sql_type.get_size()));
    query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
    row_size += slot_bytes;
  }

#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  const bool has_multi_gpus = get_gpu_count() > 1;
  const auto input_size = query_mem_desc.getBufferSizeBytes(device_type);
#else
  const bool has_multi_gpus = false;
#endif  // HAVE_CUDA
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  std::vector<std::unique_ptr<ResultSet>> results;
  for (size_t i = 0; i < result_count; ++i) {
    auto rs = boost::make_unique<ResultSet>(target_infos, device_type, query_mem_desc, row_set_mem_owner, nullptr);
    rs->allocateStorage();
    results.push_back(std::move(rs));
  }

  std::vector<std::pair<int64_t, int64_t>> ranges;
  for (size_t k = 0; k < key_count; ++k) {
    ranges.push_back({-(entry_count / 2), (entry_count / 2)});
  }

  for (size_t v = 0; v < val_count; ++v) {
    ranges.push_back(get_default_range(agg_ops[v]));
  }
  std::vector<DIST_KIND> distributions(col_count, DIST_KIND::UNI);

  std::cout << "ResultSet Count: " << results.size() << std::endl;
  std::vector<size_t> rs_row_counts(results.size(), entry_count);
  // Generate random data.
  auto gen_func = [&](int8_t* input_buffer, const size_t device_id) -> size_t {
    auto actual_row_count = entry_count;
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
    if (has_multi_gpus) {
      cudaSetDevice(device_id);
    }
    int8_t* dev_input_buffer = nullptr;
    cudaMalloc(&dev_input_buffer, input_size);
    if (generate_columns_on_device(
            dev_input_buffer, entry_count, col_count, col_widths, ranges, is_columnar, distributions)) {
      actual_row_count = deduplicate_rows_on_device(dev_input_buffer, entry_count, key_count, col_widths, is_columnar);
      auto dev_input_copy =
          get_hashed_copy(dev_input_buffer, entry_count, entry_count, col_widths, agg_ops, init_vals, is_columnar);
      cudaFree(dev_input_buffer);
      actual_row_count = drop_rows(dev_input_copy, entry_count, row_size, actual_row_count, fill_rate, is_columnar);
      cudaMemcpy(input_buffer, dev_input_copy, input_size, cudaMemcpyDeviceToHost);
    } else
#endif
    {
      generate_columns_on_host(input_buffer, entry_count, col_count, col_widths, ranges, is_columnar, distributions);
      actual_row_count = Deduplicater<false>(input_buffer, row_size, entry_count, key_count).run();
    }
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
    if (dev_input_buffer) {
      cudaFree(dev_input_buffer);
    }
#endif
    return actual_row_count;
  };
  if (has_multi_gpus) {
    std::vector<std::future<size_t>> gener_threads;
    for (size_t i = 0; i < results.size(); ++i) {
      gener_threads.push_back(
          std::async(std::launch::async, gen_func, results[i]->getStorage()->getUnderlyingBuffer(), i));
    }

    for (size_t i = 0; i < gener_threads.size(); ++i) {
      rs_row_counts[i] = gener_threads[i].get();
    }
  } else {
    for (size_t i = 0; i < results.size(); ++i) {
      rs_row_counts[i] = gen_func(results[i]->getStorage()->getUnderlyingBuffer(), i);
    }
  }

  for (size_t i = 0; i < rs_row_counts.size(); ++i) {
    std::cout << "ResultSet " << i << " has " << rs_row_counts[i] << " rows and " << entry_count - rs_row_counts[i]
              << " empty buckets\n";
  }
  AggregateEmulator<int64_t, int64_t> emulator(agg_ops);
  std::vector<decltype(emulator)::ResultType> ref_results;
  for (auto& rs : results) {
    auto ref_rs = emulator.run(rs->getStorage()->getUnderlyingBuffer(), key_count, val_count, entry_count, is_columnar);
    ref_results.push_back(std::move(ref_rs));
  }
  auto ref_reduced_result = emulator.reduce(ref_results);
  ResultSetManager rs_manager;
  std::vector<ResultSet*> storage_set;
  for (auto& rs : results) {
    storage_set.push_back(rs.get());
  }
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  CHECK_GT(results.size(), 0);
  std::vector<int64_t> gpu_reduced_result(input_size / sizeof(int64_t), 0);
  memcpy(&gpu_reduced_result[0], results[0]->getStorage()->getUnderlyingBuffer(), input_size);
#endif
  ResultSet* reduced_result = nullptr;
  std::cout << "CPU reduction: ";
  auto elapsedTime = measure<>::execution([&]() {
    // Do calculation on host
    reduced_result = rs_manager.reduce(storage_set);
  });
  CHECK(reduced_result != nullptr);
  std::cout << "Current reduction took " << elapsedTime << " ms and got reduced " << reduced_result->rowCount()
            << " rows\n";
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  std::vector<int8_t*> host_reduced_buffers(result_count, nullptr);
  host_reduced_buffers[0] = reinterpret_cast<int8_t*>(&gpu_reduced_result[0]);
  for (size_t i = 1; i < storage_set.size(); ++i) {
    host_reduced_buffers[i] = storage_set[i]->getStorage()->getUnderlyingBuffer();
  }
  std::vector<int8_t*> dev_reduced_buffers(result_count, nullptr);
  std::vector<size_t> rs_entry_count(result_count, entry_count);
  std::cout << "GPU reduction: ";
  elapsedTime = measure<>::execution([&]() {
    for (size_t device_id = 0; device_id < result_count; ++device_id) {
      if (has_multi_gpus) {
        cudaSetDevice(device_id);
      }
      int8_t* dev_reduced_buffer = nullptr;
      cudaMalloc(&dev_reduced_buffer, input_size);
      cudaMemcpy(dev_reduced_buffer, host_reduced_buffers[device_id], input_size, cudaMemcpyHostToDevice);
      dev_reduced_buffers[device_id] = dev_reduced_buffer;
    }
    for (size_t stride = 1, end = (result_count + 1) / 2; stride <= end; stride <<= 1) {
      std::vector<std::future<void>> reducer_threads;
      for (size_t device_id = 0; device_id + stride < result_count; device_id += stride * 2) {
        reducer_threads.push_back(std::async(std::launch::async,
                                             [&](const size_t dev_id) {
                                               if (has_multi_gpus) {
                                                 cudaSetDevice(dev_id);
                                               }
                                               reduce_on_device(dev_reduced_buffers[dev_id],
                                                                dev_id,
                                                                rs_entry_count[dev_id],
                                                                dev_reduced_buffers[dev_id + stride],
                                                                has_multi_gpus ? dev_id + stride : dev_id,
                                                                rs_entry_count[dev_id + stride],
                                                                rs_row_counts[dev_id + stride],
                                                                col_widths,
                                                                agg_ops,
                                                                init_vals,
                                                                is_columnar);
                                             },
                                             device_id));
      }
      for (auto& child : reducer_threads) {
        child.get();
      }
    }
  });
  std::cout << "Current reduction took " << elapsedTime << " ms\n";
  {
    std::vector<int64_t> temp_buffer(rs_entry_count[0] * col_count, 0);
    cudaMemcpy(&temp_buffer[0], dev_reduced_buffers[0], temp_buffer.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < dev_reduced_buffers.size(); ++i) {
      if (has_multi_gpus) {
        cudaSetDevice(i);
      }
      cudaFree(dev_reduced_buffers[i]);
      dev_reduced_buffers[i] = nullptr;
    }
    gpu_reduced_result.swap(temp_buffer);
  }
#endif
  ASSERT_TRUE(emulator.compare(reduced_result->getStorage()->getUnderlyingBuffer(),
                               key_count,
                               val_count,
                               reduced_result->getQueryMemDesc().entry_count,
                               is_columnar,
                               ref_reduced_result));
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  ASSERT_TRUE(emulator.compare(reinterpret_cast<int8_t*>(&gpu_reduced_result[0]),
                               key_count,
                               val_count,
                               gpu_reduced_result.size() / col_count,
                               is_columnar,
                               ref_reduced_result));

#endif
}

TEST(Reduction, PerfectHash) {
  // Config
  std::vector<OP_KIND> agg_ops{OP_SUM, OP_MAX};
  const size_t key_count = 1;
  const size_t val_count = 2;
  const size_t entry_count = 2000000;
  const bool is_columnar = false;
  const size_t result_count = std::max(size_t(2), get_gpu_count());

  const size_t col_count = key_count + val_count;
  const std::vector<size_t> col_widths(col_count, sizeof(int64_t));
  std::vector<size_t> init_vals(key_count, EMPTY_KEY_64);
  for (size_t i = 0; i < val_count; ++i) {
    init_vals.push_back(get_default_value(agg_ops[i]));
  }
  std::vector<TargetInfo> target_infos;
  const SQLTypeInfo bigint_ti(kBIGINT, true);
  switch (val_count) {
    case 3:
      target_infos.push_back(TargetInfo{true, kMIN, bigint_ti, bigint_ti, true, false});
    case 2:
      target_infos.push_back(TargetInfo{true, kMAX, bigint_ti, bigint_ti, true, false});
    case 1:
      target_infos.push_back(TargetInfo{true, kSUM, bigint_ti, bigint_ti, true, false});
      break;
    default:
      CHECK(false);
  }
  std::reverse(target_infos.begin(), target_infos.end());

  const auto device_type = ExecutorDeviceType::CPU;
  QueryMemoryDescriptor query_mem_desc{};
  query_mem_desc.keyless_hash = false;
  query_mem_desc.has_nulls = false;
  query_mem_desc.hash_type =
      key_count == 1 ? GroupByColRangeType::OneColKnownRange : GroupByColRangeType::MultiColPerfectHash;
  query_mem_desc.output_columnar = is_columnar;
  query_mem_desc.entry_count = entry_count;
  size_t row_size = 0;
  for (size_t k = 0; k < key_count; ++k) {
    query_mem_desc.group_col_widths.emplace_back(sizeof(int64_t));
    row_size += sizeof(int64_t);
  }
  for (const auto& target_info : target_infos) {
    const auto slot_bytes = std::max(int8_t(8), static_cast<int8_t>(target_info.sql_type.get_size()));
    query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
    row_size += slot_bytes;
  }

#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  const bool has_multi_gpus = get_gpu_count() > 1;
  const auto input_size = query_mem_desc.getBufferSizeBytes(device_type);
#else
  const bool has_multi_gpus = false;
#endif  // HAVE_CUDA
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  std::vector<std::unique_ptr<ResultSet>> results;
  for (size_t i = 0; i < result_count; ++i) {
    auto rs = boost::make_unique<ResultSet>(target_infos, device_type, query_mem_desc, row_set_mem_owner, nullptr);
    rs->allocateStorage();
    results.push_back(std::move(rs));
  }

  std::vector<std::pair<int64_t, int64_t>> ranges(
      key_count, {0, (static_cast<int64_t>(std::exp((std::log(entry_count) / key_count))) - 1)});

  for (size_t v = 0; v < val_count; ++v) {
    ranges.push_back(get_default_range(agg_ops[v]));
  }
  std::vector<DIST_KIND> distributions(col_count, DIST_KIND::UNI);

  std::cout << "ResultSet Count: " << results.size() << std::endl;
  std::vector<size_t> rs_row_counts(results.size(), entry_count);
  // Generate random data.
  auto gen_func = [&](int8_t* input_buffer, const size_t device_id) -> size_t {
    auto actual_row_count = entry_count;
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
    if (has_multi_gpus) {
      cudaSetDevice(device_id);
    }
    int8_t* dev_input_buffer = nullptr;
    cudaMalloc(&dev_input_buffer, input_size);
    if (generate_columns_on_device(
            dev_input_buffer, entry_count, col_count, col_widths, ranges, is_columnar, distributions)) {
      int8_t* dev_input_copy = nullptr;
      std::tie(dev_input_copy, actual_row_count) =
          get_perfect_hashed_copy(dev_input_buffer, entry_count, col_widths, ranges, agg_ops, init_vals, is_columnar);
      cudaFree(dev_input_buffer);
      cudaMemcpy(input_buffer, dev_input_copy, input_size, cudaMemcpyDeviceToHost);
    } else
#endif
    {
      generate_columns_on_host(input_buffer, entry_count, col_count, col_widths, ranges, is_columnar, distributions);
      actual_row_count = Deduplicater<false>(input_buffer, row_size, entry_count, key_count).run();
    }
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
    if (dev_input_buffer) {
      cudaFree(dev_input_buffer);
    }
#endif
    return actual_row_count;
  };

  if (has_multi_gpus) {
    std::vector<std::future<size_t>> gener_threads;
    for (size_t i = 0; i < results.size(); ++i) {
      gener_threads.push_back(
          std::async(std::launch::async, gen_func, results[i]->getStorage()->getUnderlyingBuffer(), i));
    }

    for (size_t i = 0; i < gener_threads.size(); ++i) {
      rs_row_counts[i] = gener_threads[i].get();
    }
  } else {
    for (size_t i = 0; i < results.size(); ++i) {
      rs_row_counts[i] = gen_func(results[i]->getStorage()->getUnderlyingBuffer(), i);
    }
  }

  for (size_t i = 0; i < rs_row_counts.size(); ++i) {
    std::cout << "ResultSet " << i << " has " << rs_row_counts[i] << " rows and " << entry_count - rs_row_counts[i]
              << " empty buckets\n";
  }
  AggregateEmulator<int64_t, int64_t> emulator(agg_ops);
  std::vector<decltype(emulator)::ResultType> ref_results;
  for (auto& rs : results) {
    auto ref_rs = emulator.run(rs->getStorage()->getUnderlyingBuffer(), key_count, val_count, entry_count, is_columnar);
    ref_results.push_back(std::move(ref_rs));
  }
  auto ref_reduced_result = emulator.reduce(ref_results);
  ResultSetManager rs_manager;
  std::vector<ResultSet*> storage_set;
  for (auto& rs : results) {
    storage_set.push_back(rs.get());
  }
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  CHECK_GT(results.size(), 0);
  std::vector<int64_t> gpu_reduced_result(input_size / sizeof(int64_t), 0);
  memcpy(&gpu_reduced_result[0], results[0]->getStorage()->getUnderlyingBuffer(), input_size);
#endif
  ResultSet* reduced_result = nullptr;
  std::cout << "CPU reduction: ";
  auto elapsedTime = measure<>::execution([&]() {
    // Do calculation on host
    reduced_result = rs_manager.reduce(storage_set);
  });
  CHECK(reduced_result != nullptr);
  std::cout << "Current reduction took " << elapsedTime << " ms and got reduced " << reduced_result->rowCount()
            << " rows\n";
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  std::vector<int8_t*> host_reduced_buffers(result_count, nullptr);
  host_reduced_buffers[0] = reinterpret_cast<int8_t*>(&gpu_reduced_result[0]);
  for (size_t i = 1; i < storage_set.size(); ++i) {
    host_reduced_buffers[i] = storage_set[i]->getStorage()->getUnderlyingBuffer();
  }
  std::vector<int8_t*> dev_reduced_buffers(result_count, nullptr);
  std::vector<int8_t*> dev_seg_copies(result_count, nullptr);
  const auto seg_count = has_multi_gpus ? result_count : size_t(1);
  const auto stride = (entry_count + (seg_count - 1)) / seg_count;

  std::cout << "GPU reduction: ";
  elapsedTime = measure<>::execution([&]() {
    std::vector<std::future<void>> uploader_threads;
    for (size_t device_id = 0; device_id < result_count; ++device_id) {
      uploader_threads.push_back(
          std::async(std::launch::async,
                     [&](const size_t dev_id) {
                       if (has_multi_gpus) {
                         cudaSetDevice(dev_id);
                       }
                       int8_t* dev_reduced_buffer = nullptr;
                       cudaMalloc(&dev_reduced_buffer, input_size);
                       cudaMemcpy(dev_reduced_buffer, host_reduced_buffers[dev_id], input_size, cudaMemcpyHostToDevice);
                       dev_reduced_buffers[dev_id] = dev_reduced_buffer;
                     },
                     device_id));
    }
    for (auto& child : uploader_threads) {
      child.get();
    }
  });
  std::cout << "Current reduction took " << elapsedTime << " ms to upload to VRAM and ";

  elapsedTime = measure<>::execution([&]() {
    // Redistribute across devices
    if (has_multi_gpus) {
      std::vector<std::future<void>> redis_threads;
      for (size_t device_id = 0, start_entry = 0; device_id < result_count; ++device_id, start_entry += stride) {
        const auto end_entry = std::min(start_entry + stride, entry_count);
        redis_threads.push_back(std::async(
            std::launch::async,
            [&](const size_t dev_id, const size_t start, const size_t end) {
              cudaSetDevice(dev_id);
              dev_seg_copies[dev_id] = fetch_segs_from_others(
                  dev_reduced_buffers, entry_count, dev_id, result_count, col_widths, is_columnar, start, end);
            },
            device_id,
            start_entry,
            end_entry));
      }
      for (auto& child : redis_threads) {
        child.get();
      }
    } else {
      CHECK_EQ(dev_reduced_buffers.size(), size_t(2));
      dev_seg_copies[0] = dev_reduced_buffers[1];
    }
    // Reduce
    std::vector<std::future<void>> reducer_threads;
    for (size_t device_id = 0, start_entry = 0; device_id < seg_count; ++device_id, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      reducer_threads.push_back(std::async(std::launch::async,
                                           [&](const size_t dev_id, const size_t start, const size_t end) {
                                             if (has_multi_gpus) {
                                               cudaSetDevice(dev_id);
                                             }
                                             reduce_segment_on_device(dev_reduced_buffers[dev_id],
                                                                      dev_seg_copies[dev_id],
                                                                      entry_count,
                                                                      seg_count,
                                                                      col_widths,
                                                                      agg_ops,
                                                                      is_columnar,
                                                                      start,
                                                                      end);
                                           },
                                           device_id,
                                           start_entry,
                                           end_entry));
    }
    for (auto& child : reducer_threads) {
      child.get();
    }
  });
  std::cout << elapsedTime << " ms to reduce.\n";
  {
    for (size_t device_id = 0, start = 0; device_id < seg_count; ++device_id, start += stride) {
      const auto end = std::min(start + stride, entry_count);
      if (has_multi_gpus) {
        cudaSetDevice(device_id);
        cudaFree(dev_seg_copies[device_id]);
        dev_seg_copies[device_id] = nullptr;
      }
      if (is_columnar) {
        for (size_t c = 0, col_base = start; c < col_count; ++c, col_base += entry_count) {
          cudaMemcpy(&gpu_reduced_result[col_base],
                     dev_reduced_buffers[device_id] + col_base * sizeof(int64_t),
                     (end - start) * sizeof(int64_t),
                     cudaMemcpyDeviceToHost);
        }
      } else {
        cudaMemcpy(&gpu_reduced_result[start * col_count],
                   dev_reduced_buffers[device_id] + start * row_size,
                   (end - start) * row_size,
                   cudaMemcpyDeviceToHost);
      }
      cudaFree(dev_reduced_buffers[device_id]);
      dev_reduced_buffers[device_id] = nullptr;
    }
  }
#endif
  ASSERT_TRUE(emulator.compare(reduced_result->getStorage()->getUnderlyingBuffer(),
                               key_count,
                               val_count,
                               reduced_result->getQueryMemDesc().entry_count,
                               is_columnar,
                               ref_reduced_result));
#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
  ASSERT_TRUE(emulator.compare(reinterpret_cast<int8_t*>(&gpu_reduced_result[0]),
                               key_count,
                               val_count,
                               gpu_reduced_result.size() / col_count,
                               is_columnar,
                               ref_reduced_result));

#endif
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  g_gpus_present = is_gpu_present();
#ifndef HAVE_CUDA
  testing::GTEST_FLAG(filter) = "-Hash.Baseline";
#endif
  auto err = RUN_ALL_TESTS();
  return err;
}
