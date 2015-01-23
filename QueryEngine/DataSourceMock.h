#ifndef QUERYENGINE_DATASOURCE_H
#define QUERYENGINE_DATASOURCE_H

#include "Shared/sqltypes.h"
#include "Shared/types.h"

#include <memory>


struct FragmentInfo {
  const int fragment_id;
  const size_t num_tuples;
};

std::vector<FragmentInfo> get_fragments(const int table_id) {
  std::vector<FragmentInfo> result;
  for (int i = 0; i < 1; ++i) {
    result.push_back(FragmentInfo { i, 1 * 1000 * 1000L });
  }
  return result;
}

struct ChunkStatistics {};

ChunkStatistics get_chunk_stats(const ChunkKey&, const size_t num_tuples);

std::vector<ChunkStatistics> get_chunk_stats_multi(const std::vector<std::pair<const ChunkKey&, const size_t>>&);

template<class T>
class FixedWidthMockDataBuffer {
public:
  FixedWidthMockDataBuffer(const size_t num_rows) : num_rows_(num_rows) {
    data_ = new T[num_rows_];
    for (size_t i = 0; i < num_rows_; ++i) {
      data_[i] = 42;
    }
  }
  ~FixedWidthMockDataBuffer() {
    delete data_;
  }
  const int8_t* data() const {
    return reinterpret_cast<int8_t*>(data_);
  }
private:
  T* data_;
  const size_t num_rows_;
};

template<class T>
class MockDataBuffer {
public:
  MockDataBuffer(const size_t p_num_rows, const int p_device_id)
    : fixed_width_data_(new FixedWidthMockDataBuffer<T>(p_num_rows))
    , data(fixed_width_data_->data())
    , num_rows(p_num_rows)
    , compression(kENCODING_FIXED)
    , comp_param(sizeof(T))
    , device_id(p_device_id) {}
private:
  std::unique_ptr<FixedWidthMockDataBuffer<T>> fixed_width_data_;
public:
  const int8_t* data;
  const size_t num_rows;
  const EncodingType compression;
  const int comp_param;
  const int device_id;  // either the core or the GPU index
};

enum class AddressSpace {
  CPU,
  GPU
};

template<class T>
MockDataBuffer<T> get_chunk(const ChunkKey&, const size_t num_tuples, const AddressSpace) {
  return MockDataBuffer<T>(num_tuples, 0);
}

template<class T>
std::vector<MockDataBuffer<T>> get_chunk_multi(
    const std::vector<std::pair<const ChunkKey, const size_t>>& chunk_keys,
    const AddressSpace) {
  std::vector<MockDataBuffer<T>> result;
  for (const auto& chunk : chunk_keys) {
    result.push_back(MockDataBuffer<T>(chunk.second, 0));
  }
  return result;
}

#endif // QUERYENGINE_DATASOURCE_H
