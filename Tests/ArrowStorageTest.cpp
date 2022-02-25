/*
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

#include "ArrowStorage/ArrowStorage.h"

#include "TestHelpers.h"

#include <gtest/gtest.h>

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;

using namespace std::string_literals;

using TestHelpers::inline_null_array_value;

namespace {

[[maybe_unused]] void dumpTableMeta(ArrowStorage& storage, int table_id) {
  std::cout << "Table #" << table_id << std::endl;

  std::cout << "  Schema:";
  auto col_infos = storage.listColumns(TEST_DB_ID, table_id);
  for (auto& col_info : col_infos) {
    std::cout << " " << col_info->name << "[" << col_info->column_id << "]("
              << col_info->type.toString() << ")";
  }
  std::cout << std::endl;

  std::cout << "  Fragments:" << std::endl;
  auto meta = storage.getTableMetadata(TEST_DB_ID, table_id);
  for (auto& frag : meta.fragments) {
    std::cout << "    Fragment #" << frag.fragmentId << " - " << frag.getNumTuples()
              << " row(s)" << std::endl;
    for (int col_id = 1; col_id < col_infos.back()->column_id; ++col_id) {
      auto& chunk_meta = frag.getChunkMetadataMap().at(col_id);
      std::cout << "      col" << col_id << " meta: " << chunk_meta->dump() << std::endl;
    }
  }
}

class TestBuffer : public Data_Namespace::AbstractBuffer {
 public:
  TestBuffer(size_t size) : Data_Namespace::AbstractBuffer(0) {
    size_ = size;
    data_.reset(new int8_t[size]);
  }

  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel dst_buffer_type = CPU_LEVEL,
            const int dst_device_id = -1) override {
    UNREACHABLE();
  }

  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel src_buffer_type = CPU_LEVEL,
             const int src_device_id = -1) override {
    UNREACHABLE();
  }

  void reserve(size_t num_bytes) override { UNREACHABLE(); }

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type = CPU_LEVEL,
              const int device_id = -1) override {
    UNREACHABLE();
  }

  int8_t* getMemoryPtr() override { return data_.get(); }

  void setMemoryPtr(int8_t* new_ptr) override { UNREACHABLE(); }

  size_t pageCount() const override { return size_; }

  size_t pageSize() const override { return 1; }

  size_t reservedSize() const override { return size_; }

  MemoryLevel getType() const override { return MemoryLevel::CPU_LEVEL; }

 protected:
  std::unique_ptr<int8_t[]> data_;
};

std::string getFilePath(const std::string& file_name) {
  return std::string("../../Tests/ArrowStorageDataFiles/") + file_name;
}

template <typename T>
std::vector<T> duplicate(const std::vector<T>& v) {
  std::vector<T> res;
  res.reserve(v.size() * 2);
  res.insert(res.end(), v.begin(), v.end());
  res.insert(res.end(), v.begin(), v.end());
  return res;
}

template <typename T>
std::vector<T> range(size_t size, T step) {
  std::vector<T> res(size);
  for (size_t i = 0; i < size; ++i) {
    res[i] = static_cast<T>((i + 1) * step);
  }
  return res;
}

void checkTableInfo(TableInfoPtr table_info,
                    int db_id,
                    int table_id,
                    const std::string& name,
                    size_t fragments) {
  CHECK_EQ(table_info->db_id, db_id);
  CHECK_EQ(table_info->table_id, table_id);
  CHECK_EQ(table_info->name, name);
  CHECK_EQ(table_info->fragments, fragments);
  CHECK_EQ(table_info->is_view, false);
  CHECK_EQ(table_info->persistence_level, Data_Namespace::MemoryLevel::CPU_LEVEL);
}

void checkColumnInfo(ColumnInfoPtr col_info,
                     int db_id,
                     int table_id,
                     int col_id,
                     const std::string& name,
                     const SQLTypeInfo& type,
                     bool is_rowid = false) {
  CHECK_EQ(col_info->db_id, db_id);
  CHECK_EQ(col_info->table_id, table_id);
  CHECK_EQ(col_info->column_id, col_id);
  CHECK_EQ(col_info->name, name);
  CHECK_EQ(col_info->type, type);
  CHECK_EQ(col_info->is_rowid, is_rowid);
}

template <typename T>
void checkDatum(const Datum& actual, T expected, const SQLTypeInfo& type) {
  switch (type.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
      ASSERT_EQ(static_cast<T>(actual.tinyintval), expected);
      break;
    case kSMALLINT:
      ASSERT_EQ(static_cast<T>(actual.smallintval), expected);
      break;
    case kINT:
      ASSERT_EQ(static_cast<T>(actual.intval), expected);
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      ASSERT_EQ(actual.bigintval, static_cast<int64_t>(expected));
      break;
    case kFLOAT:
      CHECK_EQ(static_cast<T>(actual.floatval), expected);
      break;
    case kDOUBLE:
      CHECK_EQ(static_cast<T>(actual.doubleval), expected);
      break;
    case kVARCHAR:
    case kCHAR:
    case kTEXT:
      if (type.get_compression() == kENCODING_DICT) {
        CHECK_EQ(static_cast<T>(actual.intval), expected);
      }
      break;
    default:
      break;
  }
}

void checkChunkMeta(std::shared_ptr<ChunkMetadata> meta,
                    const SQLTypeInfo& type,
                    size_t num_rows,
                    size_t num_bytes,
                    bool has_nulls) {
  CHECK_EQ(meta->sqlType, type);
  CHECK_EQ(meta->numElements, num_rows);
  CHECK_EQ(meta->numBytes, num_bytes);
  CHECK_EQ(meta->chunkStats.has_nulls, has_nulls);
}

template <typename T>
void checkChunkMeta(std::shared_ptr<ChunkMetadata> meta,
                    const SQLTypeInfo& type,
                    size_t num_rows,
                    size_t num_bytes,
                    bool has_nulls,
                    T min,
                    T max) {
  checkChunkMeta(meta, type, num_rows, num_bytes, has_nulls);
  checkDatum(meta->chunkStats.min, min, type);
  checkDatum(meta->chunkStats.max, max, type);
}

template <typename T>
void checkFetchedData(ArrowStorage& storage,
                      int table_id,
                      int col_id,
                      int frag_id,
                      const std::vector<T>& expected,
                      const std::vector<int>& key_suffix = {}) {
  size_t buf_size = expected.size() * sizeof(T);
  TestBuffer dst(buf_size);
  ChunkKey key{TEST_DB_ID, table_id, col_id, frag_id};
  key.insert(key.end(), key_suffix.begin(), key_suffix.end());
  storage.fetchBuffer(key, &dst, buf_size);
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK_EQ(reinterpret_cast<const T*>(dst.getMemoryPtr())[i], expected[i]);
  }
}

template <typename T>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<T>& expected) {
  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  std::vector<T> expected_chunk(expected.begin() + start_row, expected.begin() + end_row);

  auto has_nulls = std::any_of(expected_chunk.begin(),
                               expected_chunk.end(),
                               [](const T& v) { return v == inline_null_value<T>(); });
  auto min =
      std::accumulate(expected_chunk.begin(),
                      expected_chunk.end(),
                      std::numeric_limits<T>::max(),
                      [](T min, T val) -> T {
                        return val == inline_null_value<T>() ? min : std::min(min, val);
                      });
  auto max =
      std::accumulate(expected_chunk.begin(),
                      expected_chunk.end(),
                      std::numeric_limits<T>::min(),
                      [](T max, T val) -> T {
                        return val == inline_null_value<T>() ? max : std::max(max, val);
                      });
  auto col_type = storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1)->type;
  if (col_type.get_type() == kDATE) {
    int64_t date_min = min == std::numeric_limits<T>::max()
                           ? std::numeric_limits<int64_t>::max()
                           : DateConverters::get_epoch_seconds_from_days(min);
    int64_t date_max = max == std::numeric_limits<T>::min()
                           ? std::numeric_limits<int64_t>::min()
                           : DateConverters::get_epoch_seconds_from_days(max);
    checkChunkMeta(chunk_meta_map.at(col_idx + 1),
                   col_type,
                   frag_rows,
                   frag_rows * sizeof(T),
                   has_nulls,
                   date_min,
                   date_max);
  } else {
    checkChunkMeta(chunk_meta_map.at(col_idx + 1),
                   col_type,
                   frag_rows,
                   frag_rows * sizeof(T),
                   has_nulls,
                   min,
                   max);
  }
  checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_chunk);
}

void checkStringColumnData(ArrowStorage& storage,
                           const ChunkMetadataMap& chunk_meta_map,
                           int table_id,
                           size_t row_count,
                           size_t fragment_size,
                           size_t col_idx,
                           size_t frag_idx,
                           const std::vector<std::string>& vals) {
  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  size_t chunk_size = 0;
  for (size_t i = start_row; i < end_row; ++i) {
    chunk_size += vals[i].size();
  }
  checkChunkMeta(chunk_meta_map.at(col_idx + 1),
                 storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1)->type,
                 frag_rows,
                 chunk_size,
                 false);
  std::vector<int8_t> expected_data(chunk_size);
  std::vector<uint32_t> expected_offset(frag_rows + 1);
  uint32_t data_offset = 0;
  for (size_t i = start_row; i < end_row; ++i) {
    expected_offset[i - start_row] = data_offset;
    memcpy(expected_data.data() + data_offset, vals[i].data(), vals[i].size());
    data_offset += vals[i].size();
  }
  expected_offset.back() = data_offset;
  checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_offset, {2});
  checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_data, {1});
}

template <typename IndexType>
void checkStringDictColumnData(ArrowStorage& storage,
                               const ChunkMetadataMap& chunk_meta_map,
                               int table_id,
                               size_t row_count,
                               size_t fragment_size,
                               size_t col_idx,
                               size_t frag_idx,
                               const std::vector<std::string>& expected) {
  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;

  auto col_info = storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1);
  auto& dict =
      *storage.getDictMetadata(TEST_DB_ID, col_info->type.get_comp_param())->stringDict;

  std::vector<IndexType> expected_ids(frag_rows);
  for (size_t i = start_row; i < end_row; ++i) {
    expected_ids[i - start_row] = static_cast<IndexType>(dict.getIdOfString(expected[i]));
  }

  checkChunkMeta(chunk_meta_map.at(col_idx + 1),
                 storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1)->type,
                 frag_rows,
                 frag_rows * sizeof(IndexType),
                 false,
                 *std::min_element(expected_ids.begin(), expected_ids.end()),
                 *std::max_element(expected_ids.begin(), expected_ids.end()));

  checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_ids);
}

template <typename T>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<std::vector<T>>& expected) {
  CHECK_EQ(row_count, expected.size());
  auto col_info = storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1);

  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  size_t chunk_elems = 0;
  for (size_t i = start_row; i < end_row; ++i) {
    if (expected[i].empty() || !col_info->type.is_varlen_array() ||
        expected[i].front() != inline_null_array_value<T>()) {
      chunk_elems += expected[i].size();
    }
  }
  size_t chunk_size = chunk_elems * sizeof(T);
  checkChunkMeta(chunk_meta_map.at(col_idx + 1),
                 storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1)->type,
                 frag_rows,
                 chunk_size,
                 false);
  std::vector<T> expected_data(chunk_elems);
  std::vector<uint32_t> expected_offset(frag_rows + 1);
  uint32_t data_offset = 0;
  bool use_negative_offset = false;
  for (size_t i = start_row; i < end_row; ++i) {
    expected_offset[i - start_row] =
        use_negative_offset ? -data_offset * sizeof(T) : data_offset * sizeof(T);
    if (!expected[i].empty() && expected[i].front() == inline_null_array_value<T>() &&
        col_info->type.is_varlen_array()) {
      use_negative_offset = true;
    } else {
      use_negative_offset = false;
      memcpy(expected_data.data() + data_offset,
             expected[i].data(),
             expected[i].size() * sizeof(T));
      data_offset += expected[i].size();
    }
  }
  expected_offset.back() =
      use_negative_offset ? -data_offset * sizeof(T) : data_offset * sizeof(T);
  if (col_info->type.is_varlen_array()) {
    checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_offset, {2});
    checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_data, {1});
  } else {
    checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_data);
  }
}

template <>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<std::vector<std::string>>& expected) {
  CHECK_EQ(row_count, expected.size());
  auto col_info = storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1);
  auto& dict =
      *storage.getDictMetadata(TEST_DB_ID, col_info->type.get_comp_param())->stringDict;

  size_t start_row = frag_idx * fragment_size;
  size_t end_row = std::min(row_count, start_row + fragment_size);
  size_t frag_rows = end_row - start_row;
  std::vector<int32_t> expected_ids;

  // varlen string arrays are not yet supported.
  CHECK(col_info->type.is_fixlen_array());
  size_t list_size =
      col_info->type.get_size() / col_info->type.get_elem_type().get_size();
  size_t chunk_elems = frag_rows * list_size;
  for (size_t i = start_row; i < end_row; ++i) {
    if (expected[i].empty()) {
      expected_ids.push_back(inline_int_null_array_value<int32_t>());
      for (size_t j = 1; j < list_size; ++j) {
        expected_ids.push_back(inline_int_null_value<int32_t>());
      }
    } else {
      ASSERT_EQ(expected[i].size(), list_size);
      for (auto& val : expected[i]) {
        if (val == "<NULL>") {
          expected_ids.push_back(inline_int_null_value<int32_t>());
        } else {
          expected_ids.push_back(dict.getIdOfString(val));
        }
      }
    }
  }

  size_t chunk_size = chunk_elems * sizeof(int32_t);
  checkChunkMeta(chunk_meta_map.at(col_idx + 1),
                 storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1)->type,
                 frag_rows,
                 chunk_size,
                 false);

  checkFetchedData(storage, table_id, col_idx + 1, frag_idx + 1, expected_ids);
}

template <>
void checkChunkData(ArrowStorage& storage,
                    const ChunkMetadataMap& chunk_meta_map,
                    int table_id,
                    size_t row_count,
                    size_t fragment_size,
                    size_t col_idx,
                    size_t frag_idx,
                    const std::vector<std::string>& expected) {
  CHECK_EQ(row_count, expected.size());
  auto col_info = storage.getColumnInfo(TEST_DB_ID, table_id, col_idx + 1);
  if (col_info->type.is_dict_encoded_string()) {
    switch (col_info->type.get_size()) {
      case 1:
        checkStringDictColumnData<int8_t>(storage,
                                          chunk_meta_map,
                                          table_id,
                                          row_count,
                                          fragment_size,
                                          col_idx,
                                          frag_idx,
                                          expected);
        break;
      case 2:
        checkStringDictColumnData<int16_t>(storage,
                                           chunk_meta_map,
                                           table_id,
                                           row_count,
                                           fragment_size,
                                           col_idx,
                                           frag_idx,
                                           expected);
        break;
      case 4:
        checkStringDictColumnData<int32_t>(storage,
                                           chunk_meta_map,
                                           table_id,
                                           row_count,
                                           fragment_size,
                                           col_idx,
                                           frag_idx,
                                           expected);
        break;
      default:
        GTEST_FAIL();
    }
  } else {
    checkStringColumnData(storage,
                          chunk_meta_map,
                          table_id,
                          row_count,
                          fragment_size,
                          col_idx,
                          frag_idx,
                          expected);
  }
}

void checkColumnData(ArrowStorage& storage,
                     const ChunkMetadataMap& chunk_meta_map,
                     int table_id,
                     size_t row_count,
                     size_t fragment_size,
                     size_t col_idx,
                     size_t frag_idx) {}

template <typename T, typename... Ts>
void checkColumnData(ArrowStorage& storage,
                     const ChunkMetadataMap& chunk_meta_map,
                     int table_id,
                     size_t row_count,
                     size_t fragment_size,
                     size_t col_idx,
                     size_t frag_idx,
                     const std::vector<T>& expected,
                     const std::vector<Ts>&... more_expected) {
  checkChunkData(storage,
                 chunk_meta_map,
                 table_id,
                 row_count,
                 fragment_size,
                 col_idx,
                 frag_idx,
                 expected);

  checkColumnData(storage,
                  chunk_meta_map,
                  table_id,
                  row_count,
                  fragment_size,
                  col_idx + 1,
                  frag_idx,
                  more_expected...);
}

template <typename... Ts>
void checkData(ArrowStorage& storage,
               int table_id,
               size_t row_count,
               size_t fragment_size,
               Ts... expected) {
  size_t frag_count = (row_count + fragment_size - 1) / fragment_size;
  auto meta = storage.getTableMetadata(TEST_DB_ID, table_id);
  CHECK_EQ(meta.getNumTuples(), row_count);
  CHECK_EQ(meta.getPhysicalNumTuples(), row_count);
  CHECK_EQ(meta.fragments.size(), frag_count);
  for (size_t frag_idx = 0; frag_idx < frag_count; ++frag_idx) {
    size_t start_row = frag_idx * fragment_size;
    size_t end_row = std::min(row_count, start_row + fragment_size);
    size_t frag_rows = end_row - start_row;
    CHECK_EQ(meta.fragments[frag_idx].fragmentId, static_cast<int>(frag_idx + 1));
    CHECK_EQ(meta.fragments[frag_idx].physicalTableId, table_id);
    CHECK_EQ(meta.fragments[frag_idx].getNumTuples(), frag_rows);
    CHECK_EQ(meta.fragments[frag_idx].getPhysicalNumTuples(), frag_rows);

    auto chunk_meta_map = meta.fragments[frag_idx].getChunkMetadataMap();
    CHECK_EQ(chunk_meta_map.size(), sizeof...(Ts));
    for (int i = 0; i < static_cast<int>(chunk_meta_map.size()); ++i) {
      CHECK_EQ(chunk_meta_map.count(i + 1), (size_t)1);
    }
    checkColumnData(storage,
                    chunk_meta_map,
                    table_id,
                    row_count,
                    fragment_size,
                    0,
                    frag_idx,
                    expected...);
  }
}

}  // anonymous namespace

class ArrowStorageTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {}

  static void TearDownTestSuite() {}
};

TEST_F(ArrowStorageTest, CreateTable_OK) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto tinfo = storage.createTable("table1",
                                   {{"col1", SQLTypeInfo(kINT)},
                                    {"col2", SQLTypeInfo(kFLOAT)},
                                    {"col3", SQLTypeInfo(kDOUBLE)}});
  checkTableInfo(tinfo, TEST_DB_ID, tinfo->table_id, "table1", 0);
  auto col_infos = storage.listColumns(*tinfo);
  CHECK_EQ(col_infos.size(), (size_t)4);
  checkColumnInfo(
      col_infos[0], TEST_DB_ID, tinfo->table_id, 1, "col1", SQLTypeInfo(kINT));
  checkColumnInfo(
      col_infos[1], TEST_DB_ID, tinfo->table_id, 2, "col2", SQLTypeInfo(kFLOAT));
  checkColumnInfo(
      col_infos[2], TEST_DB_ID, tinfo->table_id, 3, "col3", SQLTypeInfo(kDOUBLE));
  checkColumnInfo(
      col_infos[3], TEST_DB_ID, tinfo->table_id, 4, "rowid", SQLTypeInfo(kBIGINT), true);
}

TEST_F(ArrowStorageTest, CreateTable_EmptyTableName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("", {{"col1", SQLTypeInfo(kINT)}}),
               std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_DuplicatedTableName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_NO_THROW(storage.createTable("table1", {{"col1", SQLTypeInfo(kINT)}}));
  ASSERT_THROW(storage.createTable("table1", {{"col1", SQLTypeInfo(kINT)}}),
               std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_NoColumns) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("table1", {}), std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_DuplicatedColumns) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable(
                   "table1", {{"col1", SQLTypeInfo(kINT)}, {"col1", SQLTypeInfo(kINT)}}),
               std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_EmptyColumnName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("table1", {{"", SQLTypeInfo(kINT)}}),
               std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_ReservedColumnName) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ASSERT_THROW(storage.createTable("table1", {{"rowid", SQLTypeInfo(kINT)}}),
               std::runtime_error);
}

TEST_F(ArrowStorageTest, CreateTable_SharedDict) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  SQLTypeInfo type_dict0(kTEXT, false, kENCODING_DICT);
  type_dict0.set_comp_param(0);
  SQLTypeInfo type_dict1(kTEXT, false, kENCODING_DICT);
  type_dict1.set_comp_param(-1);
  SQLTypeInfo type_dict2(kTEXT, false, kENCODING_DICT);
  type_dict2.set_comp_param(-2);
  auto tinfo = storage.createTable("table1",
                                   {{"col1", type_dict0},
                                    {"col2", type_dict1},
                                    {"col3", type_dict2},
                                    {"col4", type_dict1},
                                    {"col5", type_dict2}});
  auto col_infos = storage.listColumns(*tinfo);
  CHECK_EQ(col_infos[0]->type.get_comp_param(), 1);
  CHECK_EQ(col_infos[1]->type.get_comp_param(), 2);
  CHECK_EQ(col_infos[2]->type.get_comp_param(), 3);
  CHECK_EQ(col_infos[3]->type.get_comp_param(), 2);
  CHECK_EQ(col_infos[4]->type.get_comp_param(), 3);
  CHECK(storage.getDictMetadata(TEST_DB_ID, 1));
  CHECK(storage.getDictMetadata(TEST_DB_ID, 2));
  CHECK(storage.getDictMetadata(TEST_DB_ID, 3));
}

TEST_F(ArrowStorageTest, CreateTable_WrongDictId) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  SQLTypeInfo type(kTEXT, false, kENCODING_DICT);
  type.set_comp_param(1);
  ASSERT_THROW(storage.createTable("table1", {{"col1", type}}), std::runtime_error);
}

TEST_F(ArrowStorageTest, DropTable) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto tinfo = storage.createTable("table1",
                                   {{"col1", SQLTypeInfo(kINT)},
                                    {"col2", SQLTypeInfo(kFLOAT)},
                                    {"col3", SQLTypeInfo(kTEXT, false, kENCODING_DICT)}});
  auto col_infos = storage.listColumns(*tinfo);
  storage.dropTable("table1");
  ASSERT_EQ(storage.getTableInfo(*tinfo), nullptr);
  for (auto& col_info : col_infos) {
    ASSERT_EQ(storage.getColumnInfo(*col_info), nullptr);
    if (col_info->type.is_dict_encoded_string()) {
      ASSERT_EQ(storage.getDictMetadata(TEST_DB_ID, col_info->type.get_comp_param()),
                nullptr);
    }
  }
}

TEST_F(ArrowStorageTest, DropTable_SharedDicts) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  auto tinfo1 = storage.createTable(
      "table1", {{"col1", SQLTypeInfo(kTEXT, false, kENCODING_DICT)}});
  auto col1_info = storage.getColumnInfo(*tinfo1, "col1");

  SQLTypeInfo type_dict1(kTEXT, false, kENCODING_DICT);
  type_dict1.set_comp_param(col1_info->type.get_comp_param());
  SQLTypeInfo type_dict2(kTEXT, false, kENCODING_DICT);
  type_dict2.set_comp_param(-1);
  auto tinfo2 = storage.createTable(
      "table2", {{"col1", type_dict1}, {"col2", type_dict2}, {"col3", type_dict2}});
  auto col2_info = storage.getColumnInfo(*tinfo2, "col2");

  auto col_infos = storage.listColumns(*tinfo2);
  storage.dropTable("table2");
  ASSERT_EQ(storage.getTableInfo(*tinfo2), nullptr);
  for (auto& col_info : col_infos) {
    ASSERT_EQ(storage.getColumnInfo(*col_info), nullptr);
  }

  ASSERT_NE(storage.getDictMetadata(TEST_DB_ID, col1_info->type.get_comp_param()),
            nullptr);
  ASSERT_EQ(storage.getDictMetadata(TEST_DB_ID, col2_info->type.get_comp_param()),
            nullptr);
}

void Test_ImportCsv_Numbers(const std::string& file_name,
                            const ArrowStorage::CsvParseOptions parse_options,
                            bool pass_schema,
                            size_t fragment_size = 32'000'000) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  if (pass_schema) {
    tinfo = storage.importCsvFile(
        getFilePath(file_name),
        "table1",
        {{"col1", SQLTypeInfo(kINT)}, {"col2", SQLTypeInfo(kFLOAT)}},
        table_options,
        parse_options);
    checkData(storage,
              tinfo->table_id,
              9,
              fragment_size,
              range(9, (int32_t)1),
              range(9, 10.0f));
  } else {
    tinfo = storage.importCsvFile(
        getFilePath(file_name), "table1", table_options, parse_options);
    checkData(
        storage, tinfo->table_id, 9, fragment_size, range(9, (int64_t)1), range(9, 10.0));
  }
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_Header) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_NoHeader) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  Test_ImportCsv_Numbers("numbers_noheader.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_Delim) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.delimiter = '|';
  Test_ImportCsv_Numbers("numbers_delim.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 5);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 2);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 20;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true);
}

TEST_F(ArrowStorageTest, ImportCsv_KnownSchema_Numbers_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 20;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 5);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 2);
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, true, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_Numbers_Header) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Numbers("numbers_header.csv", parse_options, false);
}

TEST_F(ArrowStorageTest, ImportCsv_UnknownSchema_Numbers_NoHeader) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  Test_ImportCsv_Numbers("numbers_noheader.csv", parse_options, false);
}

TEST_F(ArrowStorageTest, AppendCsvData) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  TableInfoPtr tinfo = storage.createTable(
      "table1", {{"col1", SQLTypeInfo(kINT)}, {"col2", SQLTypeInfo(kFLOAT)}});
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData("1,10.0\n2,20.0\n3,30.0\n", tinfo->table_id, parse_options);
  checkData(
      storage, tinfo->table_id, 3, 32'000'000, range(3, (int32_t)1), range(3, 10.0f));
}

void Test_AppendCsv_Numbers(size_t fragment_size) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  ArrowStorage::CsvParseOptions parse_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  tinfo =
      storage.importCsvFile(getFilePath("numbers_header.csv"),
                            "table1",
                            {{"col1", SQLTypeInfo(kINT)}, {"col2", SQLTypeInfo(kFLOAT)}},
                            table_options,
                            parse_options);
  storage.appendCsvFile(getFilePath("numbers_header2.csv"), "table1");

  checkData(storage,
            tinfo->table_id,
            18,
            table_options.fragment_size,
            range(18, (int32_t)1),
            range(18, 10.0f));
}

TEST_F(ArrowStorageTest, AppendCsv_Numbers) {
  Test_AppendCsv_Numbers(100);
}

TEST_F(ArrowStorageTest, AppendCsv_Numbers_Multifrag) {
  Test_AppendCsv_Numbers(10);
  Test_AppendCsv_Numbers(5);
  Test_AppendCsv_Numbers(2);
  Test_AppendCsv_Numbers(1);
}

void Test_ImportCsv_Strings(bool pass_schema,
                            bool read_twice,
                            const ArrowStorage::CsvParseOptions& parse_options,
                            size_t fragment_size = 32'000'000) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  if (pass_schema) {
    tinfo = storage.importCsvFile(
        getFilePath("strings.csv"),
        "table1",
        {{"col1", SQLTypeInfo(kTEXT)}, {"col2", SQLTypeInfo(kTEXT)}},
        table_options,
        parse_options);
  } else {
    tinfo = storage.importCsvFile(
        getFilePath("strings.csv"), "table1", table_options, parse_options);
  }

  if (read_twice) {
    storage.appendCsvFile(getFilePath("strings.csv"), "table1", parse_options);
  }

  std::vector<std::string> col1_expected = {"s1"s, "ss2"s, "sss3"s, "ssss4"s, "sssss5"s};
  std::vector<std::string> col2_expected = {
      "dd1"s, "dddd2"s, "dddddd3"s, "dddddddd4"s, "dddddddddd5"s};
  if (read_twice) {
    col1_expected = duplicate(col1_expected);
    col2_expected = duplicate(col2_expected);
  }
  checkData(storage,
            tinfo->table_id,
            read_twice ? 10 : 5,
            table_options.fragment_size,
            col1_expected,
            col2_expected);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, false, parse_options, 3);
  Test_ImportCsv_Strings(true, false, parse_options, 2);
  Test_ImportCsv_Strings(true, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, false, parse_options, 3);
  Test_ImportCsv_Strings(true, false, parse_options, 2);
  Test_ImportCsv_Strings(true, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_Strings_NoSchema) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(false, false, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(true, true, parse_options, 7);
  Test_ImportCsv_Strings(true, true, parse_options, 5);
  Test_ImportCsv_Strings(true, true, parse_options, 3);
  Test_ImportCsv_Strings(true, true, parse_options, 2);
  Test_ImportCsv_Strings(true, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Strings(true, true, parse_options, 7);
  Test_ImportCsv_Strings(true, true, parse_options, 5);
  Test_ImportCsv_Strings(true, true, parse_options, 3);
  Test_ImportCsv_Strings(true, true, parse_options, 2);
  Test_ImportCsv_Strings(true, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Strings_NoSchema) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Strings(false, true, parse_options);
}

void Test_ImportCsv_Dict(bool shared_dict,
                         bool read_twice,
                         const ArrowStorage::CsvParseOptions& parse_options,
                         size_t fragment_size = 32'000'000,
                         int index_size = 4) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = fragment_size;
  TableInfoPtr tinfo;
  SQLTypeInfo dict1_type(kTEXT, false, kENCODING_DICT);
  SQLTypeInfo dict2_type(kTEXT, false, kENCODING_DICT);
  if (shared_dict) {
    dict1_type.set_comp_param(-1);
    dict2_type.set_comp_param(-1);
  }
  dict1_type.set_size(index_size);
  dict2_type.set_size(index_size);
  tinfo = storage.importCsvFile(getFilePath("strings.csv"),
                                "table1",
                                {{"col1", dict1_type}, {"col2", dict1_type}},
                                table_options,
                                parse_options);
  if (read_twice) {
    storage.appendCsvFile(getFilePath("strings.csv"), "table1", parse_options);
  }

  if (shared_dict) {
    auto col1_info = storage.getColumnInfo(*tinfo, "col1");
    auto dict1 =
        storage.getDictMetadata(TEST_DB_ID, col1_info->type.get_comp_param())->stringDict;
    CHECK_EQ(dict1->storageEntryCount(), (size_t)10);
  } else {
    auto col1_info = storage.getColumnInfo(*tinfo, "col1");
    auto dict1 =
        storage.getDictMetadata(TEST_DB_ID, col1_info->type.get_comp_param())->stringDict;
    CHECK_EQ(dict1->storageEntryCount(), (size_t)5);
    auto col2_info = storage.getColumnInfo(*tinfo, "col2");
    auto dict2 =
        storage.getDictMetadata(TEST_DB_ID, col2_info->type.get_comp_param())->stringDict;
    CHECK_EQ(dict2->storageEntryCount(), (size_t)5);
  }

  std::vector<std::string> col1_expected = {"s1"s, "ss2"s, "sss3"s, "ssss4"s, "sssss5"s};
  std::vector<std::string> col2_expected = {
      "dd1"s, "dddd2"s, "dddddd3"s, "dddddddd4"s, "dddddddddd5"s};
  if (read_twice) {
    col1_expected = duplicate(col1_expected);
    col2_expected = duplicate(col2_expected);
  }
  checkData(storage,
            tinfo->table_id,
            read_twice ? 10 : 5,
            table_options.fragment_size,
            col1_expected,
            col2_expected);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, false, parse_options, 5);
  Test_ImportCsv_Dict(false, false, parse_options, 3);
  Test_ImportCsv_Dict(false, false, parse_options, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_Dict_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, false, parse_options, 5);
  Test_ImportCsv_Dict(false, false, parse_options, 3);
  Test_ImportCsv_Dict(false, false, parse_options, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_SharedDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(true, false, parse_options);
}

TEST_F(ArrowStorageTest, ImportCsv_SmallDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, false, parse_options, 100, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 3, 2);
  Test_ImportCsv_Dict(false, false, parse_options, 2, 1);
}

TEST_F(ArrowStorageTest, ImportCsv_SharedSmallDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(true, true, parse_options, 100, 2);
  Test_ImportCsv_Dict(true, true, parse_options, 3, 2);
  Test_ImportCsv_Dict(true, true, parse_options, 2, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict_SmallBlock) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(false, true, parse_options, 5);
  Test_ImportCsv_Dict(false, true, parse_options, 3);
  Test_ImportCsv_Dict(false, true, parse_options, 2);
  Test_ImportCsv_Dict(false, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_Dict_SmallBlock_Multifrag) {
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.block_size = 50;
  Test_ImportCsv_Dict(false, true, parse_options, 5);
  Test_ImportCsv_Dict(false, true, parse_options, 3);
  Test_ImportCsv_Dict(false, true, parse_options, 2);
  Test_ImportCsv_Dict(false, true, parse_options, 1);
}

TEST_F(ArrowStorageTest, AppendCsv_SharedDict) {
  ArrowStorage::CsvParseOptions parse_options;
  Test_ImportCsv_Dict(true, true, parse_options);
}

TEST_F(ArrowStorageTest, AppendJsonData) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  TableInfoPtr tinfo = storage.createTable("table1",
                                           {{"col1", SQLTypeInfo(kINT)},
                                            {"col2", SQLTypeInfo(kFLOAT)},
                                            {"col3", SQLTypeInfo(kTEXT)}});
  storage.appendJsonData(R"___({"col1": 1, "col2": 10.0, "col3": "s1"}
{"col1": 2, "col2": 20.0, "col3": "s2"}
{"col1": 3, "col2": 30.0, "col3": "s3"})___",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            range(3, (int32_t)1),
            range(3, 10.0f),
            std::vector<std::string>({"s1"s, "s2"s, "s3"s}));
}

TEST_F(ArrowStorageTest, AppendJsonData_Arrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  SQLTypeInfo int_array(kARRAY, false);
  int_array.set_subtype(kINT);
  SQLTypeInfo float_array(kARRAY, false);
  float_array.set_subtype(kFLOAT);
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", int_array}, {"col2", float_array}});
  storage.appendJsonData(R"___({"col1": [1, 2, 3], "col2": [10.0, 20.0, 30.0]}
{"col1": [4, 5, 6], "col2": [40.0, 50.0, 60.0]}
{"col1": [7, 8, 9], "col2": [70.0, 80.0, 90.0]})___",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            std::vector<std::vector<int>>({std::vector<int>({1, 2, 3}),
                                           std::vector<int>({4, 5, 6}),
                                           std::vector<int>({7, 8, 9})}),
            std::vector<std::vector<float>>({std::vector<float>({10.0f, 20.0f, 30.0f}),
                                             std::vector<float>({40.0f, 50.0f, 60.0f}),
                                             std::vector<float>({70.0f, 80.0f, 90.0f})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_ArraysWithNulls) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  SQLTypeInfo int_array(kARRAY, false);
  int_array.set_subtype(kINT);
  SQLTypeInfo float_array(kARRAY, false);
  float_array.set_subtype(kFLOAT);
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", int_array}, {"col2", float_array}});
  storage.appendJsonData(R"___({"col1": [1, 2, 3], "col2": [null, 20.0, 30.0]}
{"col1": null, "col2": [40.0, null, 60.0]}
{"col1": [7, 8, 9], "col2": null})___",
                         tinfo->table_id);
  checkData(
      storage,
      tinfo->table_id,
      3,
      32'000'000,
      std::vector<std::vector<int>>({std::vector<int>({1, 2, 3}),
                                     std::vector<int>({inline_null_array_value<int>()}),
                                     std::vector<int>({7, 8, 9})}),
      std::vector<std::vector<float>>(
          {std::vector<float>({inline_null_value<float>(), 20.0f, 30.0f}),
           std::vector<float>({40.0f, inline_null_value<float>(), 60.0f}),
           std::vector<float>({inline_null_array_value<float>()})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_FixedSizeArrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  SQLTypeInfo int3_array(kARRAY, false);
  int3_array.set_subtype(kINT);
  int3_array.set_size(3 * sizeof(int));
  SQLTypeInfo float2_array(kARRAY, false);
  float2_array.set_subtype(kFLOAT);
  float2_array.set_size(2 * sizeof(float));
  TableInfoPtr tinfo =
      storage.createTable("table1", {{"col1", int3_array}, {"col2", float2_array}});
  storage.appendJsonData(R"___({"col1": [1, 2], "col2": [null, 20.0]}
{"col1": null, "col2": [40.0, null, 60.0, 70.0]}
{"col1": [7, 8, 9, 10], "col2": null}
{"col1": [11, 12, 13], "col2": [110.0]})___",
                         tinfo->table_id);
  checkData(
      storage,
      tinfo->table_id,
      4,
      32'000'000,
      std::vector<std::vector<int>>({std::vector<int>({1, 2, inline_null_value<int>()}),
                                     std::vector<int>({inline_null_array_value<int>(),
                                                       inline_null_value<int>(),
                                                       inline_null_value<int>()}),
                                     std::vector<int>({7, 8, 9}),
                                     std::vector<int>({11, 12, 13})}),
      std::vector<std::vector<float>>(
          {std::vector<float>({inline_null_value<float>(), 20.0f}),
           std::vector<float>({40.0f, inline_null_value<float>()}),
           std::vector<float>(
               {inline_null_array_value<float>(), inline_null_value<float>()}),
           std::vector<float>({110.f, inline_null_value<float>()})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_StringFixedSizeArrays) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  SQLTypeInfo string3_array_type(kARRAY, kENCODING_DICT, 0, kTEXT);
  string3_array_type.set_size(3 * sizeof(int32_t));
  TableInfoPtr tinfo = storage.createTable("table1", {{"col1", string3_array_type}});
  storage.appendJsonData(R"___({"col1": ["str1", "str2"]}
{"col1": null}
{"col1": ["str2", "str8", "str3", "str10"]}
{"col1": ["str1", null, "str3"]})___",
                         tinfo->table_id);
  checkData(storage,
            tinfo->table_id,
            4,
            32'000'000,
            std::vector<std::vector<std::string>>(
                {std::vector<std::string>({"str1"s, "str2"s, "<NULL>"}),
                 std::vector<std::string>({}),
                 std::vector<std::string>({"str2"s, "str8"s, "str3"s}),
                 std::vector<std::string>({"str1"s, "<NULL>", "str3"s})}));
}

TEST_F(ArrowStorageTest, AppendJsonData_DateTime) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  TableInfoPtr tinfo = storage.createTable(
      "table1",
      {{"ts_0", SQLTypeInfo(kTIMESTAMP, 0, 0)},
       {"ts_3", SQLTypeInfo(kTIMESTAMP, 3, 0)},
       {"ts_6", SQLTypeInfo(kTIMESTAMP, 6, 0)},
       {"ts_9", SQLTypeInfo(kTIMESTAMP, 9, 0)},
       {"t", SQLTypeInfo(kTIME)},
       {"d", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)},
       {"o1", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 16, kNULLT)},
       {"o2", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)}});
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData(
      "2014-12-13 22:23:15,2014-12-13 22:23:15.323,1999-07-11 14:02:53.874533,2006-04-26 "
      "03:49:04.607435125,15:13:14,1999-09-09,1999-09-09,1999-09-09",
      tinfo->table_id,
      parse_options);
  storage.appendCsvData(
      "2014-12-13 22:23:15,2014-12-13 22:23:15.323,2014-12-13 22:23:15.874533,2014-12-13 "
      "22:23:15.607435763,15:13:14,,,",
      tinfo->table_id,
      parse_options);
  storage.appendCsvData(
      "2014-12-14 22:23:15,2014-12-14 22:23:15.750,2014-12-14 22:23:15.437321,2014-12-14 "
      "22:23:15.934567401,15:13:14,1999-09-09,1999-09-09,1999-09-09",
      tinfo->table_id,
      parse_options);
  checkData(storage,
            tinfo->table_id,
            3,
            32'000'000,
            std::vector<int64_t>({1418509395, 1418509395, 1418595795}),
            std::vector<int64_t>({1418509395323, 1418509395323, 1418595795750}),
            std::vector<int64_t>({931701773874533, 1418509395874533, 1418595795437321}),
            std::vector<int64_t>(
                {1146023344607435125, 1418509395607435763, 1418595795934567401}),
            std::vector<int64_t>({54794, 54794, 54794}),
            std::vector<int32_t>({10843, inline_null_value<int32_t>(), 10843}),
            std::vector<int16_t>({10843, inline_null_value<int16_t>(), 10843}),
            std::vector<int32_t>({10843, inline_null_value<int32_t>(), 10843}));
}

TEST_F(ArrowStorageTest, AppendJsonData_DateTime_Multifrag) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 2;
  TableInfoPtr tinfo = storage.createTable(
      "table1",
      {{"ts_0", SQLTypeInfo(kTIMESTAMP, 0, 0)},
       {"ts_3", SQLTypeInfo(kTIMESTAMP, 3, 0)},
       {"ts_6", SQLTypeInfo(kTIMESTAMP, 6, 0)},
       {"ts_9", SQLTypeInfo(kTIMESTAMP, 9, 0)},
       {"t", SQLTypeInfo(kTIME)},
       {"d", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 0, kNULLT)},
       {"o1", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 16, kNULLT)},
       {"o2", SQLTypeInfo(kDATE, kENCODING_DATE_IN_DAYS, 32, kNULLT)}},
      table_options);
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  for (int i = 0; i < 4; ++i) {
    storage.appendCsvData(
        "2014-12-13 22:23:15,2014-12-13 22:23:15.323,1999-07-11 "
        "14:02:53.874533,2006-04-26 "
        "03:49:04.607435125,15:13:14,1999-09-09,1999-09-09,1999-09-09",
        tinfo->table_id,
        parse_options);
  }
  for (int i = 0; i < 3; ++i) {
    storage.appendCsvData(
        "2014-12-13 22:23:15,2014-12-13 22:23:15.323,2014-12-13 "
        "22:23:15.874533,2014-12-13 "
        "22:23:15.607435763,15:13:14,,,",
        tinfo->table_id,
        parse_options);
  }
  for (int i = 0; i < 3; ++i) {
    storage.appendCsvData(
        "2014-12-14 22:23:15,2014-12-14 22:23:15.750,2014-12-14 "
        "22:23:15.437321,2014-12-14 "
        "22:23:15.934567401,15:13:14,1999-09-09,1999-09-09,1999-09-09",
        tinfo->table_id,
        parse_options);
  }
  checkData(storage,
            tinfo->table_id,
            10,
            2,
            std::vector<int64_t>({1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418509395,
                                  1418595795,
                                  1418595795,
                                  1418595795}),
            std::vector<int64_t>({1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418509395323,
                                  1418595795750,
                                  1418595795750,
                                  1418595795750}),
            std::vector<int64_t>({931701773874533,
                                  931701773874533,
                                  931701773874533,
                                  931701773874533,
                                  1418509395874533,
                                  1418509395874533,
                                  1418509395874533,
                                  1418595795437321,
                                  1418595795437321,
                                  1418595795437321}),
            std::vector<int64_t>({1146023344607435125,
                                  1146023344607435125,
                                  1146023344607435125,
                                  1146023344607435125,
                                  1418509395607435763,
                                  1418509395607435763,
                                  1418509395607435763,
                                  1418595795934567401,
                                  1418595795934567401,
                                  1418595795934567401}),
            std::vector<int64_t>(
                {54794, 54794, 54794, 54794, 54794, 54794, 54794, 54794, 54794, 54794}),
            std::vector<int32_t>({10843,
                                  10843,
                                  10843,
                                  10843,
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  10843,
                                  10843,
                                  10843}),
            std::vector<int16_t>({10843,
                                  10843,
                                  10843,
                                  10843,
                                  inline_null_value<int16_t>(),
                                  inline_null_value<int16_t>(),
                                  inline_null_value<int16_t>(),
                                  10843,
                                  10843,
                                  10843}),
            std::vector<int32_t>({10843,
                                  10843,
                                  10843,
                                  10843,
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  inline_null_value<int32_t>(),
                                  10843,
                                  10843,
                                  10843}));
}

TEST_F(ArrowStorageTest, AppendCsvData_BoolWithNulls) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 2;
  TableInfoPtr tinfo =
      storage.createTable("table1",
                          {{"b1", SQLTypeInfo(kBOOLEAN)}, {"b2", SQLTypeInfo(kBOOLEAN)}},
                          table_options);
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData("true,true", tinfo->table_id, parse_options);
  storage.appendCsvData("true,false", tinfo->table_id, parse_options);
  storage.appendCsvData("false,true", tinfo->table_id, parse_options);
  storage.appendCsvData(",true", tinfo->table_id, parse_options);
  storage.appendCsvData(",false", tinfo->table_id, parse_options);
  storage.appendCsvData("true,", tinfo->table_id, parse_options);
  storage.appendCsvData("false,", tinfo->table_id, parse_options);
  storage.appendCsvData(",", tinfo->table_id, parse_options);
  checkData(storage,
            tinfo->table_id,
            8,
            2,
            std::vector<int8_t>({1, 1, 0, -128, -128, 1, 0, -128}),
            std::vector<int8_t>({1, 0, 1, 1, 0, -128, -128, -128}));
}

TEST_F(ArrowStorageTest, AppendCsvData_BoolWithNulls_SingleChunk) {
  ArrowStorage storage(TEST_SCHEMA_ID, "test", TEST_DB_ID);
  ArrowStorage::TableOptions table_options;
  table_options.fragment_size = 2;
  TableInfoPtr tinfo =
      storage.createTable("table1",
                          {{"b1", SQLTypeInfo(kBOOLEAN)}, {"b2", SQLTypeInfo(kBOOLEAN)}},
                          table_options);
  ArrowStorage::CsvParseOptions parse_options;
  parse_options.header = false;
  storage.appendCsvData(
      "true,true\n"
      "true,false\n"
      "false,true\n"
      ",true\n"
      ",false\n"
      "true,\n"
      "false,\n"
      ",",
      tinfo->table_id,
      parse_options);
  checkData(storage,
            tinfo->table_id,
            8,
            2,
            std::vector<int8_t>({1, 1, 0, -128, -128, 1, 0, -128}),
            std::vector<int8_t>({1, 0, 1, 1, 0, -128, -128, -128}));
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
