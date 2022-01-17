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

#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/RelAlgExecutor.h"

#include "TestHelpers.h"

#include <gtest/gtest.h>

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;
constexpr int TEST1_TABLE_ID = 1;
constexpr int TEST2_TABLE_ID = 2;

namespace {

template <typename T>
void set_datum(Datum& d, const T& v) {
  UNREACHABLE();
}

template <>
void set_datum<int32_t>(Datum& d, const int32_t& v) {
  d.intval = v;
}

template <>
void set_datum<int64_t>(Datum& d, const int64_t& v) {
  d.bigintval = v;
}

template <>
void set_datum<float>(Datum& d, const float& v) {
  d.floatval = v;
}

template <>
void set_datum<double>(Datum& d, const double& v) {
  d.doubleval = v;
}

template <typename... Ts>
void compare_res_data(const ExecutionResult& res, const std::vector<Ts>&... expected) {
  std::vector<std::string> col_names;
  for (auto& target : res.getTargetsMeta()) {
    col_names.push_back(target.get_resname());
  }
  auto converter =
      std::make_unique<ArrowResultSetConverter>(res.getDataPtr(), col_names, -1);
  auto at = converter->convertToArrowTable();

  TestHelpers::compare_arrow_table(at, expected...);
}

}  // anonymous namespace

class TestSchemaProvider : public SchemaProvider {
 public:
  TestSchemaProvider() {
    id_ = TEST_SCHEMA_ID;
    db_id_ = TEST_DB_ID;

    auto t1_info = std::make_shared<TableInfo>(db_id_,
                                               TEST1_TABLE_ID,
                                               "test1",
                                               false,
                                               -1,
                                               Data_Namespace::MemoryLevel::CPU_LEVEL,
                                               1);
    auto t2_info = std::make_shared<TableInfo>(db_id_,
                                               TEST2_TABLE_ID,
                                               "test2",
                                               false,
                                               -1,
                                               Data_Namespace::MemoryLevel::CPU_LEVEL,
                                               1);

    table_infos_[*t1_info] = t1_info;
    table_index_by_name_[t1_info->db_id][t1_info->name] = t1_info;
    table_infos_[*t2_info] = t2_info;
    table_index_by_name_[t2_info->db_id][t2_info->name] = t2_info;

    auto t1_col1_info = std::make_shared<ColumnInfo>(db_id_,
                                                     TEST1_TABLE_ID,
                                                     1,
                                                     "col_bi",
                                                     SQLTypeInfo(SQLTypes::kBIGINT),
                                                     false,
                                                     false);
    auto t1_col2_info = std::make_shared<ColumnInfo>(
        db_id_, TEST1_TABLE_ID, 2, "col_i", SQLTypeInfo(SQLTypes::kINT), false, false);
    auto t1_col3_info = std::make_shared<ColumnInfo>(
        db_id_, TEST1_TABLE_ID, 3, "col_f", SQLTypeInfo(SQLTypes::kFLOAT), false, false);
    auto t1_col4_info = std::make_shared<ColumnInfo>(
        db_id_, TEST1_TABLE_ID, 4, "col_d", SQLTypeInfo(SQLTypes::kDOUBLE), false, false);

    auto t2_col1_info = std::make_shared<ColumnInfo>(db_id_,
                                                     TEST2_TABLE_ID,
                                                     1,
                                                     "col_bi",
                                                     SQLTypeInfo(SQLTypes::kBIGINT),
                                                     false,
                                                     false);
    auto t2_col2_info = std::make_shared<ColumnInfo>(
        db_id_, TEST2_TABLE_ID, 2, "col_i", SQLTypeInfo(SQLTypes::kINT), false, false);
    auto t2_col3_info = std::make_shared<ColumnInfo>(
        db_id_, TEST2_TABLE_ID, 3, "col_f", SQLTypeInfo(SQLTypes::kFLOAT), false, false);
    auto t2_col4_info = std::make_shared<ColumnInfo>(
        db_id_, TEST2_TABLE_ID, 4, "col_d", SQLTypeInfo(SQLTypes::kDOUBLE), false, false);

    column_infos_[*t1_col1_info] = t1_col1_info;
    column_index_by_name_[{t1_col1_info->db_id, t1_col1_info->table_id}]
                         [t1_col1_info->name] = t1_col1_info;
    column_infos_[*t1_col2_info] = t1_col2_info;
    column_index_by_name_[{t1_col2_info->db_id, t1_col2_info->table_id}]
                         [t1_col2_info->name] = t1_col2_info;
    column_infos_[*t1_col3_info] = t1_col3_info;
    column_index_by_name_[{t1_col3_info->db_id, t1_col3_info->table_id}]
                         [t1_col3_info->name] = t1_col3_info;
    column_infos_[*t1_col4_info] = t1_col4_info;
    column_index_by_name_[{t1_col4_info->db_id, t1_col4_info->table_id}]
                         [t1_col4_info->name] = t1_col4_info;

    column_infos_[*t2_col1_info] = t2_col1_info;
    column_index_by_name_[{t2_col1_info->db_id, t2_col1_info->table_id}]
                         [t2_col1_info->name] = t2_col1_info;
    column_infos_[*t2_col2_info] = t2_col2_info;
    column_index_by_name_[{t2_col2_info->db_id, t2_col2_info->table_id}]
                         [t2_col2_info->name] = t2_col2_info;
    column_infos_[*t2_col3_info] = t2_col3_info;
    column_index_by_name_[{t2_col3_info->db_id, t2_col3_info->table_id}]
                         [t2_col3_info->name] = t2_col3_info;
    column_infos_[*t2_col4_info] = t2_col4_info;
    column_index_by_name_[{t2_col4_info->db_id, t2_col4_info->table_id}]
                         [t2_col4_info->name] = t2_col4_info;
  }

  ~TestSchemaProvider() override = default;

  int getId() const override { return id_; }
  std::string_view getName() const override { return "test"; }

  std::vector<int> listDatabases() const override { UNREACHABLE(); }

  TableInfoList listTables(int db_id) const override { UNREACHABLE(); }

  ColumnInfoList listColumns(int db_id, int table_id) const override {
    CHECK_EQ(column_index_by_name_.count({db_id, table_id}), 1);
    auto& table_cols = column_index_by_name_.at({db_id, table_id});
    ColumnInfoList res;
    res.reserve(table_cols.size());
    for (auto [col_name, col_info] : table_cols) {
      res.push_back(col_info);
    }
    return res;
  }

  TableInfoPtr getTableInfo(int db_id, int table_id) const override {
    auto it = table_infos_.find({db_id, table_id});
    if (it != table_infos_.end()) {
      return it->second;
    }
    return nullptr;
  }

  TableInfoPtr getTableInfo(int db_id, const std::string& table_name) const override {
    auto db_it = table_index_by_name_.find(db_id);
    if (db_it != table_index_by_name_.end()) {
      auto table_it = db_it->second.find(table_name);
      if (table_it != db_it->second.end()) {
        return table_it->second;
      }
    }
    return nullptr;
  }

  ColumnInfoPtr getColumnInfo(int db_id, int table_id, int col_id) const override {
    auto it = column_infos_.find({db_id, table_id, col_id});
    if (it != column_infos_.end()) {
      return it->second;
    }
    return nullptr;
  }

  ColumnInfoPtr getColumnInfo(int db_id,
                              int table_id,
                              const std::string& col_name) const override {
    auto table_it = column_index_by_name_.find({db_id, table_id});
    if (table_it != column_index_by_name_.end()) {
      auto col_it = table_it->second.find(col_name);
      if (col_it != table_it->second.end()) {
        return col_it->second;
      }
    }
    return nullptr;
  }

 private:
  using TableByNameMap = std::unordered_map<std::string, TableInfoPtr>;
  using ColumnByNameMap = std::unordered_map<std::string, ColumnInfoPtr>;

  int id_;
  int db_id_;
  TableInfoMap table_infos_;
  std::unordered_map<int, TableByNameMap> table_index_by_name_;
  ColumnInfoMap column_infos_;
  std::unordered_map<TableRef, ColumnByNameMap> column_index_by_name_;
};

class TableTableData {
 public:
  TableTableData(int db_id, int table_id, int cols, SchemaProviderPtr schema_provider_)
      : ref_(db_id, table_id) {
    info_.chunkKeyPrefix.push_back(db_id);
    info_.chunkKeyPrefix.push_back(table_id);
    data_.resize(cols);

    auto col_infos = schema_provider_->listColumns(ref_);
    for (auto& col_info : col_infos) {
      col_types_[col_info->column_id] = col_info->type;
    }
  }

  template <typename T>
  void addColFragment(size_t col_id, std::vector<T> vals) {
    CHECK_LE(col_id, data_.size());
    CHECK_EQ(col_types_.count(col_id), 1);
    auto& frag_data = data_[col_id - 1].emplace_back();
    frag_data.resize(vals.size() * sizeof(T));
    memcpy(frag_data.data(), vals.data(), frag_data.size());

    if (info_.fragments.size() < data_[col_id - 1].size()) {
      Fragmenter_Namespace::FragmentInfo& frag_info = info_.fragments.emplace_back();
      frag_info.fragmentId = info_.fragments.size();
      frag_info.physicalTableId = info_.chunkKeyPrefix[CHUNK_KEY_TABLE_IDX];
      frag_info.setPhysicalNumTuples(vals.size());
      frag_info.deviceIds.push_back(0);  // Data_Namespace::DISK_LEVEL
      frag_info.deviceIds.push_back(0);  // Data_Namespace::CPU_LEVEL
      frag_info.deviceIds.push_back(0);  // Data_Namespace::GPU_LEVEL
    }

    auto chunk_meta = std::make_shared<ChunkMetadata>();
    chunk_meta->sqlType = col_types_.at(col_id);
    chunk_meta->numBytes = frag_data.size();
    chunk_meta->numElements = vals.size();

    // No computed meta.
    chunk_meta->chunkStats.has_nulls = true;
    set_datum(chunk_meta->chunkStats.min, std::numeric_limits<T>::min());
    set_datum(chunk_meta->chunkStats.max, std::numeric_limits<T>::max());

    auto& frag_info = info_.fragments[data_[col_id - 1].size() - 1];
    frag_info.setChunkMetadata(col_id, chunk_meta);
  }

  void fetchData(int col_id, int frag_id, int8_t* dst, size_t size) {
    CHECK_LE(col_id, data_.size());
    CHECK_LE(frag_id, data_[col_id - 1].size());
    auto& chunk = data_[col_id - 1][frag_id - 1];
    CHECK_LE(chunk.size(), size);
    memcpy(dst, data_[col_id - 1][frag_id - 1].data(), size);
  }

  const Fragmenter_Namespace::TableInfo& getTableInfo() const { return info_; }

 private:
  TableRef ref_;
  std::vector<std::vector<std::vector<int8_t>>> data_;
  Fragmenter_Namespace::TableInfo info_;
  std::unordered_map<int, SQLTypeInfo> col_types_;
};

class TestDataProvider : public AbstractBufferMgr {
 public:
  TestDataProvider(SchemaProviderPtr schema_provider)
      : AbstractBufferMgr(0), schema_provider_(schema_provider) {
    TableTableData test1(TEST_DB_ID, TEST1_TABLE_ID, 4, schema_provider_);
    test1.addColFragment<int64_t>(1, {1, 2, 3, 4, 5});
    test1.addColFragment<int32_t>(2, {10, 20, 30, 40, 50});
    test1.addColFragment<float>(3, {1.1, 2.2, 3.3, 4.4, 5.5});
    test1.addColFragment<double>(4, {10.1, 20.2, 30.3, 40.4, 50.5});
    tables_.emplace(std::make_pair(TEST1_TABLE_ID, test1));

    TableTableData test2(TEST_DB_ID, TEST2_TABLE_ID, 4, schema_provider_);
    test2.addColFragment<int64_t>(1, {1, 2, 3});
    test2.addColFragment<int64_t>(1, {4, 5, 6});
    test2.addColFragment<int64_t>(1, {7, 8, 9});
    test2.addColFragment<int32_t>(2, {110, 120, 130});
    test2.addColFragment<int32_t>(2, {140, 150, 160});
    test2.addColFragment<int32_t>(2, {170, 180, 190});
    test2.addColFragment<float>(3, {101.1, 102.2, 103.3});
    test2.addColFragment<float>(3, {104.4, 105.5, 106.6});
    test2.addColFragment<float>(3, {107.7, 108.8, 109.9});
    test2.addColFragment<double>(4, {110.1, 120.2, 130.3});
    test2.addColFragment<double>(4, {140.4, 150.5, 160.6});
    test2.addColFragment<double>(4, {170.7, 180.8, 190.9});
    tables_.emplace(std::make_pair(TEST2_TABLE_ID, test2));
  }

  // Chunk API
  AbstractBuffer* createBuffer(const ChunkKey& key,
                               const size_t pageSize = 0,
                               const size_t initialSize = 0) override {
    UNREACHABLE();
  }

  void deleteBuffer(const ChunkKey& key, const bool purge = true) override {
    UNREACHABLE();
  }  // purge param only used in FileMgr

  void deleteBuffersWithPrefix(const ChunkKey& keyPrefix,
                               const bool purge = true) override {
    UNREACHABLE();
  }

  AbstractBuffer* getBuffer(const ChunkKey& key, const size_t numBytes = 0) override {
    UNREACHABLE();
  }

  void fetchBuffer(const ChunkKey& key,
                   AbstractBuffer* destBuffer,
                   const size_t numBytes = 0) override {
    CHECK_EQ(key[CHUNK_KEY_DB_IDX], TEST_DB_ID);
    CHECK_EQ(tables_.count(key[CHUNK_KEY_TABLE_IDX]), 1);
    auto& data = tables_.at(key[CHUNK_KEY_TABLE_IDX]);
    data.fetchData(key[CHUNK_KEY_COLUMN_IDX],
                   key[CHUNK_KEY_FRAGMENT_IDX],
                   destBuffer->getMemoryPtr(),
                   numBytes);
  }

  AbstractBuffer* putBuffer(const ChunkKey& key,
                            AbstractBuffer* srcBuffer,
                            const size_t numBytes = 0) override {
    UNREACHABLE();
  }

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override {
    UNREACHABLE();
  }

  bool isBufferOnDevice(const ChunkKey& key) override { UNREACHABLE(); }
  std::string printSlabs() override { UNREACHABLE(); }
  size_t getMaxSize() override { UNREACHABLE(); }
  size_t getInUseSize() override { UNREACHABLE(); }
  size_t getAllocated() override { UNREACHABLE(); }
  bool isAllocationCapped() override { UNREACHABLE(); }

  void checkpoint() override { UNREACHABLE(); }
  void checkpoint(const int db_id, const int tb_id) override { UNREACHABLE(); }
  void removeTableRelatedDS(const int db_id, const int table_id) override {
    UNREACHABLE();
  }

  const DictDescriptor* getDictMetadata(int db_id,
                                        int dict_id,
                                        bool load_dict = true) override {
    UNREACHABLE();
  }

  Fragmenter_Namespace::TableInfo getTableInfo(int db_id, int table_id) const override {
    CHECK_EQ(db_id, TEST_DB_ID);
    CHECK_EQ(tables_.count(table_id), 1);
    std::cout << "TestDataProvider::getTableInfo return table info with "
              << tables_.at(table_id).getTableInfo().fragments.size() << " fragments"
              << std::endl;
    return tables_.at(table_id).getTableInfo();
  }

  // Buffer API
  AbstractBuffer* alloc(const size_t numBytes = 0) override { UNREACHABLE(); }
  void free(AbstractBuffer* buffer) override { UNREACHABLE(); }
  MgrType getMgrType() override { UNREACHABLE(); }
  std::string getStringMgrType() override { UNREACHABLE(); }
  size_t getNumChunks() override { UNREACHABLE(); }

 private:
  SchemaProviderPtr schema_provider_;
  std::unordered_map<int, TableTableData> tables_;
};

class NoCatalogRelAlgTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    schema_provider_ = std::make_shared<TestSchemaProvider>();

    SystemParameters system_parameters;
    data_mgr_ = std::make_shared<DataMgr>("", system_parameters, nullptr, false);
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID,
                                 std::make_unique<TestDataProvider>(schema_provider_));

    executor_ = std::make_shared<Executor>(0,
                                           data_mgr_.get(),
                                           system_parameters.cuda_block_size,
                                           system_parameters.cuda_grid_size,
                                           system_parameters.max_gpu_slab_size,
                                           "",
                                           "");
  }

  static void TearDownTestSuite() {}

  ExecutionResult runRelAlgQuery(const std::string& ra) {
    auto dag =
        std::make_unique<RelAlgDagBuilder>(ra, TEST_DB_ID, schema_provider_, nullptr);
    auto ra_executor =
        RelAlgExecutor(executor_.get(), TEST_DB_ID, schema_provider_, std::move(dag));
    return ra_executor.executeRelAlgQuery(
        CompilationOptions(), ExecutionOptions(), false, nullptr);
  }

 private:
  static std::shared_ptr<DataMgr> data_mgr_;
  static SchemaProviderPtr schema_provider_;
  static std::shared_ptr<Executor> executor_;
};

std::shared_ptr<DataMgr> NoCatalogRelAlgTest::data_mgr_;
SchemaProviderPtr NoCatalogRelAlgTest::schema_provider_;
std::shared_ptr<Executor> NoCatalogRelAlgTest::executor_;

TEST_F(NoCatalogRelAlgTest, SelectSingleColumn) {
  auto ra = R"""(
{
  "rels": [
    {
      "id": "0",
      "relOp": "EnumerableTableScan",
      "table": [
        "omnisci",
        "test1"
      ],
      "fieldNames": [
        "col_i"
      ],
      "inputs": []
    },
    {
      "id": "1",
      "relOp": "LogicalProject",
      "fields": [
        "coli"
      ],
      "exprs": [
        {
          "input": 0
        }
      ]
    }
  ]
})""";
  auto res = runRelAlgQuery(ra);
  compare_res_data(res, std::vector<int>({10, 20, 30, 40, 50}));
}

TEST_F(NoCatalogRelAlgTest, SelectAllColumns) {
  auto ra = R"""(
{
  "rels": [
    {
      "id": "0",
      "relOp": "EnumerableTableScan",
      "table": [
        "omnisci",
        "test1"
      ],
      "fieldNames": [
        "col_bi",
        "col_i",
        "col_f",
        "col_d"
      ],
      "inputs": []
    },
    {
      "id": "1",
      "relOp": "LogicalProject",
      "fields": [
        "col_bi",
        "col_i",
        "col_f",
        "col_d"
      ],
      "exprs": [
        {
          "input": 0
        },
        {
          "input": 1
        },
        {
          "input": 2
        },
        {
          "input": 3
        }
      ]
    }
  ]
})""";
  auto res = runRelAlgQuery(ra);
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3, 4, 5}),
                   std::vector<int>({10, 20, 30, 40, 50}),
                   std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5}),
                   std::vector<double>({10.1, 20.2, 30.3, 40.4, 50.5}));
}

TEST_F(NoCatalogRelAlgTest, SelectAllColumnsMultiFrag) {
  auto ra = R"""(
{
  "rels": [
    {
      "id": "0",
      "relOp": "EnumerableTableScan",
      "table": [
        "omnisci",
        "test2"
      ],
      "fieldNames": [
        "col_bi",
        "col_i",
        "col_f",
        "col_d"
      ],
      "inputs": []
    },
    {
      "id": "1",
      "relOp": "LogicalProject",
      "fields": [
        "col_bi",
        "col_i",
        "col_f",
        "col_d"
      ],
      "exprs": [
        {
          "input": 0
        },
        {
          "input": 1
        },
        {
          "input": 2
        },
        {
          "input": 3
        }
      ]
    }
  ]
})""";
  auto res = runRelAlgQuery(ra);
  compare_res_data(
      res,
      std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}),
      std::vector<int>({110, 120, 130, 140, 150, 160, 170, 180, 190}),
      std::vector<float>({101.1, 102.2, 103.3, 104.4, 105.5, 106.6, 107.7, 108.8, 109.9}),
      std::vector<double>(
          {110.1, 120.2, 130.3, 140.4, 150.5, 160.6, 170.7, 180.8, 190.9}));
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