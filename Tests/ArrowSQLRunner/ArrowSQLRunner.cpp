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

#include "ArrowSQLRunner.h"

#include "Calcite/CalciteJNI.h"
#include "DataMgr/DataMgr.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/RelAlgExecutor.h"

#include "SQLiteComparator.h"
#include "SchemaJson.h"

#include <gtest/gtest.h>

extern bool g_enable_columnar_output;
extern double g_gpu_mem_limit_percent;

namespace TestHelpers::ArrowSQLRunner {

bool g_hoist_literals = true;

namespace {

class ArrowSQLRunnerImpl {
 public:
  static void init(size_t max_gpu_mem, const std::string& udf_filename) {
    instance_.reset(new ArrowSQLRunnerImpl(max_gpu_mem, udf_filename));
  }

  static void reset() { instance_.reset(); }

  static ArrowSQLRunnerImpl* get() {
    CHECK(instance_) << "ArrowSQLRunner is not initialized";
    return instance_.get();
  }

  bool gpusPresent() { return data_mgr_->gpusPresent(); }

  void printStats() {
    std::cout << "Total schema to JSON time: " << (schema_to_json_time_ / 1000) << "ms."
              << std::endl;
    std::cout << "Total Calcite parsing time: " << (calcite_time_ / 1000) << "ms."
              << std::endl;
    std::cout << "Total execution time: " << (execution_time_ / 1000) << "ms."
              << std::endl;
  }

  void createTable(
      const std::string& table_name,
      const std::vector<ArrowStorage::ColumnDescription>& columns,
      const ArrowStorage::TableOptions& options = ArrowStorage::TableOptions()) {
    storage_->createTable(table_name, columns, options);
  }

  void dropTable(const std::string& table_name) { storage_->dropTable(table_name); }

  void insertCsvValues(const std::string& table_name, const std::string& values) {
    ArrowStorage::CsvParseOptions parse_options;
    parse_options.header = false;
    storage_->appendCsvData(values, table_name, parse_options);
  }

  void insertJsonValues(const std::string& table_name, const std::string& values) {
    storage_->appendJsonData(values, table_name);
  }

  std::string getSqlQueryRelAlg(const std::string& sql) {
    std::string schema_json;
    std::string query_ra;

    schema_to_json_time_ += measure<std::chrono::microseconds>::execution(
        [&]() { schema_json = schema_to_json(storage_); });

    calcite_time_ += measure<std::chrono::microseconds>::execution([&]() {
      query_ra =
          calcite_->process("admin", "test_db", pg_shim(sql), schema_json, {}, true);
    });

    return query_ra;
  }

  std::unique_ptr<RelAlgExecutor> makeRelAlgExecutor(const std::string& sql) {
    std::string query_ra = getSqlQueryRelAlg(sql);

    auto dag = std::make_unique<RelAlgDagBuilder>(query_ra, TEST_DB_ID, storage_);

    return std::make_unique<RelAlgExecutor>(executor_.get(),
                                            TEST_DB_ID,
                                            storage_,
                                            data_mgr_->getDataProvider(),
                                            std::move(dag));
  }

  ExecutionResult runSqlQuery(const std::string& sql,
                              const CompilationOptions& co,
                              const ExecutionOptions& eo) {
    auto ra_executor = makeRelAlgExecutor(sql);
    ExecutionResult res;

    execution_time_ += measure<std::chrono::microseconds>::execution(
        [&]() { res = ra_executor->executeRelAlgQuery(co, eo, false); });

    return res;
  }

  RegisteredQueryHint getParsedQueryHint(const std::string& query_str) {
    auto ra_executor = makeRelAlgExecutor(query_str);
    auto query_hints =
        ra_executor->getParsedQueryHint(ra_executor->getRootRelAlgNodeShPtr().get());
    return query_hints ? *query_hints : RegisteredQueryHint::defaults();
  }

  std::optional<std::unordered_map<size_t, RegisteredQueryHint>> getParsedQueryHints(
      const std::string& query_str) {
    auto ra_executor = makeRelAlgExecutor(query_str);
    auto query_hints = ra_executor->getParsedQueryHints();
    return query_hints ? query_hints : std::nullopt;
  }

  ExecutionResult runSqlQuery(const std::string& sql,
                              ExecutorDeviceType device_type,
                              const ExecutionOptions& eo) {
    return runSqlQuery(sql, getCompilationOptions(device_type), eo);
  }

  ExecutionResult runSqlQuery(const std::string& sql,
                              ExecutorDeviceType device_type,
                              bool allow_loop_joins) {
    return runSqlQuery(sql, device_type, getExecutionOptions(allow_loop_joins));
  }

  ExecutionOptions getExecutionOptions(bool allow_loop_joins, bool just_explain = false) {
    return {g_enable_columnar_output,
            true,
            just_explain,
            allow_loop_joins,
            false,
            false,
            false,
            false,
            10000,
            false,
            false,
            g_gpu_mem_limit_percent,
            false,
            1000};
  }

  CompilationOptions getCompilationOptions(ExecutorDeviceType device_type) {
    auto co = CompilationOptions::defaults(device_type);
    co.hoist_literals = g_hoist_literals;
    return co;
  }

  std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                              const ExecutorDeviceType device_type,
                                              const bool allow_loop_joins = true) {
    return runSqlQuery(query_str, device_type, allow_loop_joins).getRows();
  }

  TargetValue run_simple_agg(const std::string& query_str,
                             const ExecutorDeviceType device_type,
                             const bool allow_loop_joins = true) {
    auto rows = run_multiple_agg(query_str, device_type, allow_loop_joins);
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size()) << query_str;
    return crt_row[0];
  }

  void run_sqlite_query(const std::string& query_string) {
    sqlite_comparator_.query(query_string);
  }

  void sqlite_batch_insert(const std::string& table_name,
                           std::vector<std::vector<std::string>>& insert_vals) {
    sqlite_comparator_.batch_insert(table_name, insert_vals);
  }

  void c(const std::string& query_string, const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare(
        run_multiple_agg(query_string, device_type), query_string, device_type);
  }

  void c(const std::string& query_string,
         const std::string& sqlite_query_string,
         const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare(
        run_multiple_agg(query_string, device_type), sqlite_query_string, device_type);
  }

  /* timestamp approximate checking for NOW() */
  void cta(const std::string& query_string, const ExecutorDeviceType device_type) {
    sqlite_comparator_.compare_timstamp_approx(
        run_multiple_agg(query_string, device_type), query_string, device_type);
  }

  void check_arrow_dictionaries(
      const ArrowResultSet* arrow_result_set,
      const ResultSetPtr omnisci_results,
      const size_t min_result_size_for_bulk_dictionary_fetch,
      const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {
    const size_t num_columns = arrow_result_set->colCount();
    std::unordered_set<size_t> dictionary_encoded_col_idxs;
    std::vector<std::unordered_set<std::string>> per_column_dictionary_sets(num_columns);
    for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
      const auto column_typeinfo = arrow_result_set->getColType(col_idx);
      if (column_typeinfo.get_type() != kTEXT) {
        continue;
      }
      dictionary_encoded_col_idxs.emplace(col_idx);
      ASSERT_EQ(column_typeinfo.get_compression(), kENCODING_DICT);

      const auto dictionary_strings = arrow_result_set->getDictionaryStrings(col_idx);
      auto& dictionary_set = per_column_dictionary_sets[col_idx];
      for (const auto& dictionary_string : dictionary_strings) {
        ASSERT_EQ(dictionary_set.emplace(dictionary_string).second, true);
      }
    }
    const size_t row_count = arrow_result_set->rowCount();
    auto row_iterator = arrow_result_set->rowIterator(true, true);
    std::vector<std::unordered_set<std::string>> per_column_unique_strings(num_columns);
    for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
      const auto crt_row = *row_iterator++;
      for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        if (dictionary_encoded_col_idxs.find(col_idx) ==
            dictionary_encoded_col_idxs.end()) {
          continue;
        }
        const auto omnisci_variant = crt_row[col_idx];
        const auto scalar_omnisci_variant =
            boost::get<ScalarTargetValue>(&omnisci_variant);
        CHECK(scalar_omnisci_variant);
        const auto omnisci_as_str_ptr =
            boost::get<NullableString>(scalar_omnisci_variant);
        ASSERT_NE(nullptr, omnisci_as_str_ptr);
        const auto omnisci_str_notnull_ptr = boost::get<std::string>(omnisci_as_str_ptr);
        if (omnisci_str_notnull_ptr) {
          const auto omnisci_str = *omnisci_str_notnull_ptr;
          CHECK(per_column_dictionary_sets[col_idx].find(omnisci_str) !=
                per_column_dictionary_sets[col_idx].end())
              << omnisci_str;
          per_column_unique_strings[col_idx].emplace(omnisci_str);
        }
      }
    }
    for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
      if (dictionary_encoded_col_idxs.find(col_idx) ==
          dictionary_encoded_col_idxs.end()) {
        continue;
      }
      const auto omnisci_col_type = omnisci_results->getColType(col_idx);
      const auto dict_id = omnisci_col_type.get_comp_param();
      const auto str_dict_proxy = omnisci_results->getStringDictionaryProxy(dict_id);
      const size_t omnisci_dict_proxy_size = str_dict_proxy->entryCount();

      const auto col_dictionary_size = per_column_dictionary_sets[col_idx].size();
      const auto col_unique_strings = per_column_unique_strings[col_idx].size();
      const bool arrow_dictionary_definitely_sparse =
          col_dictionary_size < omnisci_dict_proxy_size;
      const bool arrow_dictionary_definitely_dense =
          col_unique_strings < col_dictionary_size;
      const double dictionary_to_result_size_ratio =
          static_cast<double>(omnisci_dict_proxy_size) / row_count;

      const bool arrow_dictionary_should_be_dense =
          row_count > min_result_size_for_bulk_dictionary_fetch &&
          dictionary_to_result_size_ratio <=
              max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch;

      if (arrow_dictionary_definitely_sparse) {
        ASSERT_EQ(col_unique_strings, col_dictionary_size);
        ASSERT_EQ(arrow_dictionary_should_be_dense, false);
      } else if (arrow_dictionary_definitely_dense) {
        ASSERT_EQ(col_dictionary_size, omnisci_dict_proxy_size);
        ASSERT_EQ(arrow_dictionary_should_be_dense, true);
      }
    }
  }

  void c_arrow(const std::string& query_string,
               const ExecutorDeviceType device_type,
               size_t min_result_size_for_bulk_dictionary_fetch,
               double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {
    auto results = run_multiple_agg(query_string, device_type);
    auto arrow_omnisci_results = result_set_arrow_loopback(
        nullptr,
        results,
        device_type,
        min_result_size_for_bulk_dictionary_fetch,
        max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);
    sqlite_comparator_.compare_arrow_output(
        arrow_omnisci_results, query_string, device_type);
    // Below we test the newly added sparse dictionary capability,
    // where only entries in a dictionary-encoded arrow column should be in the
    // corresponding dictionary (vs all the entries in the underlying OmniSci dictionary)
    check_arrow_dictionaries(
        arrow_omnisci_results.get(),
        results,
        min_result_size_for_bulk_dictionary_fetch,
        max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);
  }

  void clearCpuMemory() {
    Executor::clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL, data_mgr_.get());
  }

  BufferPoolStats getBufferPoolStats(Data_Namespace::MemoryLevel mmeory_level) {
    return ::getBufferPoolStats(data_mgr_.get(), mmeory_level);
  }

  std::shared_ptr<ArrowStorage> getStorage() { return storage_; }

  DataMgr* getDataMgr() { return data_mgr_.get(); }

  Executor* getExecutor() { return executor_.get(); }

  std::shared_ptr<CalciteJNI> getCalcite() { return calcite_; }

  ~ArrowSQLRunnerImpl() {
    storage_.reset();
    executor_.reset();
    data_mgr_.reset();
    calcite_.reset();
  }

 protected:
  ArrowSQLRunnerImpl(size_t max_gpu_mem, const std::string& udf_filename) {
    storage_ = std::make_shared<ArrowStorage>(TEST_SCHEMA_ID, "test", TEST_DB_ID);

    std::map<GpuMgrName, std::unique_ptr<GpuMgr>> gpu_mgrs;
    bool uses_gpu = false;
#ifdef HAVE_CUDA
    gpu_mgrs[GpuMgrName::CUDA] = std::make_unique<CudaMgr_Namespace::CudaMgr>(-1, 0);
    uses_gpu = true;
#elif HAVE_L0
    gpu_mgrs[GpuMgrName::L0] = std::make_unique<l0::L0Manager>();
    uses_gpu = true;
#endif

    SystemParameters system_parameters;
    system_parameters.gpu_buffer_mem_bytes = max_gpu_mem;
    data_mgr_ =
        std::make_shared<DataMgr>("", system_parameters, std::move(gpu_mgrs), uses_gpu);
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID, storage_);

    executor_ = Executor::getExecutor(
        /*executor_id=*/0,
        data_mgr_.get(),
        data_mgr_->getBufferProvider(),
        "",
        "",
        system_parameters);
    executor_->setSchemaProvider(storage_);
    executor_->setDatabaseId(TEST_DB_ID);

    calcite_ = std::make_shared<CalciteJNI>(udf_filename, 1024);
    ExtensionFunctionsWhitelist::add(calcite_->getExtensionFunctionWhitelist());
    if (!udf_filename.empty()) {
      ExtensionFunctionsWhitelist::addUdfs(calcite_->getUserDefinedFunctionWhitelist());
    }

    table_functions::TableFunctionsFactory::init();
    auto udtfs =
        table_functions::TableFunctionsFactory::get_table_funcs(/*is_runtime=*/false);
    std::vector<ExtensionFunction> udfs = {};
    calcite_->setRuntimeExtensionFunctions(udfs, udtfs, /*is_runtime=*/false);
  }

  std::shared_ptr<DataMgr> data_mgr_;
  std::shared_ptr<ArrowStorage> storage_;
  std::shared_ptr<Executor> executor_;
  std::shared_ptr<CalciteJNI> calcite_;
  SQLiteComparator sqlite_comparator_;
  int64_t schema_to_json_time_ = 0;
  int64_t calcite_time_ = 0;
  int64_t execution_time_ = 0;

  static std::unique_ptr<ArrowSQLRunnerImpl> instance_;
};

std::unique_ptr<ArrowSQLRunnerImpl> ArrowSQLRunnerImpl::instance_;

}  // namespace

void init(size_t max_gpu_mem, const std::string& udf_filename) {
  ArrowSQLRunnerImpl::init(max_gpu_mem, udf_filename);
}

void reset() {
  ArrowSQLRunnerImpl::reset();
}

bool gpusPresent() {
  return ArrowSQLRunnerImpl::get()->gpusPresent();
}

void printStats() {
  return ArrowSQLRunnerImpl::get()->printStats();
}

void createTable(const std::string& table_name,
                 const std::vector<ArrowStorage::ColumnDescription>& columns,
                 const ArrowStorage::TableOptions& options) {
  ArrowSQLRunnerImpl::get()->createTable(table_name, columns, options);
}

void dropTable(const std::string& table_name) {
  ArrowSQLRunnerImpl::get()->dropTable(table_name);
}

void insertCsvValues(const std::string& table_name, const std::string& values) {
  ArrowSQLRunnerImpl::get()->insertCsvValues(table_name, values);
}

void insertJsonValues(const std::string& table_name, const std::string& values) {
  ArrowSQLRunnerImpl::get()->insertJsonValues(table_name, values);
}

std::string getSqlQueryRelAlg(const std::string& query_str) {
  return ArrowSQLRunnerImpl::get()->getSqlQueryRelAlg(query_str);
}

ExecutionResult runSqlQuery(const std::string& sql,
                            const CompilationOptions& co,
                            const ExecutionOptions& eo) {
  return ArrowSQLRunnerImpl::get()->runSqlQuery(sql, co, eo);
}

ExecutionResult runSqlQuery(const std::string& sql,
                            ExecutorDeviceType device_type,
                            const ExecutionOptions& eo) {
  return ArrowSQLRunnerImpl::get()->runSqlQuery(sql, device_type, eo);
}

ExecutionResult runSqlQuery(const std::string& sql,
                            ExecutorDeviceType device_type,
                            bool allow_loop_joins) {
  return ArrowSQLRunnerImpl::get()->runSqlQuery(sql, device_type, allow_loop_joins);
}

ExecutionOptions getExecutionOptions(bool allow_loop_joins, bool just_explain) {
  return ArrowSQLRunnerImpl::get()->getExecutionOptions(allow_loop_joins, just_explain);
}

CompilationOptions getCompilationOptions(ExecutorDeviceType device_type) {
  return ArrowSQLRunnerImpl::get()->getCompilationOptions(device_type);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool allow_loop_joins) {
  return ArrowSQLRunnerImpl::get()->run_multiple_agg(
      query_str, device_type, allow_loop_joins);
}

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type,
                           const bool allow_loop_joins) {
  return ArrowSQLRunnerImpl::get()->run_simple_agg(
      query_str, device_type, allow_loop_joins);
}

void run_sqlite_query(const std::string& query_string) {
  ArrowSQLRunnerImpl::get()->run_sqlite_query(query_string);
}

void sqlite_batch_insert(const std::string& table_name,
                         std::vector<std::vector<std::string>>& insert_vals) {
  ArrowSQLRunnerImpl::get()->sqlite_batch_insert(table_name, insert_vals);
}

void c(const std::string& query_string, const ExecutorDeviceType device_type) {
  ArrowSQLRunnerImpl::get()->c(query_string, device_type);
}

void c(const std::string& query_string,
       const std::string& sqlite_query_string,
       const ExecutorDeviceType device_type) {
  ArrowSQLRunnerImpl::get()->c(query_string, sqlite_query_string, device_type);
}

void cta(const std::string& query_string, const ExecutorDeviceType device_type) {
  ArrowSQLRunnerImpl::get()->cta(query_string, device_type);
}

void c_arrow(const std::string& query_string,
             const ExecutorDeviceType device_type,
             size_t min_result_size_for_bulk_dictionary_fetch,
             double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {
  ArrowSQLRunnerImpl::get()->c_arrow(
      query_string,
      device_type,
      min_result_size_for_bulk_dictionary_fetch,
      max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);
}

void clearCpuMemory() {
  ArrowSQLRunnerImpl::get()->clearCpuMemory();
}

BufferPoolStats getBufferPoolStats(const Data_Namespace::MemoryLevel memory_level) {
  return ArrowSQLRunnerImpl::get()->getBufferPoolStats(memory_level);
}

std::shared_ptr<ArrowStorage> getStorage() {
  return ArrowSQLRunnerImpl::get()->getStorage();
}

DataMgr* getDataMgr() {
  return ArrowSQLRunnerImpl::get()->getDataMgr();
}

Executor* getExecutor() {
  return ArrowSQLRunnerImpl::get()->getExecutor();
}

std::shared_ptr<CalciteJNI> getCalcite() {
  return ArrowSQLRunnerImpl::get()->getCalcite();
}

RegisteredQueryHint getParsedQueryHint(const std::string& query_str) {
  return ArrowSQLRunnerImpl::get()->getParsedQueryHint(query_str);
}

std::optional<std::unordered_map<size_t, RegisteredQueryHint>> getParsedQueryHints(
    const std::string& query_str) {
  return ArrowSQLRunnerImpl::get()->getParsedQueryHints(query_str);
}

std::unique_ptr<RelAlgExecutor> makeRelAlgExecutor(const std::string& query_str) {
  return ArrowSQLRunnerImpl::get()->makeRelAlgExecutor(query_str);
}

}  // namespace TestHelpers::ArrowSQLRunner
