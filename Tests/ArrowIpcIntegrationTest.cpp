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

#include <fmt/core.h>
#include <fmt/format.h>
#include "TestHelpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <thrift/Thrift.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TSocket.h>
#include <boost/program_options.hpp>
#include "Shared/SysDefinitions.h"
#include "Shared/ThriftJSONProtocolInclude.h"

#ifdef HAVE_CUDA
#include <arrow/gpu/cuda_api.h>
#endif

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace std::literals;

#include "Logger/Logger.h"
#include "QueryEngine/CompilationOptions.h"
#include "Shared/ArrowUtil.h"
#include "Shared/ThriftClient.h"
#include "Shared/scope.h"

#include "gen-cpp/Heavy.h"

TSessionId g_session_id;
std::shared_ptr<HeavyClient> g_client;

#ifdef HAVE_CUDA
bool g_cpu_only{false};
#else
bool g_cpu_only{true};
#endif

#define SKIP_NO_GPU()                                        \
  if (g_cpu_only) {                                          \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

namespace {

TQueryResult run_multiple_agg(const std::string& sql) {
  TQueryResult result;
  g_client->sql_execute(result, g_session_id, sql, false, "", -1, -1);
  return result;
}

class ArrowOutput {
 public:
  ArrowOutput(TDataFrame& tdf,
              const ExecutorDeviceType device_type,
              const TArrowTransport::type transport_method) {
    if (transport_method == TArrowTransport::WIRE) {
      arrow::io::BufferReader reader(
          reinterpret_cast<const uint8_t*>(tdf.df_buffer.data()), tdf.df_buffer.size());

      ARROW_ASSIGN_OR_THROW(batch_reader,
                            arrow::ipc::RecordBatchStreamReader::Open(&reader));

      auto read_result = batch_reader->ReadNext(&record_batch);
      if (read_result.code() != arrow::StatusCode::OK || !record_batch) {
        LOG(WARNING) << "Unable to read record batch";
        schema = batch_reader->schema();
      } else {
        schema = record_batch->schema();
      }

    } else {
      if (device_type == ExecutorDeviceType::CPU) {
        key_t shmem_key = -1;
        std::memcpy(&shmem_key, tdf.df_handle.data(), sizeof(key_t));
        int shmem_id = shmget(shmem_key, tdf.df_size, 0666);
        if (shmem_id < 0) {
          throw std::runtime_error("Failed to get IPC memory segment.");
        }

        auto ipc_ptr = shmat(shmem_id, NULL, 0);
        if (reinterpret_cast<int64_t>(ipc_ptr) == -1) {
          throw std::runtime_error("Failed to attach to IPC memory segment.");
        }

        arrow::io::BufferReader reader(reinterpret_cast<const uint8_t*>(ipc_ptr),
                                       tdf.df_size);
        ARROW_ASSIGN_OR_THROW(batch_reader,
                              arrow::ipc::RecordBatchStreamReader::Open(&reader));
        auto read_result = batch_reader->ReadNext(&record_batch);
        if (!read_result.ok()) {
          LOG(WARNING) << "Unable to read record batch from shared memory buffer";
        }
        schema = batch_reader->schema();
      }
      if (device_type == ExecutorDeviceType::GPU) {
        // Read schema from IPC memory
        key_t shmem_key = -1;
        std::memcpy(&shmem_key, tdf.sm_handle.data(), sizeof(key_t));
        int shmem_id = shmget(shmem_key, tdf.sm_size, 0666);
        if (shmem_id < 0) {
          throw std::runtime_error("Failed to get IPC memory segment.");
        }

        auto ipc_ptr = shmat(shmem_id, NULL, 0);
        if (reinterpret_cast<int64_t>(ipc_ptr) == -1) {
          throw std::runtime_error("Failed to attach to IPC memory segment.");
        }

        arrow::io::BufferReader reader(reinterpret_cast<const uint8_t*>(ipc_ptr),
                                       tdf.sm_size);
        auto message_reader = arrow::ipc::MessageReader::Open(&reader);

        // read schema
        std::unique_ptr<arrow::ipc::Message> schema_message;
        ARROW_ASSIGN_OR_THROW(schema_message, message_reader->ReadNextMessage());
        ARROW_ASSIGN_OR_THROW(schema,
                              arrow::ipc::ReadSchema(*schema_message, &dict_memo));

        // read dictionaries
        if (dict_memo.fields().num_fields() > 0) {
          ARROW_ASSIGN_OR_THROW(
              gpu_dict_batch_reader,
              arrow::ipc::RecordBatchStreamReader::Open(std::move(message_reader)));

          for (size_t i = 0; i < schema->fields().size(); i++) {
            auto field = schema->field(i);
            if (field->type()->id() == arrow::Type::DICTIONARY) {
              std::shared_ptr<arrow::RecordBatch> tmp_record_batch;
              ARROW_THROW_NOT_OK(gpu_dict_batch_reader->ReadNext(&tmp_record_batch));

              int64_t dict_id = -1;
              ARROW_ASSIGN_OR_THROW(dict_id,
                                    dict_memo.fields().GetFieldId({
                                        static_cast<int>(i),
                                    }));
              CHECK_GE(dict_id, 0);

              CHECK(!dict_memo.HasDictionary(dict_id));
              ARROW_THROW_NOT_OK(
                  dict_memo.AddDictionary(dict_id, tmp_record_batch->column(0)->data()));
            }
          }
        }
#ifdef HAVE_CUDA
        const size_t device_id = 0;
        arrow::cuda::CudaDeviceManager* manager;
        ARROW_ASSIGN_OR_THROW(manager, arrow::cuda::CudaDeviceManager::Instance());
        std::shared_ptr<arrow::cuda::CudaContext> context;
        ARROW_ASSIGN_OR_THROW(context, manager->GetContext(device_id));

        std::shared_ptr<arrow::cuda::CudaIpcMemHandle> cuda_handle;
        ARROW_ASSIGN_OR_THROW(cuda_handle,
                              arrow::cuda::CudaIpcMemHandle::FromBuffer(
                                  reinterpret_cast<void*>(tdf.df_handle.data())));

        std::shared_ptr<arrow::cuda::CudaBuffer> cuda_buffer;
        ARROW_ASSIGN_OR_THROW(cuda_buffer, context->OpenIpcBuffer(*cuda_handle));

        arrow::cuda::CudaBufferReader cuda_reader(cuda_buffer);

        std::unique_ptr<arrow::ipc::Message> message;
        ARROW_ASSIGN_OR_THROW(
            message, arrow::ipc::ReadMessage(&cuda_reader, arrow::default_memory_pool()));

        ARROW_ASSIGN_OR_THROW(
            record_batch,
            arrow::ipc::ReadRecordBatch(
                *message, schema, &dict_memo, arrow::ipc::IpcReadOptions::Defaults()));

        // Copy data to host for checking
        std::shared_ptr<arrow::Buffer> host_buffer;
        const int64_t size = cuda_buffer->size();
        ARROW_ASSIGN_OR_THROW(host_buffer,
                              AllocateBuffer(size, arrow::default_memory_pool()));
        ARROW_THROW_NOT_OK(cuda_buffer->CopyToHost(0, size, host_buffer->mutable_data()));

        arrow::io::BufferReader cpu_reader(host_buffer);
        ARROW_ASSIGN_OR_THROW(
            record_batch,
            arrow::ipc::ReadRecordBatch(record_batch->schema(),
                                        &dict_memo,
                                        arrow::ipc::IpcReadOptions::Defaults(),
                                        &cpu_reader));

#endif
      }
    }
  }
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;
  std::shared_ptr<arrow::RecordBatch> record_batch;
  std::shared_ptr<arrow::RecordBatchReader> gpu_dict_batch_reader;
  arrow::ipc::DictionaryMemo dict_memo;

 private:
};

TDataFrame execute_arrow_ipc(
    const std::string& sql,
    const ExecutorDeviceType device_type,
    const size_t device_id = 0,
    const int32_t first_n = -1,
    const TArrowTransport::type transport_method = TArrowTransport::type::SHARED_MEMORY) {
  TDataFrame result;
  g_client->sql_execute_df(
      result,
      g_session_id,
      sql,
      device_type == ExecutorDeviceType::GPU ? TDeviceType::GPU : TDeviceType::CPU,
      0,
      first_n,
      transport_method);
  return result;
}

void deallocate_df(const TDataFrame& df,
                   const ExecutorDeviceType device_type,
                   const size_t device_id = 0) {
  g_client->deallocate_df(
      g_session_id,
      df,
      device_type == ExecutorDeviceType::GPU ? TDeviceType::GPU : TDeviceType::CPU,
      device_id);
}

void run_ddl_statement(const std::string& ddl) {
  run_multiple_agg(ddl);
}

// Verify that column types match
void test_scalar_values(const std::shared_ptr<arrow::RecordBatch>& read_batch) {
  using namespace arrow;
  using namespace std;

  ASSERT_EQ(read_batch->schema()->num_fields(), 6);

  const std::vector expected_types = {
      make_pair(Int16Type::type_id, Int16Type::type_name()),
      make_pair(Int32Type::type_id, Int32Type::type_name()),
      make_pair(Int64Type::type_id, Int64Type::type_name()),
      make_pair(Decimal128Type::type_id, Decimal128Type::type_name()),
      make_pair(FloatType::type_id, FloatType::type_name()),
      make_pair(DoubleType::type_id, DoubleType::type_name())};

  for (size_t i = 0; i < expected_types.size(); i++) {
    auto [type, type_name] = expected_types[i];

    const auto arr = read_batch->column(i);
    ASSERT_EQ(arr->type()->id(), type)
        << fmt::format("Expected column {} to have type {}", i, type_name);
  }
}

void test_text_values(const std::shared_ptr<arrow::RecordBatch>& record_batch) {
  using namespace std;

  // string column
  ASSERT_EQ(record_batch->schema()->num_fields(), 1);
  auto column = record_batch->column(0);
  const auto& string_array = static_cast<const arrow::StringArray&>(*column);

  std::shared_ptr<arrow::StringArray> text_truth_array;
  {
    arrow::StringBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(
        builder.AppendValues(std::vector<std::string>{"hello", "", "world", "", "text"}));
    ARROW_THROW_NOT_OK(builder.Finish(&text_truth_array));
  }
  std::vector<bool> null_strings{0, 1, 0, 1, 0};
  for (int i = 0; i < string_array.length(); i++) {
    if (null_strings[i]) {
      ASSERT_TRUE(string_array.IsNull(i));
    } else {
      const auto str = string_array.GetString(i);
      ASSERT_EQ(str, text_truth_array->GetString(i));
    }
  }
}

// Verify that column types match
void test_array_values(const std::shared_ptr<arrow::RecordBatch>& read_batch) {
  using namespace arrow;
  using namespace std;

  const std::vector expected_types = {
      make_pair(BooleanType::type_id, BooleanType::type_name()),
      make_pair(Int8Type::type_id, Int8Type::type_name()),
      make_pair(Int16Type::type_id, Int16Type::type_name()),
      make_pair(Int32Type::type_id, Int32Type::type_name()),
      make_pair(Int64Type::type_id, Int64Type::type_name()),
      make_pair(FloatType::type_id, FloatType::type_name()),
      make_pair(DoubleType::type_id, DoubleType::type_name())};

  ASSERT_EQ(read_batch->schema()->num_fields(), 7);

  for (size_t i = 0; i < expected_types.size(); i++) {
    auto [type, type_name] = expected_types[i];

    const auto arr = read_batch->column(i);
    const auto& list = static_cast<const arrow::ListArray&>(*arr);
    ASSERT_EQ(list.value_type()->id(), type)
        << fmt::format("Expected column {} to have type {}", i, type_name);
    std::string bool_expected = R"([
  [
    true,
    false,
    true
  ],
  [],
  null,
  [
    true,
    null,
    false
  ]
])";

    std::string expected = R"([
  [
    0,
    1,
    2
  ],
  [],
  null,
  [
    0,
    null,
    2
  ]
])";
    std::stringstream ss;
    ss << *arr;
    if (i == 0) {
      ASSERT_EQ(bool_expected, ss.str());
    } else {
      ASSERT_EQ(expected, ss.str());
    }
  }
}

void test_array_text_values(const std::shared_ptr<arrow::RecordBatch>& read_batch) {
  using namespace arrow;
  using namespace arrow::internal;
  ASSERT_EQ(read_batch->schema()->num_fields(), 1);

  const auto column = read_batch->column(0);
  const auto& list_array = dynamic_cast<arrow::ListArray&>(*column);

  std::vector<bool> null_values = {0, 0, 0, 1, 0};
  std::vector<std::vector<std::string>> values;

  for (int i = 0; i < list_array.length(); i++) {
    std::shared_ptr<Array> array = list_array.value_slice(i);
    const auto dict = dynamic_cast<DictionaryArray*>(array.get());
    if (!null_values[i]) {
      values.emplace_back(std::vector<std::string>());
      for (int j = 0; j < dict->indices()->length(); j++) {
        int64_t idx = dict->GetValueIndex(j);
        if (idx < 6) {
          values.back().emplace_back(
              dict->dictionary()->GetScalar(idx).ValueOrDie()->ToString());
        } else {
          values.back().emplace_back("");
        }
      }
    } else {
      ASSERT_EQ(dict->indices()->length(), 0);
    }
  }

  std::vector<std::vector<std::string>> expected = {
      {"hello", "world", "hello", "test", "world"},
      {"1", "2", "3"},
      {},
      {"hello", "", "world"}};
  ASSERT_EQ(expected, values);
}

}  // namespace

class ArrowIpcBasic : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS arrow_ipc_test;");
    run_ddl_statement("DROP TABLE IF EXISTS test_data_scalars;");
    run_ddl_statement("DROP TABLE IF EXISTS test_data_text;");
    run_ddl_statement("DROP TABLE IF EXISTS test_data_array;");
    run_ddl_statement("DROP TABLE IF EXISTS test_data_array_text;");

    run_ddl_statement(R"(
      CREATE TABLE
        arrow_ipc_test(x INT,
                       y DOUBLE,
                       t TEXT ENCODING DICT(32))
        WITH (FRAGMENT_SIZE=2);
    )");

    run_ddl_statement(R"(
        CREATE TABLE
          test_data_scalars(smallint_ SMALLINT,
                            int_ INT,
                            bigint_ BIGINT,
                            dec_ DECIMAL(12,3),
                            float_ FLOAT,
                            double_ DOUBLE);
    )");

    run_ddl_statement(R"(
        CREATE TABLE
          test_data_text(text_ TEXT ENCODING NONE);
    )");

    run_ddl_statement(R"(
        CREATE TABLE
          test_data_array(b  BOOLEAN[],
                          i1 TINYINT[],
                          i2 SMALLINT[],
                          i4 INT[],
                          i8 BIGINT[],
                          f4 FLOAT[],
                          f8 DOUBLE[]);
    )");

    run_ddl_statement(R"(
        CREATE TABLE
          test_data_array_text(t TEXT[] ENCODING DICT);
    )");

    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (1, 1.1, 'foo');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (2, 2.1, NULL);");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (NULL, 3.1, 'bar');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (4, NULL, 'hello');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (5, 5.1, 'world');");

    run_ddl_statement(
        "INSERT INTO test_data_scalars VALUES (1, 2, 3, 123.456, 0.1, 0.001);");
    run_ddl_statement(
        "INSERT INTO test_data_scalars VALUES (1, 2, 3, 345.678, 0.1, 0.001);");

    run_ddl_statement("INSERT INTO test_data_text VALUES ('hello');");
    run_ddl_statement("INSERT INTO test_data_text VALUES (NULL);");
    run_ddl_statement("INSERT INTO test_data_text VALUES ('world');");
    run_ddl_statement("INSERT INTO test_data_text VALUES ('');");
    run_ddl_statement("INSERT INTO test_data_text VALUES ('text');");

    {
      std::string arr = "{0, 1, 2}";
      std::string barr = "{'true', 'false', 'true'}";
      std::string tarr = "{'hello', 'world', 'hello', 'test', 'world'}";
      std::string iarr = "{'1', '2', '3'}";
      std::string empty_arr = "{}";
      std::string null_arr = "NULL";
      std::string null_elem_arr = "{0, NULL, 2}";
      std::string null_elem_barr = "{'true', NULL, 'false'}";
      std::string null_elem_tarr = "{'hello', NULL, 'world'}";

      const std::vector vals = {make_pair(barr, arr),
                                make_pair(empty_arr, empty_arr),
                                make_pair(null_arr, null_arr),
                                make_pair(null_elem_barr, null_elem_arr)};
      for (const auto& val : vals) {
        auto [b, i] = val;
        std::string query = fmt::format(
            "INSERT INTO test_data_array VALUES ({0}, {1}, {1}, {1}, {1}, {1}, {1});",
            b,
            i);
        run_ddl_statement(query);
      }

      for (const auto& t : {tarr, iarr, empty_arr, null_arr, null_elem_tarr}) {
        std::string query =
            fmt::format("INSERT INTO test_data_array_text VALUES ({0});", t);
        run_ddl_statement(query);
      }
    }
  }

  void TearDown()
      override { /*run_ddl_statement("DROP TABLE IF EXISTS arrow_ipc_test;");*/
  }
};

TEST_F(ArrowIpcBasic, IpcWire) {
  auto data_frame = execute_arrow_ipc("SELECT * FROM arrow_ipc_test;",
                                      ExecutorDeviceType::CPU,
                                      0,
                                      -1,
                                      TArrowTransport::type::WIRE);
  auto df = ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::type::WIRE);

  ASSERT_EQ(df.schema->num_fields(), 3);

  // int column
  auto int_array = df.record_batch->column(0);
  ASSERT_EQ(int_array->type()->id(), arrow::Type::type::INT32);
  std::shared_ptr<arrow::Array> int_truth_array;
  {
    arrow::Int32Builder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(std::vector<int32_t>{1, 2, 3, 4, 5},
                                            std::vector<bool>{1, 1, 0, 1, 1}));
    ARROW_THROW_NOT_OK(builder.Finish(&int_truth_array));
  }
  ASSERT_TRUE(int_array->Equals(int_truth_array));

  // double column
  auto double_array = df.record_batch->column(1);
  ASSERT_EQ(double_array->type()->id(), arrow::Type::type::DOUBLE);
  std::shared_ptr<arrow::Array> double_truth_array;
  {
    arrow::DoubleBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(std::vector<double>{1.1, 2.1, 3.1, 4.1, 5.1},
                                            std::vector<bool>{1, 1, 1, 0, 1}));
    ARROW_THROW_NOT_OK(builder.Finish(&double_truth_array));
  }
  ASSERT_TRUE(double_array->ApproxEquals(double_truth_array));

  // string column
  auto string_array = df.record_batch->column(2);
  ASSERT_EQ(string_array->type()->id(), arrow::Type::type::DICTIONARY);
  std::shared_ptr<arrow::Array> text_truth_array;
  {
    arrow::StringBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(
        std::vector<std::string>{"foo", "", "bar", "hello", "world"}));
    ARROW_THROW_NOT_OK(builder.Finish(&text_truth_array));
  }
  const auto& dict_array = static_cast<const arrow::DictionaryArray&>(*string_array);
  const auto& indices = static_cast<const arrow::Int32Array&>(*dict_array.indices());
  const auto& dictionary =
      static_cast<const arrow::StringArray&>(*dict_array.dictionary());
  const auto& truth_strings = static_cast<const arrow::StringArray&>(*text_truth_array);
  std::vector<bool> null_strings{1, 0, 1, 1, 1};
  for (int i = 0; i < string_array->length(); i++) {
    if (!null_strings[i]) {
      ASSERT_TRUE(indices.IsNull(i));
    } else {
      const auto index = indices.Value(i);
      const auto str = dictionary.GetString(index);
      ASSERT_EQ(str, truth_strings.GetString(i));
    }
  }
}

namespace {

void check_cpu_dataframe(TDataFrame& data_frame) {
  ASSERT_TRUE(data_frame.df_size > 0);

  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

  ASSERT_EQ(df.schema->num_fields(), 3);

  // int column
  auto int_array = df.record_batch->column(0);
  ASSERT_EQ(int_array->type()->id(), arrow::Type::type::INT32);
  std::shared_ptr<arrow::Array> int_truth_array;
  {
    arrow::Int32Builder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(std::vector<int32_t>{1, 2, 3, 4, 5},
                                            std::vector<bool>{1, 1, 0, 1, 1}));
    ARROW_THROW_NOT_OK(builder.Finish(&int_truth_array));
  }
  ASSERT_TRUE(int_array->Equals(int_truth_array));

  // double column
  auto double_array = df.record_batch->column(1);
  ASSERT_EQ(double_array->type()->id(), arrow::Type::type::DOUBLE);
  std::shared_ptr<arrow::Array> double_truth_array;
  {
    arrow::DoubleBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(std::vector<double>{1.1, 2.1, 3.1, 4.1, 5.1},
                                            std::vector<bool>{1, 1, 1, 0, 1}));
    ARROW_THROW_NOT_OK(builder.Finish(&double_truth_array));
  }
  ASSERT_TRUE(double_array->ApproxEquals(double_truth_array));

  // string column
  auto string_array = df.record_batch->column(2);
  ASSERT_EQ(string_array->type()->id(), arrow::Type::type::DICTIONARY);
  std::shared_ptr<arrow::Array> text_truth_array;
  {
    arrow::StringBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(
        std::vector<std::string>{"foo", "", "bar", "hello", "world"}));
    ARROW_THROW_NOT_OK(builder.Finish(&text_truth_array));
  }
  const auto& dict_array = static_cast<const arrow::DictionaryArray&>(*string_array);
  const auto& indices = static_cast<const arrow::Int32Array&>(*dict_array.indices());
  const auto& dictionary =
      static_cast<const arrow::StringArray&>(*dict_array.dictionary());
  const auto& truth_strings = static_cast<const arrow::StringArray&>(*text_truth_array);
  std::vector<bool> null_strings{1, 0, 1, 1, 1};
  for (int i = 0; i < string_array->length(); i++) {
    if (!null_strings[i]) {
      ASSERT_TRUE(indices.IsNull(i));
    } else {
      const auto index = indices.Value(i);
      const auto str = dictionary.GetString(index);
      ASSERT_EQ(str, truth_strings.GetString(i));
    }
  }
}

}  // namespace

TEST_F(ArrowIpcBasic, IpcCpu) {
  auto data_frame =
      execute_arrow_ipc("SELECT * FROM arrow_ipc_test;", ExecutorDeviceType::CPU);

  check_cpu_dataframe(data_frame);

  deallocate_df(data_frame, ExecutorDeviceType::CPU);
}

TEST_F(ArrowIpcBasic, IpcCpuDictionarySubquery) {
  auto data_frame = execute_arrow_ipc(
      R"(SELECT CASE WHEN x IS NOT NULL THEN t ELSE 'baz' END FROM arrow_ipc_test ORDER BY 1;)",
      ExecutorDeviceType::CPU);

  ASSERT_TRUE(data_frame.df_size > 0);

  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);
  ASSERT_EQ(df.schema->num_fields(), 1);

  // string column
  auto string_array = df.record_batch->column(0);
  ASSERT_EQ(string_array->type()->id(), arrow::Type::type::DICTIONARY);
  std::shared_ptr<arrow::Array> text_truth_array;
  {
    arrow::StringBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(
        std::vector<std::string>{"baz", "foo", "hello", "world", ""}));
    ARROW_THROW_NOT_OK(builder.Finish(&text_truth_array));
  }
  const auto& dict_array = static_cast<const arrow::DictionaryArray&>(*string_array);
  const auto& indices = static_cast<const arrow::Int32Array&>(*dict_array.indices());
  const auto& dictionary =
      static_cast<const arrow::StringArray&>(*dict_array.dictionary());
  const auto& truth_strings = static_cast<const arrow::StringArray&>(*text_truth_array);
  std::vector<bool> null_strings{1, 1, 1, 1, 0};
  for (int i = 0; i < string_array->length(); i++) {
    if (!null_strings[i]) {
      EXPECT_TRUE(indices.IsNull(i));
    } else {
      const auto index = indices.Value(i);
      const auto str = dictionary.GetString(index);
      EXPECT_EQ(str, truth_strings.GetString(i));
    }
  }

  deallocate_df(data_frame, ExecutorDeviceType::CPU);
}

TEST_F(ArrowIpcBasic, IpcCpuWithCpuExecution) {
  g_client->set_execution_mode(g_session_id, TExecuteMode::CPU);
  ScopeGuard reset_execution_mode = [&] {
    g_client->set_execution_mode(g_session_id, TExecuteMode::GPU);
  };

  auto data_frame =
      execute_arrow_ipc("SELECT * FROM arrow_ipc_test;", ExecutorDeviceType::CPU);

  check_cpu_dataframe(data_frame);

  deallocate_df(data_frame, ExecutorDeviceType::CPU);
}

TEST_F(ArrowIpcBasic, IpcCpuScalarValues) {
  auto data_frame =
      execute_arrow_ipc("SELECT * FROM test_data_scalars;", ExecutorDeviceType::CPU);

  ASSERT_TRUE(data_frame.df_size > 0);

  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

  test_scalar_values(df.record_batch);

  deallocate_df(data_frame, ExecutorDeviceType::CPU);
}

TEST_F(ArrowIpcBasic, IpcGpuScalarValues) {
  const size_t device_id = 0;
  if (g_cpu_only) {
    LOG(ERROR) << "Test not valid in CPU mode.";
    return;
  }
  auto data_frame = execute_arrow_ipc(
      "SELECT * FROM test_data_scalars;", ExecutorDeviceType::GPU, device_id);

  ASSERT_TRUE(data_frame.df_size > 0);
#ifdef HAVE_CUDA
  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::GPU, TArrowTransport::SHARED_MEMORY);

  test_scalar_values(df.record_batch);
#else
  ASSERT_TRUE(false) << "Test should be skipped in CPU-only mode!";
#endif
  deallocate_df(data_frame, ExecutorDeviceType::GPU);
}

TEST_F(ArrowIpcBasic, IpcCpuTextValues) {
  auto data_frame =
      execute_arrow_ipc("SELECT * FROM test_data_text;", ExecutorDeviceType::CPU);

  ASSERT_TRUE(data_frame.df_size > 0);

  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

  test_text_values(df.record_batch);

  deallocate_df(data_frame, ExecutorDeviceType::CPU);
}

TEST_F(ArrowIpcBasic, IpcCpuArrayValues) {
  auto data_frame =
      execute_arrow_ipc("SELECT * FROM test_data_array;", ExecutorDeviceType::CPU);

  ASSERT_TRUE(data_frame.df_size > 0);

  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

  test_array_values(df.record_batch);

  deallocate_df(data_frame, ExecutorDeviceType::CPU);
}

TEST_F(ArrowIpcBasic, IpcCpuArrayTextValues) {
  auto data_frame =
      execute_arrow_ipc("SELECT * FROM test_data_array_text;", ExecutorDeviceType::CPU);

  ASSERT_TRUE(data_frame.df_size > 0);

  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

  test_array_text_values(df.record_batch);

  deallocate_df(data_frame, ExecutorDeviceType::CPU);
}

TEST_F(ArrowIpcBasic, IpcGpu) {
  const size_t device_id = 0;
  if (g_cpu_only) {
    LOG(ERROR) << "Test not valid in CPU mode.";
    return;
  }
  auto data_frame = execute_arrow_ipc(
      "SELECT * FROM arrow_ipc_test;", ExecutorDeviceType::GPU, device_id);
  auto ipc_handle = data_frame.sm_handle;
  auto ipc_handle_size = data_frame.sm_size;

  ASSERT_TRUE(ipc_handle_size > 0);

#ifdef HAVE_CUDA
  auto df =
      ArrowOutput(data_frame, ExecutorDeviceType::GPU, TArrowTransport::SHARED_MEMORY);

  // int column
  auto int_array = df.record_batch->column(0);
  ASSERT_EQ(int_array->type()->id(), arrow::Type::type::INT32);
  std::shared_ptr<arrow::Array> int_truth_array;
  {
    arrow::Int32Builder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(std::vector<int32_t>{1, 2, 3, 4, 5},
                                            std::vector<bool>{1, 1, 0, 1, 1}));
    ARROW_THROW_NOT_OK(builder.Finish(&int_truth_array));
  }
  ASSERT_TRUE(int_array->Equals(int_truth_array));

  // double column
  auto double_array = df.record_batch->column(1);
  ASSERT_EQ(double_array->type()->id(), arrow::Type::type::DOUBLE);
  std::shared_ptr<arrow::Array> double_truth_array;
  {
    arrow::DoubleBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(std::vector<double>{1.1, 2.1, 3.1, 4.1, 5.1},
                                            std::vector<bool>{1, 1, 1, 0, 1}));
    ARROW_THROW_NOT_OK(builder.Finish(&double_truth_array));
  }
  ASSERT_TRUE(double_array->ApproxEquals(double_truth_array));

  // string column
  auto string_array = df.record_batch->column(2);
  ASSERT_EQ(string_array->type()->id(), arrow::Type::type::DICTIONARY);
  std::shared_ptr<arrow::Array> text_truth_array;
  {
    arrow::StringBuilder builder(arrow::default_memory_pool());
    ARROW_THROW_NOT_OK(builder.AppendValues(
        std::vector<std::string>{"foo", "", "bar", "hello", "world"}));
    ARROW_THROW_NOT_OK(builder.Finish(&text_truth_array));
  }
  const auto& dict_array = static_cast<const arrow::DictionaryArray&>(*string_array);
  const auto& indices = static_cast<const arrow::Int32Array&>(*dict_array.indices());
  const auto& dictionary =
      static_cast<const arrow::StringArray&>(*dict_array.dictionary());
  const auto& truth_strings = static_cast<const arrow::StringArray&>(*text_truth_array);
  std::vector<bool> null_strings{1, 0, 1, 1, 1};
  for (int i = 0; i < string_array->length(); i++) {
    if (!null_strings[i]) {
      ASSERT_TRUE(indices.IsNull(i));
    } else {
      const auto index = indices.Value(i);
      const auto str = dictionary.GetString(index);
      ASSERT_EQ(str, truth_strings.GetString(i));
    }
  }
#else
  ASSERT_TRUE(false) << "Test should be skipped in CPU-only mode!";
#endif
  deallocate_df(data_frame, ExecutorDeviceType::GPU);
}

TEST_F(ArrowIpcBasic, IpcGpuWithCpuQuery) {
  const size_t device_id = 0;
  if (g_cpu_only) {
    LOG(ERROR) << "Test not valid in CPU mode.";
    return;
  }

  g_client->set_execution_mode(g_session_id, TExecuteMode::CPU);
  ScopeGuard reset_execution_mode = [&] {
    g_client->set_execution_mode(g_session_id, TExecuteMode::GPU);
  };

  EXPECT_ANY_THROW(execute_arrow_ipc(
      "SELECT * FROM arrow_ipc_test;", ExecutorDeviceType::GPU, device_id));
}

void check_empty_result_validity(const std::string& query,
                                 const std::vector<std::string>& expected_fields) {
  // Make following an int32_t type to match ArrowOutput::num_fields()
  const int32_t num_expected_fields = static_cast<int32_t>(expected_fields.size());
  {  // wire transport
    auto data_frame =
        execute_arrow_ipc(query, ExecutorDeviceType::CPU, 0, -1, TArrowTransport::WIRE);

    auto df = ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::WIRE);

    ASSERT_EQ(df.schema->num_fields(), num_expected_fields);

    EXPECT_THAT(df.schema->field_names(),
                ::testing::UnorderedElementsAreArray(expected_fields));
  }

  {  // shared memory transport
    auto data_frame = execute_arrow_ipc(
        query, ExecutorDeviceType::CPU, 0, -1, TArrowTransport::SHARED_MEMORY);

    auto df =
        ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

    ASSERT_EQ(df.schema->num_fields(), num_expected_fields);

    EXPECT_THAT(df.schema->field_names(),
                ::testing::UnorderedElementsAreArray(expected_fields));
  }
}

TEST_F(ArrowIpcBasic, EmptyResultSet) {
  char const* drop_flights = "DROP TABLE IF EXISTS flights;";
  run_ddl_statement(drop_flights);
  std::string create_flights =
      "CREATE TABLE flights (id INT, plane_model TEXT ENCODING DICT(32), dest_city TEXT "
      "ENCODING DICT(32)) WITH (fragment_size = 2);";
  run_ddl_statement(create_flights);
  std::vector<std::pair<int, std::string>> plane_models;
  for (int i = 1; i < 10; i++) {
    plane_models.emplace_back(i, "B-" + std::to_string(i));
  }

  for (const auto& [id, plane_model] : plane_models) {
    for (auto dest_city : {"Austin", "Dallas", "Chicago"}) {
      std::string const insert = fmt::format(
          "INSERT INTO flights VALUES ({}, '{}', '{}');", id, plane_model, dest_city);
      run_multiple_agg(insert);
    }
  }

  // group by
  {
    const std::string query =
        "SELECT plane_model, COUNT(*) FROM flights WHERE id < -1 GROUP BY plane_model;";

    std::vector<std::string> expected_fields{"plane_model", "EXPR$1"};

    check_empty_result_validity(query, expected_fields);
  }

  // projection
  {
    const std::string query = "SELECT * FROM flights WHERE id < -1;";

    const std::vector<std::string> expected_fields{"plane_model", "id", "dest_city"};

    check_empty_result_validity(query, expected_fields);
  }

  // non-empty projection result, but empty when offset is taken into account
  {
    const std::string query =
        "SELECT * FROM flights WHERE id > 5 ORDER BY id LIMIT 50 OFFSET 50;";

    const std::vector<std::string> expected_fields{"plane_model", "id", "dest_city"};

    check_empty_result_validity(query, expected_fields);
  }

  // non-empty group by result, but empty when offset is taken into account
  {
    const std::string query =
        "SELECT dest_city, COUNT(*) AS n, AVG(id) AS avg_id FROM flights GROUP BY "
        "dest_city "
        "ORDER "
        "BY n DESC LIMIT 5 OFFSET 50;";

    const std::vector<std::string> expected_fields{"dest_city", "n", "avg_id"};

    check_empty_result_validity(query, expected_fields);
  }
}

int main(int argc, char* argv[]) {
  int err = 0;
  TestHelpers::init_logger_stderr_only(argc, argv);

  namespace po = boost::program_options;

  try {
    testing::InitGoogleTest(&argc, argv);

    std::string host = "localhost";
    int port = 6274;
    std::string cert = "";

    std::string user = "admin";
    std::string pwd = "HyperInteractive";
    std::string db = shared::kDefaultDbName;

    po::options_description desc("Options");

    desc.add_options()(
        "host",
        po::value<std::string>(&host)->default_value(host)->implicit_value(host),
        "hostname of target server");
    desc.add_options()("port",
                       po::value<int>(&port)->default_value(port)->implicit_value(port),
                       "tcp port of target server");
    desc.add_options()(
        "cert",
        po::value<std::string>(&cert)->default_value(cert)->implicit_value(cert),
        "tls/ssl certificate to use for contacting target server");
    desc.add_options()(
        "user",
        po::value<std::string>(&user)->default_value(user)->implicit_value(user),
        "user name to connect as");
    desc.add_options()(
        "pwd",
        po::value<std::string>(&pwd)->default_value(pwd)->implicit_value(pwd),
        "password to connect with");
    desc.add_options()("db",
                       po::value<std::string>(&db)->default_value(db)->implicit_value(db),
                       "db to connect to");
    desc.add_options()(
        "cpu-only",
        po::value<bool>(&g_cpu_only)->default_value(g_cpu_only)->implicit_value(true),
        "Only run CPU tests.");
    desc.add_options()("test-help",
                       "Print all Arrow IPC Integration Test specific options (for gtest "
                       "options use `--help`).");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    if (vm.count("test-help")) {
      std::cout << "Usage: ArrowIpcIntegrationTest" << std::endl << std::endl;
      std::cout << desc << std::endl;
      return 0;
    }

    std::shared_ptr<ThriftClientConnection> conn_mgr;
    conn_mgr = std::make_shared<ThriftClientConnection>();

    auto transport = conn_mgr->open_buffered_client_transport(host, port, cert);
    transport->open();
    auto protocol = std::make_shared<TBinaryProtocol>(transport);
    g_client = std::make_shared<HeavyClient>(protocol);

    g_client->connect(g_session_id, user, pwd, db);

    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = -1;
  }
  return err;
}
