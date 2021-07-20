/*
 * Copyright 2020, OmniSci, Inc.
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
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TSocket.h>
#include <boost/program_options.hpp>

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

#include "gen-cpp/OmniSci.h"

TSessionId g_session_id;
std::shared_ptr<OmniSciClient> g_client;

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

}  // namespace

class ArrowIpcBasic : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS arrow_ipc_test;");
    run_ddl_statement("DROP TABLE IF EXISTS test_data_scalars;");

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

    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (1, 1.1, 'foo');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (2, 2.1, NULL);");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (NULL, 3.1, 'bar');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (4, NULL, 'hello');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (5, 5.1, 'world');");

    run_ddl_statement(
        "INSERT INTO test_data_scalars VALUES (1, 2, 3, 123.456, 0.1, 0.001);");
    run_ddl_statement(
        "INSERT INTO test_data_scalars VALUES (1, 2, 3, 345.678, 0.1, 0.001);");
  }

  void TearDown() override { run_ddl_statement("DROP TABLE IF EXISTS arrow_ipc_test;"); }
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

TEST_F(ArrowIpcBasic, IpcCpu) {
  auto data_frame =
      execute_arrow_ipc("SELECT * FROM arrow_ipc_test;", ExecutorDeviceType::CPU);

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
    char const* select =
        "SELECT plane_model, COUNT(*) FROM flights WHERE id < -1 GROUP BY plane_model;";

    std::array expected_fields{"plane_model"s, "EXPR$1"s};

    {  // wire transport
      auto data_frame = execute_arrow_ipc(
          select, ExecutorDeviceType::CPU, 0, -1, TArrowTransport::WIRE);

      auto df = ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::WIRE);

      ASSERT_EQ(df.schema->num_fields(), 2);

      EXPECT_THAT(df.schema->field_names(),
                  ::testing::UnorderedElementsAreArray(expected_fields));
    }

    {  // shared memory transport
      auto data_frame = execute_arrow_ipc(
          select, ExecutorDeviceType::CPU, 0, -1, TArrowTransport::SHARED_MEMORY);

      auto df = ArrowOutput(
          data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

      ASSERT_EQ(df.schema->num_fields(), 2);

      EXPECT_THAT(df.schema->field_names(),
                  ::testing::UnorderedElementsAreArray(expected_fields));
    }
  }

  // projection
  {
    char const* select = "SELECT * FROM flights WHERE id < -1";

    std::array expected_fields{"plane_model"s, "id"s, "dest_city"s};

    {  // wire transport
      auto data_frame = execute_arrow_ipc(
          select, ExecutorDeviceType::CPU, 0, -1, TArrowTransport::WIRE);

      auto df = ArrowOutput(data_frame, ExecutorDeviceType::CPU, TArrowTransport::WIRE);

      ASSERT_EQ(df.schema->num_fields(), 3);

      EXPECT_THAT(df.schema->field_names(),
                  ::testing::UnorderedElementsAreArray(expected_fields));
    }

    {  // shared memory transport
      auto data_frame = execute_arrow_ipc(
          select, ExecutorDeviceType::CPU, 0, -1, TArrowTransport::SHARED_MEMORY);

      auto df = ArrowOutput(
          data_frame, ExecutorDeviceType::CPU, TArrowTransport::SHARED_MEMORY);

      ASSERT_EQ(df.schema->num_fields(), 3);

      EXPECT_THAT(df.schema->field_names(),
                  ::testing::UnorderedElementsAreArray(expected_fields));
    }
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
    std::string db = "omnisci";

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
    g_client = std::make_shared<OmniSciClient>(protocol);

    g_client->connect(g_session_id, user, pwd, db);

    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = -1;
  }
  return err;
}
