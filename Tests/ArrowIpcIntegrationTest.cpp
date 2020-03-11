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

#include "TestHelpers.h"

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
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

#include "QueryEngine/ArrowUtil.h"
#include "QueryEngine/CompilationOptions.h"
#include "Shared/Logger.h"
#include "Shared/ThriftClient.h"

#include "gen-cpp/MapD.h"

TSessionId g_session_id;
std::shared_ptr<MapDClient> g_client;

bool g_cpu_only{false};

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

TDataFrame execute_arrow_ipc(const std::string& sql,
                             const ExecutorDeviceType device_type,
                             const size_t device_id = 0,
                             const int32_t first_n = -1) {
  TDataFrame result;
  g_client->sql_execute_df(
      result,
      g_session_id,
      sql,
      device_type == ExecutorDeviceType::GPU ? TDeviceType::GPU : TDeviceType::CPU,
      0,
      first_n);
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

}  // namespace

class ArrowIpcBasic : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS arrow_ipc_test;");
    run_ddl_statement(
        "CREATE TABLE arrow_ipc_test(x INT, y DOUBLE, t TEXT ENCODING DICT(32)) WITH "
        "(FRAGMENT_SIZE=2);");

    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (1, 1.1, 'foo');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (2, 2.1, NULL);");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (NULL, 3.1, 'bar');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (4, NULL, 'hello');");
    run_ddl_statement("INSERT INTO arrow_ipc_test VALUES (5, 5.1, 'world');");
  }

  void TearDown() override { run_ddl_statement("DROP TABLE IF EXISTS arrow_ipc_test;"); }
};

TEST_F(ArrowIpcBasic, IpcCpu) {
  auto data_frame =
      execute_arrow_ipc("SELECT * FROM arrow_ipc_test;", ExecutorDeviceType::CPU);
  auto ipc_handle = data_frame.df_handle;
  auto ipc_handle_size = data_frame.df_size;

  ASSERT_TRUE(ipc_handle_size > 0);

  key_t shmem_key = -1;
  std::memcpy(&shmem_key, ipc_handle.data(), sizeof(key_t));
  int shmem_id = shmget(shmem_key, ipc_handle_size, 0666);
  if (shmem_id < 0) {
    throw std::runtime_error("Failed to get IPC memory segment.");
  }

  auto ipc_ptr = shmat(shmem_id, NULL, 0);
  if (reinterpret_cast<int64_t>(ipc_ptr) == -1) {
    throw std::runtime_error("Failed to attach to IPC memory segment.");
  }

  arrow::io::BufferReader reader(reinterpret_cast<const uint8_t*>(ipc_ptr),
                                 ipc_handle_size);
  std::shared_ptr<arrow::RecordBatchReader> batch_reader;
  ARROW_THROW_NOT_OK(arrow::ipc::RecordBatchStreamReader::Open(&reader, &batch_reader));
  std::shared_ptr<arrow::RecordBatch> read_batch;
  ARROW_THROW_NOT_OK(batch_reader->ReadNext(&read_batch));

  ASSERT_EQ(read_batch->schema()->num_fields(), 3);

  // int column
  auto int_array = read_batch->column(0);
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
  auto double_array = read_batch->column(1);
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
  auto string_array = read_batch->column(2);
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

  // Read schema from IPC memory
  key_t shmem_key = -1;
  std::memcpy(&shmem_key, ipc_handle.data(), sizeof(key_t));
  int shmem_id = shmget(shmem_key, ipc_handle_size, 0666);
  if (shmem_id < 0) {
    throw std::runtime_error("Failed to get IPC memory segment.");
  }

  auto ipc_ptr = shmat(shmem_id, NULL, 0);
  if (reinterpret_cast<int64_t>(ipc_ptr) == -1) {
    throw std::runtime_error("Failed to attach to IPC memory segment.");
  }

  arrow::io::BufferReader reader(reinterpret_cast<const uint8_t*>(ipc_ptr),
                                 ipc_handle_size);
  auto message_reader = arrow::ipc::MessageReader::Open(&reader);

  // read schema
  std::shared_ptr<arrow::Schema> schema;
  arrow::ipc::DictionaryMemo memo;
  std::unique_ptr<arrow::ipc::Message> schema_message;
  ARROW_THROW_NOT_OK(message_reader->ReadNextMessage(&schema_message));
  ARROW_THROW_NOT_OK(arrow::ipc::ReadSchema(*schema_message, &memo, &schema));

  // read dictionaries
  std::shared_ptr<arrow::RecordBatchReader> dict_batch_reader;
  ARROW_THROW_NOT_OK(arrow::ipc::RecordBatchStreamReader::Open(std::move(message_reader),
                                                               &dict_batch_reader));
  for (int i = 0; i < schema->num_fields(); i++) {
    auto field = schema->field(i);
    if (field->type()->id() == arrow::Type::DICTIONARY) {
      std::shared_ptr<arrow::RecordBatch> tmp_record_batch;
      ARROW_THROW_NOT_OK(dict_batch_reader->ReadNext(&tmp_record_batch));
      ASSERT_TRUE(tmp_record_batch);

      // auto dict_schema = std::make_shared<arrow::Schema>(
      // std::vector<std::shared_ptr<arrow::Field>>{field});
      // ARROW_THROW_NOT_OK(arrow::ipc::ReadRecordBatch(
      // *dict_message, dict_schema, &memo, &tmp_record_batch));
      ASSERT_EQ(tmp_record_batch->num_columns(), 1);
      int64_t dict_id = -1;
      ARROW_THROW_NOT_OK(memo.GetId(*field, &dict_id));
      CHECK_GE(dict_id, 0);

      CHECK(!memo.HasDictionary(dict_id));
      ARROW_THROW_NOT_OK(memo.AddDictionary(dict_id, tmp_record_batch->column(0)));
    }
  }

  // read data
  std::shared_ptr<arrow::RecordBatch> read_batch;
#ifdef HAVE_CUDA
  arrow::cuda::CudaDeviceManager* manager;
  ARROW_THROW_NOT_OK(arrow::cuda::CudaDeviceManager::GetInstance(&manager));
  std::shared_ptr<arrow::cuda::CudaContext> context;
  ARROW_THROW_NOT_OK(manager->GetContext(device_id, &context));

  std::shared_ptr<arrow::cuda::CudaIpcMemHandle> cuda_handle;
  ARROW_THROW_NOT_OK(arrow::cuda::CudaIpcMemHandle::FromBuffer(
      reinterpret_cast<void*>(data_frame.df_handle.data()), &cuda_handle));

  std::shared_ptr<arrow::cuda::CudaBuffer> cuda_buffer;
  ARROW_THROW_NOT_OK(context->OpenIpcBuffer(*cuda_handle, &cuda_buffer));

  arrow::cuda::CudaBufferReader cuda_reader(cuda_buffer);

  std::unique_ptr<arrow::ipc::Message> message;
  ARROW_THROW_NOT_OK(
      arrow::cuda::ReadMessage(&cuda_reader, arrow::default_memory_pool(), &message));

  ASSERT_TRUE(message);

  ARROW_THROW_NOT_OK(arrow::ipc::ReadRecordBatch(*message, schema, &memo, &read_batch));

  // Copy data to host for checking
  std::shared_ptr<arrow::Buffer> host_buffer;
  const int64_t size = cuda_buffer->size();
  ARROW_THROW_NOT_OK(AllocateBuffer(arrow::default_memory_pool(), size, &host_buffer));
  ARROW_THROW_NOT_OK(cuda_buffer->CopyToHost(0, size, host_buffer->mutable_data()));

  std::shared_ptr<arrow::RecordBatch> cpu_batch;
  arrow::io::BufferReader cpu_reader(host_buffer);
  ARROW_THROW_NOT_OK(
      arrow::ipc::ReadRecordBatch(read_batch->schema(), &memo, &cpu_reader, &cpu_batch));

  // int column
  auto int_array = cpu_batch->column(0);
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
  auto double_array = cpu_batch->column(1);
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
  auto string_array = cpu_batch->column(2);
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

    mapd::shared_ptr<ThriftClientConnection> conn_mgr;
    conn_mgr = std::make_shared<ThriftClientConnection>();

    auto transport = conn_mgr->open_buffered_client_transport(host, port, cert);
    transport->open();
    auto protocol = std::make_shared<TBinaryProtocol>(transport);
    g_client = std::make_shared<MapDClient>(protocol);

    g_client->connect(g_session_id, user, pwd, db);

    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = -1;
  }
  return err;
}
