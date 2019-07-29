/*
 * Copyright 2019 OmniSci, Inc.
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

#include <Catalog/Catalog.h>
#include <CudaMgr/CudaMgr.h>
#include <QueryEngine/ResultSet.h>
#include <QueryRunner/QueryRunner.h>

#include <gtest/gtest.h>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

extern bool g_allow_cpu_retry;
extern size_t g_max_memory_allocation_size;
extern size_t g_min_memory_allocation_size;
extern bool g_enable_bump_allocator;

namespace {

size_t g_num_gpus{0};
bool g_keep_data{false};

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !QR::get()->gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(ExecutorDeviceType::GPU)) {                 \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
  }

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, true, true);
}

// Note: This assumes a homogenous GPU setup
struct GpuInfo {
  size_t global_memory_size;
  size_t num_gpus;
};

GpuInfo get_gpu_info() {
  auto cat = QR::get()->getCatalog();
  CHECK(cat);

  auto& data_mgr = cat->getDataMgr();
  auto cuda_mgr = data_mgr.getCudaMgr();
  CHECK(cuda_mgr);
  const auto device_props_vec = cuda_mgr->getAllDeviceProperties();
  CHECK_GE(device_props_vec.size(), size_t(1));
  return GpuInfo{device_props_vec.front().globalMem, device_props_vec.size()};
}

bool setup() {
  // Only initialize the QueryRunner once. We will reset the system catalog using the
  // resetWithParameters method as needed for each test.
  QR::init(BASE_PATH, /*udf_filename=*/"", /*max_gpu_mem=*/1000000000);

  if (!QR::get()->gpusPresent()) {
    LOG(WARNING) << "No GPUs detected. Skipping all Bump Allocator tests.";
    return false;
  }

  const auto gpu_info = get_gpu_info();
  g_num_gpus = gpu_info.num_gpus;  // Using the global to pass into tests
  return true;
}

}  // namespace

class LowGpuBufferMemory : public ::testing::Test {
 public:
  void SetUp() override {
    const std::string drop_table_stmt{"DROP TABLE IF EXISTS test;"};
    QR::get()->runDDLStatement(drop_table_stmt);
    const std::string create_table_stmt{
        "CREATE TABLE test (x INT, y BIGINT) WITH (FRAGMENT_SIZE=32);"};
    QR::get()->runDDLStatement(create_table_stmt);

    // Insert enough data to just barely overflow
    for (size_t i = 0; i < 64 * g_num_gpus; i++) {
      const std::string insert_stmt{"INSERT INTO test VALUES (" + std::to_string(i) +
                                    ", " + std::to_string(i) + ");"};
      run_multiple_agg(insert_stmt, ExecutorDeviceType::CPU);
    }

    // Min memory allocation size set to 2GB to guarantee OOM during allocation
    min_mem_allocation_state_ = g_min_memory_allocation_size;
    g_min_memory_allocation_size = 2000000000;

    // CPU retry off to disable automatic punt to CPU
    allow_cpu_retry_state_ = g_allow_cpu_retry;
    g_allow_cpu_retry = false;
  }

  void TearDown() override {
    if (!g_keep_data) {
      const std::string drop_table_stmt{"DROP TABLE IF EXISTS test;"};
      QR::get()->runDDLStatement(drop_table_stmt);
    }

    g_min_memory_allocation_size = min_mem_allocation_state_;
    g_allow_cpu_retry = allow_cpu_retry_state_;
  }

 private:
  size_t min_mem_allocation_state_;
  bool allow_cpu_retry_state_;
};

TEST_F(LowGpuBufferMemory, CPUMode) {
  // Baseline correctness
  auto result_rows =
      run_multiple_agg("SELECT x FROM test WHERE x < 500;", ExecutorDeviceType::CPU);
  ASSERT_EQ(result_rows->rowCount(), size_t(64 * g_num_gpus));
}

TEST_F(LowGpuBufferMemory, OutOfMemory) {
  SKIP_NO_GPU();

  try {
    run_multiple_agg("SELECT x FROM test WHERE x < 500;", ExecutorDeviceType::GPU);
    ASSERT_TRUE(false) << "Expected query to throw exception";
  } catch (const std::exception& e) {
    ASSERT_EQ(
        std::string{"Query ran out of GPU memory, unable to automatically retry on CPU"},
        std::string(e.what()));
  }
}

constexpr size_t row_count_per_gpu = 64;

class LowGpuBufferMemoryCpuRetry : public ::testing::Test {
 public:
  void SetUp() override {
    const std::string drop_table_stmt{"DROP TABLE IF EXISTS test;"};
    QR::get()->runDDLStatement(drop_table_stmt);
    const std::string create_table_stmt{
        "CREATE TABLE test (x INT, y BIGINT) WITH (FRAGMENT_SIZE=32);"};
    QR::get()->runDDLStatement(create_table_stmt);

    // Insert enough data to exceed the max buffer entry guess and force a pre-flight CPU
    // count
    for (size_t i = 0; i < row_count_per_gpu * g_num_gpus; i++) {
      const std::string insert_stmt{"INSERT INTO test VALUES (" + std::to_string(i) +
                                    ", " + std::to_string(i) + ");"};
      run_multiple_agg(insert_stmt, ExecutorDeviceType::CPU);
    }

    // Min memory allocation size set to 2GB to guarantee OOM during allocation
    min_mem_allocation_state_ = g_min_memory_allocation_size;
    g_min_memory_allocation_size = 2000000000;

    // allow CPU retry on
    allow_cpu_retry_state_ = g_allow_cpu_retry;
    g_allow_cpu_retry = true;
  }

  void TearDown() override {
    if (!g_keep_data) {
      const std::string drop_table_stmt{"DROP TABLE IF EXISTS test;"};
      QR::get()->runDDLStatement(drop_table_stmt);
    }

    g_min_memory_allocation_size = min_mem_allocation_state_;
    g_allow_cpu_retry = allow_cpu_retry_state_;
  }

 private:
  size_t min_mem_allocation_state_;
  bool allow_cpu_retry_state_;
};

TEST_F(LowGpuBufferMemoryCpuRetry, OOMRetryOnCPU) {
  SKIP_NO_GPU();

  try {
    auto result_rows =
        run_multiple_agg("SELECT x FROM test WHERE x < 500;", ExecutorDeviceType::GPU);
    ASSERT_EQ(result_rows->rowCount(), size_t(row_count_per_gpu * g_num_gpus));
  } catch (const std::exception& e) {
    ASSERT_TRUE(false) << "Expected query to not throw exception. Query threw: "
                       << e.what();
  }
}

class MediumGpuBufferMemory : public ::testing::Test {
 public:
  void SetUp() override {
    const std::string drop_table_stmt{"DROP TABLE IF EXISTS test;"};
    QR::get()->runDDLStatement(drop_table_stmt);
    const std::string create_table_stmt{
        "CREATE TABLE test (x INT, y BIGINT) WITH (FRAGMENT_SIZE=32);"};
    QR::get()->runDDLStatement(create_table_stmt);

    // Insert enough data to just barely overflow
    for (size_t i = 0; i < 64 * g_num_gpus; i++) {
      const std::string insert_stmt{"INSERT INTO test VALUES (" + std::to_string(i) +
                                    ", " + std::to_string(i) + ");"};
      run_multiple_agg(insert_stmt, ExecutorDeviceType::CPU);
    }

    max_mem_allocation_state_ = g_max_memory_allocation_size;
    g_max_memory_allocation_size = 512;

    // CPU retry off to disable automatic punt to CPU
    allow_cpu_retry_state_ = g_allow_cpu_retry;
    g_allow_cpu_retry = false;
  }

  void TearDown() override {
    if (!g_keep_data) {
      const std::string drop_table_stmt{"DROP TABLE IF EXISTS test;"};
      QR::get()->runDDLStatement(drop_table_stmt);
    }

    g_max_memory_allocation_size = max_mem_allocation_state_;

    g_allow_cpu_retry = allow_cpu_retry_state_;
  }

 private:
  size_t max_mem_allocation_state_;
  bool allow_cpu_retry_state_;
};

TEST_F(MediumGpuBufferMemory, OutOfSlots) {
  SKIP_NO_GPU();

  try {
    run_multiple_agg("SELECT x FROM test WHERE x < 5000;", ExecutorDeviceType::GPU);
    ASSERT_TRUE(false) << "Expected query to throw exception";
  } catch (const std::exception& e) {
    ASSERT_EQ(std::string(e.what()),
              std::string{"Ran out of slots in the query output buffer"});
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");
  desc.add_options()("keep-data",
                     "Don't drop tables at the end of the tests. Note that individual "
                     "tests may still create and drop tables. Use in combination with "
                     "--gtest_filter to preserve tables for a specific test group.");
  desc.add_options()(
      "test-help",
      "Print all BumpAllocatorTest specific options (for gtest options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: BumpAllocatorTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  if (vm.count("keep-data")) {
    g_keep_data = true;
  }

  logger::init(log_options);

  g_enable_bump_allocator = true;

  if (!setup()) {
    // No GPUs detected, bypass the test
    return 0;
  }

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
