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

#include "TestHelpers.h"

#include <future>
#include <string>
#include <vector>

#include "DBHandlerTestHelpers.h"
#include "Logger/Logger.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/scope.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

bool g_keep_data{false};
size_t g_max_num_executors{4};

extern bool g_enable_executor_resource_mgr;

using namespace TestHelpers;

#define SKIP_NO_GPU()                                        \
  if (skipTests(TExecuteMode::type::GPU)) {                  \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    return;                                                  \
  }

using ResourceVec =
    std::vector<std::pair<ExecutorResourceMgr_Namespace::ResourceType, size_t>>;

using ConcurrentResourceGrantPolicyVec =
    std::vector<ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy>;

class ConcurrencyTestEnv : public DBHandlerTestFixture {
 protected:
  bool skipTests(const TExecuteMode::type device_type) {
#ifdef HAVE_CUDA
    return device_type == TExecuteMode::type::GPU &&
           !(getCatalog().getDataMgr().gpusPresent());
#else
    return device_type == TExecuteMode::type::GPU;
#endif
  }

  void buildTable(const std::string& table_name) {
    setExecuteMode(TExecuteMode::type::CPU);
    sql("DROP TABLE IF EXISTS " + table_name + ";");
    sql("CREATE TABLE " + table_name + " " + table_schema);
    sql("INSERT INTO " + table_name +
        " VALUES "
        "(0, 0.1, 'hi'), "
        "(1, 1.1, 'hi'), "
        "(2, 2.1, 'hi'), "
        "(3, 3.1, 'hi'), "
        "(4, 4.1, 'hi'), "
        "(5, 5.1, 'hi'), "
        "(6, 6.1, 'hi'), "
        "(7, 7.1, 'hi'), "
        "(8, 8.1, 'hi'), "
        "(9, 9.1, 'hi'), "
        "(10, 10.1, 'hello'), "
        "(11, 11.1, 'hello'), "
        "(12, 12.1, 'hello'), "
        "(13, 13.1, 'hello'), "
        "(14, 14.1, 'hello'), "
        "(15, 15.1, 'hello'), "
        "(16, 16.1, 'hello'), "
        "(17, 17.1, 'hello'), "
        "(18, 18.1, 'hello'), "
        "(19, 19.1, 'hello'), "
        "(20, 20.1, 'hey'), "
        "(21, 21.1, 'hey'), "
        "(22, 22.1, 'hey'), "
        "(23, 23.1, 'hey'), "
        "(24, 24.1, 'hey'), "
        "(25, 25.1, 'hey'), "
        "(26, 26.1, 'hey'), "
        "(27, 27.1, 'hey'), "
        "(28, 28.1, 'hey'), "
        "(29, 29.1, 'hey'), "
        "(30, 30.1, 'howdy'), "
        "(31, 31.1, 'howdy');");
  }

  static int64_t get_query_return_time_epoch_ms(const std::string& query) {
    sql(query);
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  void SetUp() override {
    buildTable("test_concurrency");
  }

  void TearDown() override {
    if (!g_keep_data) {
      sql("DROP TABLE IF EXISTS " + table_name + ";");
    }
  }

  void runSimpleConcurrencyTest(
      const TExecuteMode::type execution_mode_slow_query,
      const TExecuteMode::type execution_mode_fast_query,
      const size_t num_executors,
      const ResourceVec& resource_overrides,
      const ConcurrentResourceGrantPolicyVec& concurrent_resource_grant_policy_overrides,
      const bool fast_query_should_finish_first) {
    ResourceVec initial_resources;
    for (const auto& resource_pair : resource_overrides) {
      initial_resources.emplace_back(
          resource_pair.first,
          Executor::get_executor_resource_pool_total_resource_quantity(
              resource_pair.first));
    }

    ConcurrentResourceGrantPolicyVec initial_concurrent_resource_grant_policies;
    for (const auto& concurrent_resource_grant_policy :
         concurrent_resource_grant_policy_overrides) {
      initial_concurrent_resource_grant_policies.emplace_back(
          Executor::get_concurrent_resource_grant_policy(
              concurrent_resource_grant_policy.resource_type));
    };

    ScopeGuard reset_resource_mgr_state = [&initial_resources,
                                           &initial_concurrent_resource_grant_policies] {
      for (const auto& resource_pair : initial_resources) {
        Executor::set_executor_resource_pool_resource(resource_pair.first,
                                                      resource_pair.second);
      }
      for (const auto& concurrent_resource_grant_policy :
           initial_concurrent_resource_grant_policies) {
        Executor::set_concurrent_resource_grant_policy(concurrent_resource_grant_policy);
      }
    };

    for (const auto& resource_pair : resource_overrides) {
      Executor::set_executor_resource_pool_resource(resource_pair.first,
                                                    resource_pair.second);
    }

    for (const auto& resource_pair : resource_overrides) {
      ASSERT_EQ(Executor::get_executor_resource_pool_total_resource_quantity(
                    resource_pair.first),
                resource_pair.second);
    }

    for (const auto& concurrent_resource_grant_policy :
         concurrent_resource_grant_policy_overrides) {
      Executor::set_concurrent_resource_grant_policy(concurrent_resource_grant_policy);
    }

    for (const auto& concurrent_resource_grant_policy :
         concurrent_resource_grant_policy_overrides) {
      ASSERT_TRUE(Executor::get_concurrent_resource_grant_policy(
                      concurrent_resource_grant_policy.resource_type) ==
                  concurrent_resource_grant_policy);
    }

    resizeDispatchQueue(num_executors);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Run the queries once to control for warmup/compilation/data fetch time
    setExecuteMode(execution_mode_slow_query);
    auto slow_query_ms_warmup_future = std::async(
        std::launch::async, get_query_return_time_epoch_ms, std::ref(slow_query_sql));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    setExecuteMode(execution_mode_fast_query);
    auto fast_query_ms_warmup_future = std::async(
        std::launch::async, get_query_return_time_epoch_ms, std::ref(fast_query_sql));
    fast_query_ms_warmup_future.get();
    slow_query_ms_warmup_future.get();

    setExecuteMode(execution_mode_slow_query);

    auto slow_query_ms_future = std::async(
        std::launch::async, get_query_return_time_epoch_ms, std::ref(slow_query_sql));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    setExecuteMode(execution_mode_fast_query);

    auto fast_query_ms_future = std::async(
        std::launch::async, get_query_return_time_epoch_ms, std::ref(fast_query_sql));

    const auto fast_query_return_ms = fast_query_ms_future.get();
    const auto slow_query_return_ms = slow_query_ms_future.get();

    if (fast_query_should_finish_first) {
      EXPECT_LT(fast_query_return_ms, slow_query_return_ms);
    } else {
      EXPECT_GE(fast_query_return_ms, slow_query_return_ms);
    }
  }

 private:
  const std::string table_name{"test_concurrency"};
  const std::string slow_query_sql =
      "SELECT SUM(ct_sleep_us(100000)) AS res FROM test_concurrency WHERE i64 < 50;";
  const std::string fast_query_sql =
      "SELECT SUM(1) AS res FROM test_concurrency WHERE i64 < 50;";
  const char* table_schema = R"(
    (
        i64 BIGINT,
        d DOUBLE,
        str TEXT ENCODING DICT
    ) WITH (FRAGMENT_SIZE=16);
  )";
};

#ifdef HAVE_ASAN
// Fix this in QE-726
TEST_F(ConcurrencyTestEnv, DISABLED_CPUConcurrencyOff) {
#else
TEST_F(ConcurrencyTestEnv, CPUConcurrencyOff) {
#endif
  const size_t num_executors{g_max_num_executors};
  constexpr size_t num_frags{2};
  const ResourceVec resource_overrides = {
      {ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS, num_frags * 4}};
  ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy
      concurrent_resource_grant_policy(
          ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST);
  const ConcurrentResourceGrantPolicyVec concurrent_resource_grant_policies = {
      concurrent_resource_grant_policy};

  const auto execution_mode{TExecuteMode::type::CPU};
  const bool fast_query_should_finish_first{false};

  runSimpleConcurrencyTest(execution_mode,  // For slow query
                           execution_mode,  // For fast query
                           num_executors,
                           resource_overrides,
                           concurrent_resource_grant_policies,
                           fast_query_should_finish_first);
}

TEST_F(ConcurrencyTestEnv, CPUConcurrencyOn) {
  const size_t num_executors{g_max_num_executors};
  constexpr size_t num_frags{2};
  const ResourceVec resource_overrides = {
      {ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS, num_frags * 4}};
  ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy
      concurrent_resource_grant_policy(
          ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::
              ALLOW_CONCURRENT_REQUESTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST);
  const ConcurrentResourceGrantPolicyVec concurrent_resource_grant_policies = {
      concurrent_resource_grant_policy};
  const auto execution_mode{TExecuteMode::type::CPU};
  const bool fast_query_should_finish_first{true};

  runSimpleConcurrencyTest(execution_mode,  // For slow query
                           execution_mode,  // For fast query
                           num_executors,
                           resource_overrides,
                           concurrent_resource_grant_policies,
                           fast_query_should_finish_first);
}

TEST_F(ConcurrencyTestEnv, CPUGPUConcurrencyOff) {
  SKIP_NO_GPU();
  const size_t num_executors{g_max_num_executors};
  constexpr size_t num_frags{2};
  const ResourceVec resource_overrides = {
      {ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS, num_frags * 4}};
  ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy
      cpu_concurrent_resource_grant_policy(
          ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::
              ALLOW_CONCURRENT_REQUESTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST);
  ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy
      gpu_concurrent_resource_grant_policy(
          ExecutorResourceMgr_Namespace::ResourceType::GPU_SLOTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::
              ALLOW_SINGLE_REQUEST_GLOBALLY,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::
              ALLOW_SINGLE_REQUEST_GLOBALLY);

  const ConcurrentResourceGrantPolicyVec concurrent_resource_grant_policies = {
      cpu_concurrent_resource_grant_policy, gpu_concurrent_resource_grant_policy};

  const auto execution_mode_slow_query{TExecuteMode::type::CPU};
  const auto execution_mode_fast_query{TExecuteMode::type::GPU};
  const bool fast_query_should_finish_first{false};

  runSimpleConcurrencyTest(execution_mode_slow_query,
                           execution_mode_fast_query,
                           num_executors,
                           resource_overrides,
                           concurrent_resource_grant_policies,
                           fast_query_should_finish_first);
}

TEST_F(ConcurrencyTestEnv, CPUGPUConcurrencyOn) {
  SKIP_NO_GPU();
  const size_t num_executors{g_max_num_executors};
  constexpr size_t num_frags{2};
  const ResourceVec resource_overrides = {
      {ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS, num_frags * 4}};
  ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy
      cpu_concurrent_resource_grant_policy(
          ExecutorResourceMgr_Namespace::ResourceType::CPU_SLOTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::
              ALLOW_CONCURRENT_REQUESTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST);
  ExecutorResourceMgr_Namespace::ConcurrentResourceGrantPolicy
      gpu_concurrent_resource_grant_policy(
          ExecutorResourceMgr_Namespace::ResourceType::GPU_SLOTS,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST,
          ExecutorResourceMgr_Namespace::ResourceConcurrencyPolicy::ALLOW_SINGLE_REQUEST);

  const ConcurrentResourceGrantPolicyVec concurrent_resource_grant_policies = {
      cpu_concurrent_resource_grant_policy, gpu_concurrent_resource_grant_policy};

  const auto execution_mode_slow_query{TExecuteMode::type::CPU};
  const auto execution_mode_fast_query{TExecuteMode::type::GPU};
  const bool fast_query_should_finish_first{true};

  runSimpleConcurrencyTest(execution_mode_slow_query,
                           execution_mode_fast_query,
                           num_executors,
                           resource_overrides,
                           concurrent_resource_grant_policies,
                           fast_query_should_finish_first);
}

int main(int argc, char* argv[]) {
  g_enable_executor_resource_mgr = true;
  g_executor_resource_mgr_allow_cpu_kernel_concurrency = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()(
      "num-executors",
      po::value<size_t>(&g_max_num_executors)->default_value(g_max_num_executors),
      "Maximum number of parallel executors to test.");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
