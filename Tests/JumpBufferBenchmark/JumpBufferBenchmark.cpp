/*
 * Copyright 2025 HEAVY.AI, Inc.
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

#include <iostream>

#include <benchmark/benchmark.h>

#include "CudaMgr/CudaMgr.h"

extern size_t g_jump_buffer_size;
extern size_t g_jump_buffer_parallel_copy_threads;
extern size_t g_jump_buffer_min_h2d_transfer_threshold;
extern size_t g_jump_buffer_min_d2h_transfer_threshold;

constexpr int32_t kTestDeviceId{0};

class GlobalTestEnvironment {
 public:
  GlobalTestEnvironment() {
    CudaMgr_Namespace::CudaMgr cuda_mgr(1);

    size_t gpu_mem = cuda_mgr.getDeviceProperties(kTestDeviceId)->globalMem;
    std::stringstream ss;
    ss << gpu_mem;
    benchmark::AddCustomContext("gpu_mem", ss.str());
    ss.str("");

    ss << cuda_mgr.getDeviceArch();
    benchmark::AddCustomContext("gpu_arch", ss.str());

    constexpr int64_t kOneMb{1024 * 1024};
    constexpr int64_t kOneGb{kOneMb * 1024};
    min_transfer_buffer_size_ = getParam("MIN_TRANSFER_BUFFER_SIZE_MB", 1024) * kOneMb;

    const int64_t default_max_transfer_buffer_size_mb =
        std::max(min_transfer_buffer_size_, 8 * kOneGb) / kOneMb;
    max_transfer_buffer_size_ =
        getParam("MAX_TRANSFER_BUFFER_SIZE_MB", default_max_transfer_buffer_size_mb) *
        kOneMb;
    transfer_buffer_size_multiplier_ = getParam("TRANSFER_BUFFER_SIZE_MULTIPLIER", 2);

    min_jump_buffer_size_ = getParam("MIN_JUMP_BUFFER_SIZE_MB", 128) * kOneMb;

    const int64_t default_max_jump_buffer_size_mb =
        std::max(min_jump_buffer_size_, 2 * kOneGb) / kOneMb;
    max_jump_buffer_size_ =
        getParam("MAX_JUMP_BUFFER_SIZE_MB", default_max_jump_buffer_size_mb) * kOneMb;
    jump_buffer_size_multiplier_ = getParam("JUMP_BUFFER_SIZE_MULTIPLIER", 2);

    min_parallel_copy_threads_ = getParam("MIN_PARALLEL_COPY_THREADS", 4);
    max_parallel_copy_threads_ = getParam("MAX_PARALLEL_COPY_THREADS", 8);
    parallel_copy_threads_multiplier_ = getParam("PARALLEL_COPY_THREADS_MULTIPLIER", 2);

    host_buffer_ = std::vector<int8_t>(max_transfer_buffer_size_);
  }

  int64_t min_transfer_buffer_size_;
  int64_t max_transfer_buffer_size_;
  int64_t transfer_buffer_size_multiplier_;

  int64_t min_jump_buffer_size_;
  int64_t max_jump_buffer_size_;
  int64_t jump_buffer_size_multiplier_;

  int64_t min_parallel_copy_threads_;
  int64_t max_parallel_copy_threads_;
  int64_t parallel_copy_threads_multiplier_;

  std::vector<int8_t> host_buffer_;

 private:
  static int64_t getParam(const std::string& param_name, int64_t default_value) {
    if (const auto env_var = std::getenv(param_name.c_str())) {
      return std::stoll(env_var);
    } else {
      return default_value;
    }
  }
};

GlobalTestEnvironment g_test_env;

class DataTransferBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State& state) override {
    benchmark::Fixture::SetUp(state);
    g_jump_buffer_min_h2d_transfer_threshold = 0;
    g_jump_buffer_min_d2h_transfer_threshold = 0;
    g_jump_buffer_size = state.range(1);
    g_jump_buffer_parallel_copy_threads = state.range(2);
    num_transfer_bytes_ = state.range(0);

    CHECK_LE(int64_t(num_transfer_bytes_), g_test_env.max_transfer_buffer_size_);
    CHECK_LE(int64_t(g_jump_buffer_size), g_test_env.max_transfer_buffer_size_);

    state.counters["transfer_buffer_size"] = num_transfer_bytes_;
    state.counters["jump_buffer_size"] = g_jump_buffer_size;
    state.counters["parallel_copy_threads"] = g_jump_buffer_parallel_copy_threads;

    cuda_mgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(1);
    device_buffer_ = cuda_mgr_->allocateDeviceMem(num_transfer_bytes_, kTestDeviceId);
  }

  void TearDown(benchmark::State& state) override {
    cuda_mgr_->freeDeviceMem(device_buffer_);
  }

  size_t num_transfer_bytes_;
  int8_t* device_buffer_;
  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr_;

  static constexpr CUstream cuda_stream_{0};
  static inline const std::string tag_{"JumpBufferTest"};
};

BENCHMARK_DEFINE_F(DataTransferBenchmark, HostToDevice)(benchmark::State& state) {
  for (auto _ : state) {
    cuda_mgr_->copyHostToDevice(device_buffer_,
                                g_test_env.host_buffer_.data(),
                                num_transfer_bytes_,
                                kTestDeviceId,
                                tag_,
                                cuda_stream_);
  }
}

BENCHMARK_DEFINE_F(DataTransferBenchmark, DeviceToHost)(benchmark::State& state) {
  cuda_mgr_->copyHostToDevice(device_buffer_,
                              g_test_env.host_buffer_.data(),
                              num_transfer_bytes_,
                              kTestDeviceId,
                              tag_,
                              cuda_stream_);

  for (auto _ : state) {
    cuda_mgr_->copyDeviceToHost(g_test_env.host_buffer_.data(),
                                device_buffer_,
                                num_transfer_bytes_,
                                tag_,
                                cuda_stream_);
  }
}

static void arg_generator(benchmark::internal::Benchmark* bench) {
  // Arg format:
  // {transfer_buffer_size, g_jump_buffer_size, g_jump_buffer_parallel_copy_threads}

  size_t arg_count{0};
  CHECK_GT(g_test_env.transfer_buffer_size_multiplier_, 1);
  CHECK_GT(g_test_env.jump_buffer_size_multiplier_, 1);
  CHECK_GT(g_test_env.parallel_copy_threads_multiplier_, 1);

  for (int64_t transfer_buffer_size = g_test_env.min_transfer_buffer_size_;
       transfer_buffer_size <= g_test_env.max_transfer_buffer_size_;
       transfer_buffer_size *= g_test_env.transfer_buffer_size_multiplier_) {
    bench->Args({transfer_buffer_size, 0, 0});
    arg_count++;

    for (int64_t jump_buffer_size = g_test_env.min_jump_buffer_size_;
         jump_buffer_size <=
         std::min(transfer_buffer_size, g_test_env.max_jump_buffer_size_);
         jump_buffer_size *= g_test_env.jump_buffer_size_multiplier_) {
      for (int64_t parallel_copy_threads = g_test_env.min_parallel_copy_threads_;
           parallel_copy_threads <= g_test_env.max_parallel_copy_threads_;
           parallel_copy_threads *= g_test_env.parallel_copy_threads_multiplier_) {
        bench->Args({transfer_buffer_size, jump_buffer_size, parallel_copy_threads});
        arg_count++;
      }
    }
  }

  std::cout << "Running benchmarks with " << arg_count << " arguments"
            << "\nTransfer buffer size range: " << g_test_env.min_transfer_buffer_size_
            << " - " << g_test_env.max_transfer_buffer_size_
            << "\nJump buffer size range: " << g_test_env.min_jump_buffer_size_ << " - "
            << g_test_env.max_jump_buffer_size_ << "\nParallel copy threads count range: "
            << g_test_env.min_parallel_copy_threads_ << " - "
            << g_test_env.max_parallel_copy_threads_ << "\n\n";
}

BENCHMARK_REGISTER_F(DataTransferBenchmark, HostToDevice)->Apply(arg_generator);

BENCHMARK_REGISTER_F(DataTransferBenchmark, DeviceToHost)->Apply(arg_generator);

BENCHMARK_MAIN();
