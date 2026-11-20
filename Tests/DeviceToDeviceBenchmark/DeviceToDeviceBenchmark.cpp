/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "CudaMgr/CudaMgr.h"

namespace {

constexpr int64_t kOneMb{1024 * 1024};

int64_t getParam(const std::string& param_name, int64_t default_value) {
  if (const auto env_var = std::getenv(param_name.c_str())) {
    return std::stoll(env_var);
  }
  return default_value;
}

void checkCuda(CUresult status, const std::string& what) {
  if (status != CUDA_SUCCESS) {
    LOG(FATAL) << what << ": " << CudaMgr_Namespace::error_message(status);
  }
}

class GlobalBenchmarkEnvironment {
 public:
  GlobalBenchmarkEnvironment()
      : min_transfer_buffer_size_{getParam("MIN_TRANSFER_BUFFER_SIZE_MB", 64) * kOneMb}
      , max_transfer_buffer_size_{getParam("MAX_TRANSFER_BUFFER_SIZE_MB", 1024) * kOneMb}
      , transfer_buffer_size_multiplier_{getParam("TRANSFER_BUFFER_SIZE_MULTIPLIER", 2)}
      , host_buffer_(max_transfer_buffer_size_) {
    std::iota(host_buffer_.begin(), host_buffer_.end(), 1);

    cuda_mgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(0);
    device_count_ = cuda_mgr_->getDeviceCount();

    std::stringstream ss;
    ss << device_count_;
    benchmark::AddCustomContext("device_count", ss.str());
    ss.str("");

    ss << cuda_mgr_->getDeviceArch();
    benchmark::AddCustomContext("gpu_arch", ss.str());

    if (device_count_ > 0) {
      ss.str("");
      ss << cuda_mgr_->getDeviceProperties(0)->globalMem;
      benchmark::AddCustomContext("gpu0_mem", ss.str());
      ss.str("");
      ss << cuda_mgr_->getGranularity(0);
      benchmark::AddCustomContext("gpu0_alloc_granularity", ss.str());
    }

    CHECK_GT(min_transfer_buffer_size_, int64_t(0));
    CHECK_GE(max_transfer_buffer_size_, min_transfer_buffer_size_);
    CHECK_GT(transfer_buffer_size_multiplier_, int64_t(1));
  }

  int32_t device_count_{0};
  int64_t min_transfer_buffer_size_;
  int64_t max_transfer_buffer_size_;
  int64_t transfer_buffer_size_multiplier_;
  std::vector<int8_t> host_buffer_;
  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr_;
};

GlobalBenchmarkEnvironment g_benchmark_env;

class DeviceToDeviceBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State& state) override {
    benchmark::Fixture::SetUp(state);

    num_transfer_bytes_ = state.range(0);
    src_device_id_ = state.range(1);
    dest_device_id_ = state.range(2);

    state.counters["transfer_buffer_size"] = num_transfer_bytes_;
    state.counters["src_device"] = src_device_id_;
    state.counters["dest_device"] = dest_device_id_;

    if (g_benchmark_env.device_count_ < 2) {
      state.SkipWithError("device-to-device benchmark requires at least two GPUs");
      return;
    }

    cuda_mgr_ = g_benchmark_env.cuda_mgr_.get();
    if (!cuda_mgr_->canAccessPeer(dest_device_id_, src_device_id_)) {
      state.SkipWithError("CUDA reports no peer access for this GPU pair");
      return;
    }

    try {
      verify_phase_ = "allocating source buffer";
      src_device_buffer_ =
          cuda_mgr_->allocateDeviceMem(num_transfer_bytes_, src_device_id_);
      verify_phase_ = "allocating destination buffer";
      dest_device_buffer_ =
          cuda_mgr_->allocateDeviceMem(num_transfer_bytes_, dest_device_id_);

      verify_phase_ = "copying host buffer to source device";
      cuda_mgr_->copyHostToDevice(src_device_buffer_,
                                  g_benchmark_env.host_buffer_.data(),
                                  num_transfer_bytes_,
                                  src_device_id_,
                                  tag_,
                                  cuda_stream_);
      cuda_mgr_->synchronizeDevices();

      verifyCopy();
    } catch (const std::exception& e) {
      LOG(FATAL) << "Device-to-device benchmark setup failed while " << verify_phase_
                 << ": " << e.what();
    }
  }

  void TearDown(benchmark::State& state) override {
    if (cuda_mgr_) {
      if (src_device_buffer_) {
        cuda_mgr_->freeDeviceMem(src_device_buffer_);
      }
      if (dest_device_buffer_) {
        cuda_mgr_->freeDeviceMem(dest_device_buffer_);
      }
    }
    benchmark::Fixture::TearDown(state);
  }

  void verifyCopy() {
    std::vector<int8_t> host_src(num_transfer_bytes_);
    verify_phase_ = "copying source buffer to host";
    cuda_mgr_->copyDeviceToHost(
        host_src.data(), src_device_buffer_, num_transfer_bytes_, tag_, cuda_stream_);
    const auto src_mismatch = std::mismatch(
        host_src.begin(), host_src.end(), g_benchmark_env.host_buffer_.begin());
    if (src_mismatch.first != host_src.end()) {
      const auto mismatch_index = std::distance(host_src.begin(), src_mismatch.first);
      LOG(FATAL) << "Host-to-device verification failed for source device "
                 << src_device_id_ << " at byte " << mismatch_index << ": expected "
                 << static_cast<int>(*src_mismatch.second) << ", got "
                 << static_cast<int>(*src_mismatch.first);
    }

    verify_phase_ = "copying source buffer to destination device";
    cuda_mgr_->copyDeviceToDevice(dest_device_buffer_,
                                  src_device_buffer_,
                                  num_transfer_bytes_,
                                  dest_device_id_,
                                  src_device_id_,
                                  tag_,
                                  cuda_stream_);
    verify_phase_ = "synchronizing devices";
    cuda_mgr_->synchronizeDevices();
    std::vector<int8_t> host_dest(num_transfer_bytes_);
    verify_phase_ = "copying destination buffer to host";
    cuda_mgr_->copyDeviceToHost(
        host_dest.data(), dest_device_buffer_, num_transfer_bytes_, tag_, cuda_stream_);
    const auto mismatch = std::mismatch(
        host_dest.begin(), host_dest.end(), g_benchmark_env.host_buffer_.begin());
    if (mismatch.first != host_dest.end()) {
      const auto mismatch_index = std::distance(host_dest.begin(), mismatch.first);
      LOG(FATAL) << "Device-to-device verification failed for source device "
                 << src_device_id_ << " and destination device " << dest_device_id_
                 << " at byte " << mismatch_index << ": expected "
                 << static_cast<int>(*mismatch.second) << ", got "
                 << static_cast<int>(*mismatch.first) << ", source pointer "
                 << static_cast<void*>(src_device_buffer_) << ", destination pointer "
                 << static_cast<void*>(dest_device_buffer_);
    }
  }

  int64_t num_transfer_bytes_{0};
  int32_t src_device_id_{0};
  int32_t dest_device_id_{0};
  int8_t* src_device_buffer_{nullptr};
  int8_t* dest_device_buffer_{nullptr};
  CudaMgr_Namespace::CudaMgr* cuda_mgr_{nullptr};
  std::string verify_phase_;

  static constexpr CUstream cuda_stream_{0};
  static inline const std::string tag_{"DeviceToDeviceBenchmark"};
};

class PeerStagingBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State& state) override {
    benchmark::Fixture::SetUp(state);

    num_transfer_bytes_ = state.range(0);
    src_device_id_ = state.range(1);
    dest_device_id_ = state.range(2);

    state.counters["transfer_buffer_size"] = num_transfer_bytes_;
    state.counters["src_device"] = src_device_id_;
    state.counters["dest_device"] = dest_device_id_;

    if (g_benchmark_env.device_count_ < 2) {
      state.SkipWithError("peer staging benchmark requires at least two GPUs");
      return;
    }

    cuda_mgr_ = g_benchmark_env.cuda_mgr_.get();
    if (!cuda_mgr_->canAccessPeer(dest_device_id_, src_device_id_)) {
      state.SkipWithError("CUDA reports no peer access for this GPU pair");
      return;
    }

    try {
      verify_phase_ = "allocating peer-copyable source buffer";
      src_device_buffer_ =
          cuda_mgr_->allocatePeerCopyableDeviceMem(num_transfer_bytes_, src_device_id_);
      verify_phase_ = "allocating peer-copyable destination buffer";
      dest_device_buffer_ =
          cuda_mgr_->allocatePeerCopyableDeviceMem(num_transfer_bytes_, dest_device_id_);

      verify_phase_ = "copying host buffer to peer-copyable source device";
      cuda_mgr_->copyHostToDevice(src_device_buffer_,
                                  g_benchmark_env.host_buffer_.data(),
                                  num_transfer_bytes_,
                                  src_device_id_,
                                  tag_,
                                  cuda_stream_);
      cuda_mgr_->synchronizeDevices();

      verifyCopy();
      verify_phase_ = "creating peer copy stream";
      cuda_mgr_->setContext(dest_device_id_);
      checkCuda(cuStreamCreate(&copy_stream_, CU_STREAM_DEFAULT), verify_phase_);
    } catch (const std::exception& e) {
      LOG(FATAL) << "Peer staging benchmark setup failed while " << verify_phase_ << ": "
                 << e.what();
    }
  }

  void TearDown(benchmark::State& state) override {
    if (cuda_mgr_) {
      if (copy_stream_) {
        cuda_mgr_->setContext(dest_device_id_);
        checkCuda(cuStreamDestroy(copy_stream_), "destroying peer copy stream");
        copy_stream_ = 0;
      }
      if (src_device_buffer_) {
        cuda_mgr_->freePeerCopyableDeviceMem(src_device_buffer_, src_device_id_);
      }
      if (dest_device_buffer_) {
        cuda_mgr_->freePeerCopyableDeviceMem(dest_device_buffer_, dest_device_id_);
      }
    }
    benchmark::Fixture::TearDown(state);
  }

  void verifyCopy() {
    std::vector<int8_t> host_src(num_transfer_bytes_);
    verify_phase_ = "copying peer-copyable source buffer to host";
    cuda_mgr_->copyDeviceToHost(host_src.data(),
                                src_device_buffer_,
                                num_transfer_bytes_,
                                src_device_id_,
                                tag_,
                                cuda_stream_);
    const auto src_mismatch = std::mismatch(
        host_src.begin(), host_src.end(), g_benchmark_env.host_buffer_.begin());
    if (src_mismatch.first != host_src.end()) {
      const auto mismatch_index = std::distance(host_src.begin(), src_mismatch.first);
      LOG(FATAL) << "Peer staging host-to-device verification failed for source device "
                 << src_device_id_ << " at byte " << mismatch_index << ": expected "
                 << static_cast<int>(*src_mismatch.second) << ", got "
                 << static_cast<int>(*src_mismatch.first);
    }

    verify_phase_ = "copying peer-copyable source buffer to destination device";
    cuda_mgr_->copyPeerToPeer(dest_device_buffer_,
                              src_device_buffer_,
                              num_transfer_bytes_,
                              dest_device_id_,
                              src_device_id_,
                              tag_,
                              cuda_stream_);
    verify_phase_ = "synchronizing devices";
    cuda_mgr_->synchronizeDevices();

    std::vector<int8_t> host_dest(num_transfer_bytes_);
    verify_phase_ = "copying peer-copyable destination buffer to host";
    cuda_mgr_->copyDeviceToHost(host_dest.data(),
                                dest_device_buffer_,
                                num_transfer_bytes_,
                                dest_device_id_,
                                tag_,
                                cuda_stream_);
    const auto mismatch = std::mismatch(
        host_dest.begin(), host_dest.end(), g_benchmark_env.host_buffer_.begin());
    if (mismatch.first != host_dest.end()) {
      const auto mismatch_index = std::distance(host_dest.begin(), mismatch.first);
      LOG(FATAL) << "Peer staging device-to-device verification failed for source device "
                 << src_device_id_ << " and destination device " << dest_device_id_
                 << " at byte " << mismatch_index << ": expected "
                 << static_cast<int>(*mismatch.second) << ", got "
                 << static_cast<int>(*mismatch.first) << ", source pointer "
                 << static_cast<void*>(src_device_buffer_) << ", destination pointer "
                 << static_cast<void*>(dest_device_buffer_);
    }
  }

  int64_t num_transfer_bytes_{0};
  int32_t src_device_id_{0};
  int32_t dest_device_id_{0};
  int8_t* src_device_buffer_{nullptr};
  int8_t* dest_device_buffer_{nullptr};
  CudaMgr_Namespace::CudaMgr* cuda_mgr_{nullptr};
  CUstream copy_stream_{0};
  std::string verify_phase_;

  static constexpr CUstream cuda_stream_{0};
  static inline const std::string tag_{"PeerStagingBenchmark"};
};

BENCHMARK_DEFINE_F(DeviceToDeviceBenchmark, CudaMgrCopy)(benchmark::State& state) {
  if (!cuda_mgr_) {
    return;
  }

  for (auto _ : state) {
    cuda_mgr_->copyDeviceToDevice(dest_device_buffer_,
                                  src_device_buffer_,
                                  num_transfer_bytes_,
                                  dest_device_id_,
                                  src_device_id_,
                                  tag_,
                                  cuda_stream_);
  }
  state.SetBytesProcessed(state.iterations() * num_transfer_bytes_);
}

BENCHMARK_DEFINE_F(DeviceToDeviceBenchmark, HostStagedCopy)(benchmark::State& state) {
  if (!cuda_mgr_) {
    return;
  }

  for (auto _ : state) {
    cuda_mgr_->copyDeviceToDeviceViaHost(dest_device_buffer_,
                                         src_device_buffer_,
                                         num_transfer_bytes_,
                                         dest_device_id_,
                                         src_device_id_,
                                         tag_,
                                         cuda_stream_);
  }
  state.SetBytesProcessed(state.iterations() * num_transfer_bytes_);
}

BENCHMARK_DEFINE_F(PeerStagingBenchmark, DirectPeerCopy)(benchmark::State& state) {
  if (!cuda_mgr_) {
    return;
  }

  for (auto _ : state) {
    cuda_mgr_->copyPeerToPeer(dest_device_buffer_,
                              src_device_buffer_,
                              num_transfer_bytes_,
                              dest_device_id_,
                              src_device_id_,
                              tag_,
                              copy_stream_);
  }
  state.SetBytesProcessed(state.iterations() * num_transfer_bytes_);
}

void argGenerator(benchmark::internal::Benchmark* bench) {
  size_t arg_count{0};
  for (int64_t transfer_buffer_size = g_benchmark_env.min_transfer_buffer_size_;
       transfer_buffer_size <= g_benchmark_env.max_transfer_buffer_size_;
       transfer_buffer_size *= g_benchmark_env.transfer_buffer_size_multiplier_) {
    for (int32_t src_device = 0; src_device < g_benchmark_env.device_count_;
         ++src_device) {
      for (int32_t dest_device = 0; dest_device < g_benchmark_env.device_count_;
           ++dest_device) {
        if (src_device == dest_device) {
          continue;
        }
        bench->Args({transfer_buffer_size, src_device, dest_device});
        arg_count++;
      }
    }
  }

  std::cout << "Running device-to-device benchmarks with " << arg_count << " arguments"
            << "\nTransfer buffer size range: "
            << g_benchmark_env.min_transfer_buffer_size_ << " - "
            << g_benchmark_env.max_transfer_buffer_size_
            << "\nTransfer buffer size multiplier: "
            << g_benchmark_env.transfer_buffer_size_multiplier_
            << "\nDevice count: " << g_benchmark_env.device_count_ << "\n\n";
}

BENCHMARK_REGISTER_F(DeviceToDeviceBenchmark, CudaMgrCopy)->Apply(argGenerator);
BENCHMARK_REGISTER_F(DeviceToDeviceBenchmark, HostStagedCopy)->Apply(argGenerator);
BENCHMARK_REGISTER_F(PeerStagingBenchmark, DirectPeerCopy)->Apply(argGenerator);

}  // namespace

BENCHMARK_MAIN();
