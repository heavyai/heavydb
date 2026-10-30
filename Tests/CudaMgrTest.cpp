/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>

#include <gtest/gtest.h>

#include "CudaMgr/CudaMgr.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

extern size_t g_jump_buffer_size;
extern size_t g_jump_buffer_min_h2d_transfer_threshold;
extern size_t g_jump_buffer_min_d2h_transfer_threshold;

class DataTransferTest : public testing::Test {
 protected:
  void SetUp() override {
    g_jump_buffer_min_h2d_transfer_threshold = 0;
    g_jump_buffer_min_d2h_transfer_threshold = 0;

    host_buffer_ = std::vector<int8_t>(num_allocated_bytes_);
    std::iota(host_buffer_.begin(), host_buffer_.end(), 1);
  }

  void TearDown() override { cuda_mgr_->freeDeviceMem(device_buffer_); }

  void copyDataToDeviceAndBackAndAssertExpectedContent() {
    cuda_mgr_ = std::make_unique<CudaMgr_Namespace::CudaMgr>(1);
    device_buffer_ = cuda_mgr_->allocateDeviceMem(num_allocated_bytes_, 0);
    cuda_mgr_->copyHostToDevice(device_buffer_,
                                host_buffer_.data(),
                                num_transfer_bytes_,
                                test_device_id_,
                                "CudaMgrTest",
                                cuda_stream_);

    std::vector<int8_t> smaller_host_buffer(num_transfer_bytes_);
    cuda_mgr_->copyDeviceToHost(smaller_host_buffer.data(),
                                device_buffer_,
                                num_transfer_bytes_,
                                "CudaMgrTest",
                                cuda_stream_);

    EXPECT_EQ(smaller_host_buffer,
              std::vector<int8_t>(host_buffer_.begin(),
                                  host_buffer_.begin() + num_transfer_bytes_));
  }

  std::vector<int8_t> host_buffer_;
  int8_t* device_buffer_;
  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr_;

  static constexpr size_t num_allocated_bytes_{100};
  static constexpr size_t num_transfer_bytes_{10};
  static constexpr CUstream cuda_stream_{0};
  static constexpr int32_t test_device_id_{0};
};

TEST_F(DataTransferTest, WithoutJumpBuffers) {
  g_jump_buffer_size = 0;
  copyDataToDeviceAndBackAndAssertExpectedContent();
}

TEST_F(DataTransferTest, WithJumpBuffers) {
  g_jump_buffer_size = 5;
  copyDataToDeviceAndBackAndAssertExpectedContent();
}

TEST(DeviceTransferTest, DeviceToDeviceCopy) {
  auto cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(0);
  if (cuda_mgr->getDeviceCount() < 2) {
    GTEST_SKIP() << "Device-to-device transfer test requires at least two GPUs";
  }

  constexpr int32_t src_device_id{0};
  constexpr int32_t dest_device_id{1};
  if (!cuda_mgr->canAccessPeer(dest_device_id, src_device_id)) {
    GTEST_SKIP() << "CUDA reports no peer access from device " << src_device_id
                 << " to device " << dest_device_id;
  }

  constexpr size_t num_transfer_bytes{4096};
  std::vector<int8_t> host_src(num_transfer_bytes);
  std::iota(host_src.begin(), host_src.end(), 1);
  std::vector<int8_t> host_dest(num_transfer_bytes);

  auto src_device_buffer = cuda_mgr->allocateDeviceMem(num_transfer_bytes, src_device_id);
  ScopeGuard free_src_device_buffer = [&] { cuda_mgr->freeDeviceMem(src_device_buffer); };
  auto dest_device_buffer =
      cuda_mgr->allocateDeviceMem(num_transfer_bytes, dest_device_id);
  ScopeGuard free_dest_device_buffer = [&] {
    cuda_mgr->freeDeviceMem(dest_device_buffer);
  };

  cuda_mgr->copyHostToDevice(src_device_buffer,
                             host_src.data(),
                             num_transfer_bytes,
                             src_device_id,
                             "CudaMgrTest");
  cuda_mgr->copyDeviceToDevice(dest_device_buffer,
                               src_device_buffer,
                               num_transfer_bytes,
                               dest_device_id,
                               src_device_id,
                               "CudaMgrTest");
  cuda_mgr->copyDeviceToHost(
      host_dest.data(), dest_device_buffer, num_transfer_bytes, "CudaMgrTest");

  EXPECT_EQ(host_dest, host_src);
}

TEST(DeviceTransferTest, PeerCopyableDeviceToDeviceCopy) {
  auto cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(0);
  if (cuda_mgr->getDeviceCount() < 2) {
    GTEST_SKIP() << "Peer-copyable device transfer test requires at least two GPUs";
  }

  constexpr int32_t src_device_id{0};
  constexpr int32_t dest_device_id{1};
  if (!cuda_mgr->canAccessPeer(dest_device_id, src_device_id)) {
    GTEST_SKIP() << "CUDA reports no peer access from device " << src_device_id
                 << " to device " << dest_device_id;
  }

  constexpr size_t num_transfer_bytes{4096};
  std::vector<int8_t> host_src(num_transfer_bytes);
  std::iota(host_src.begin(), host_src.end(), 1);
  std::vector<int8_t> host_dest(num_transfer_bytes);

  auto src_device_buffer =
      cuda_mgr->allocatePeerCopyableDeviceMem(num_transfer_bytes, src_device_id);
  ScopeGuard free_src_device_buffer = [&] {
    cuda_mgr->freePeerCopyableDeviceMem(src_device_buffer, src_device_id);
  };
  auto dest_device_buffer =
      cuda_mgr->allocatePeerCopyableDeviceMem(num_transfer_bytes, dest_device_id);
  ScopeGuard free_dest_device_buffer = [&] {
    cuda_mgr->freePeerCopyableDeviceMem(dest_device_buffer, dest_device_id);
  };

  cuda_mgr->copyHostToDevice(src_device_buffer,
                             host_src.data(),
                             num_transfer_bytes,
                             src_device_id,
                             "CudaMgrTest");
  cuda_mgr->copyPeerToPeer(dest_device_buffer,
                           src_device_buffer,
                           num_transfer_bytes,
                           dest_device_id,
                           src_device_id,
                           "CudaMgrTest");
  cuda_mgr->copyDeviceToHost(host_dest.data(),
                             dest_device_buffer,
                             num_transfer_bytes,
                             dest_device_id,
                             "CudaMgrTest");

  EXPECT_EQ(host_dest, host_src);
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
