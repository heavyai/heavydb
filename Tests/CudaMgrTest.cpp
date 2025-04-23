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

#include <numeric>

#include <gtest/gtest.h>

#include "CudaMgr/CudaMgr.h"
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
