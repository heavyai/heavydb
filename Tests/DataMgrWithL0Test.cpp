/*
 * Copyright 2021 OmniSci, Inc.
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

#include <gtest/gtest.h>

#include "DataMgr/DataMgr.h"
#include "L0Mgr/L0Mgr.h"
#include "TestHelpers.h"

TEST(DataMgrWithL0, SanityTest) {
  std::map<GpuMgrName, std::unique_ptr<GpuMgr>> gpu_mgrs;
  gpu_mgrs[GpuMgrName::L0] = std::make_unique<l0::L0Manager>();
  l0::L0Manager* original_mgr = (l0::L0Manager*)gpu_mgrs[GpuMgrName::L0].get();

  std::string data_mgr_path{"./data_mgr_test_dir"};
  SystemParameters sys_params = {};
  auto data_mgr = std::make_unique<Data_Namespace::DataMgr>(data_mgr_path,
                                                            sys_params,
                                                            std::move(gpu_mgrs),
                                                            /*use_gpus=*/true);

  ASSERT_EQ(original_mgr, data_mgr->getL0Mgr());
  ASSERT_EQ(original_mgr, data_mgr->getGpuMgr());
  ASSERT_EQ(original_mgr, data_mgr->getGpuMgr(GpuMgrName::L0));
  ASSERT_EQ(nullptr, data_mgr->getCudaMgr());
}

TEST(DataMgrWithL0, SimpleReadWriteTest) {
  std::map<GpuMgrName, std::unique_ptr<GpuMgr>> gpu_mgrs;
  gpu_mgrs[GpuMgrName::L0] = std::make_unique<l0::L0Manager>();

  std::string data_mgr_path{"./data_mgr_test_dir"};
  SystemParameters sys_params = {};
  auto data_mgr = std::make_unique<Data_Namespace::DataMgr>(data_mgr_path,
                                                            sys_params,
                                                            std::move(gpu_mgrs),
                                                            /*use_gpus=*/true);

  std::vector<int8_t> ref_data = {10, 20, 30, 40, 50, 60};
  AbstractBuffer* gpuBuff = data_mgr->alloc(
      MemoryLevel::GPU_LEVEL, /*device_id=*/0, /*num_bytes=*/ref_data.size());
  gpuBuff->write(ref_data.data(), ref_data.size(), /*offset=*/0);

  std::vector<int8_t> res_data(ref_data.size());
  gpuBuff->read(res_data.data(), ref_data.size(), /*offset=*/0);

  data_mgr->free(gpuBuff);
  for (size_t i = 0; i < ref_data.size(); i++) {
    ASSERT_EQ(ref_data[i], res_data[i]);
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  return err;
}
