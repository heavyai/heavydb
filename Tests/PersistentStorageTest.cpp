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

#include <boost/filesystem.hpp>
#include "DataMgr/PersistentStorageMgr/PersistentStorageMgr.h"
#include "DataMgrTestHelpers.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

#include "Catalog/Catalog.h"

const std::string data_path = "./tmp/" + shared::kDataDirectoryName;
extern bool g_enable_fsi;

using namespace foreign_storage;
using namespace File_Namespace;
using namespace TestHelpers;

class PersistentStorageMgrTest : public testing::Test {
 protected:
  inline static const std::string cache_path_ = "./test_foreign_data_cache";
  void TearDown() override { boost::filesystem::remove_all(cache_path_); }
};

TEST_F(PersistentStorageMgrTest, DiskCache_CustomPath) {
  PersistentStorageMgr psm(data_path, 0, {cache_path_, DiskCacheLevel::fsi});
  ASSERT_EQ(psm.getDiskCache()->getCacheDirectory(), cache_path_);
}

TEST_F(PersistentStorageMgrTest, DiskCache_InitializeWithoutCache) {
  PersistentStorageMgr psm(data_path, 0, {});
  ASSERT_EQ(psm.getDiskCache(), nullptr);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_fsi = true;
  int err{0};

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
