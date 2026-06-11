/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
