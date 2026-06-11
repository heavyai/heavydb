/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    SysDefinitions.h
 * @brief
 *
 */

#pragma once

namespace shared {
inline const std::string kSystemCatalogName = "system_catalog";
inline const std::string kDefaultDbName = "heavyai";
inline const std::string kRootUsername = "admin";
inline const int kRootUserId = 0;
inline const std::string kRootUserIdStr = "0";
inline const std::string kDefaultRootPasswd = "HyperInteractive";
inline const int32_t kTempUserIdRange = 1000000000;
inline const std::string kInfoSchemaDbName = "information_schema";
inline const std::string kInfoSchemaMigrationName = "information_schema_db_created";
inline const std::string kDefaultExportDirName = "export";
inline const std::string kDefaultImportDirName = "import";
inline const std::string kDefaultDiskCacheDirName = "disk_cache";
inline const std::string kDefaultKeyFileName = "heavyai.pem";
inline const std::string kDefaultKeyStoreDirName = "key_store";
inline const std::string kDefaultLogDirName = "log";
inline const std::string kCatalogDirectoryName = "catalogs";
inline const std::string kDataDirectoryName = "data";
inline const std::string kLockfilesDirectoryName = "lockfiles";
inline const std::string kDefaultLicenseFileName = "heavyai.license";
static constexpr size_t kDefaultSampleRowsCount{100};
inline const std::string kDefaultDelimitedServerName = "default_local_delimited";
inline const std::string kDefaultParquetServerName = "default_local_parquet";
inline const std::string kDefaultRegexServerName = "default_local_regex_parsed";
inline const std::string kDefaultRasterServerName = "default_local_raster";
inline const std::string kDeploymentDirectoryName = "deployment";
inline const std::string kDeploymentIdFileName = "id.txt";
#if defined(__aarch64__)
inline const std::string kSystemArchitecture = "aarch64";
#else
inline const std::string kSystemArchitecture = "x86_64";
#endif
}  // namespace shared
