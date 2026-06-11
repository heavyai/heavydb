/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TableArchiver/TableArchiver.h"
__declspec(dllexport) bool g_test_rollback_dump_restore{false};

void TableArchiver::dumpTable(const TableDescriptor* td,
                              const std::string& archive_path,
                              const std::string& compression) {
  throw std::runtime_error("Dump/restore table not yet supported on Windows.");
}

void TableArchiver::restoreTable(const Catalog_Namespace::SessionInfo& session,
                                 const TableDescriptor* td,
                                 const std::string& archive_path,
                                 const std::string& compression) {
  throw std::runtime_error("Dump/restore table not yet supported on Windows.");
}

void TableArchiver::restoreTable(const Catalog_Namespace::SessionInfo& session,
                                 const std::string& table_name,
                                 const std::string& archive_path,
                                 const std::string& compression,
                                 const TableArchiverS3Options& s3_options) {
  throw std::runtime_error("Dump/restore table not yet supported on Windows.");
}
