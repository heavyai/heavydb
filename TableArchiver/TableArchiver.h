/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "Catalog/Catalog.h"
#include "Catalog/SessionInfo.h"

namespace shared {
struct S3Config;
};

class TableArchiver {
 public:
  TableArchiver(Catalog_Namespace::Catalog* cat) : cat_(cat){};

  void dumpTable(const TableDescriptor* td,
                 const std::string& archive_path,
                 const std::string& compression);

  void restoreTable(const Catalog_Namespace::SessionInfo& session,
                    const std::string& table_name,
                    const std::string& archive_path,
                    const std::string& compression,
                    const shared::S3Config& s3_options);

 private:
  void restoreTable(const Catalog_Namespace::SessionInfo& session,
                    const TableDescriptor* td,
                    const std::string& archive_path,
                    const std::string& compression);

  Catalog_Namespace::Catalog* cat_;
};
