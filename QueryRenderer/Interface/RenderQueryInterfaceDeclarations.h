/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

namespace Catalog_Namespace {
class Catalog;
}

namespace Analyzer {
class TargetEntry;
}  // namespace Analyzer

struct TableDescriptor;

class TargetMetaInfo;
class ResultSet;

namespace QueryRenderer {

using TargetEntries = std::vector<std::shared_ptr<Analyzer::TargetEntry>>;

}  // namespace QueryRenderer
