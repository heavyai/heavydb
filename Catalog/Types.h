/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

#include "Catalog/ColumnDescriptor.h"
#include "Catalog/CustomExpression.h"
#include "Catalog/DashboardDescriptor.h"
#include "Catalog/DictDescriptor.h"
#include "Catalog/ForeignServer.h"
#include "Catalog/LinkDescriptor.h"
#include "Catalog/TableDescriptor.h"

namespace Catalog_Namespace {

using TableDescriptorMap = std::map<std::string, TableDescriptor*>;
using TableDescriptorMapById = std::map<int, TableDescriptor*>;
using LogicalToPhysicalTableMapById = std::map<int32_t, std::vector<int32_t>>;
using ColumnKey = std::tuple<int, std::string>;
using ColumnDescriptorMap = std::map<ColumnKey, ColumnDescriptor*>;
using ColumnIdKey = std::tuple<int, int>;
using ColumnDescriptorMapById = std::map<ColumnIdKey, ColumnDescriptor*>;
using TableDictColumnsMap = std::map<int32_t, std::set<const ColumnDescriptor*>>;
using DictDescriptorMapById = std::map<DictRef, std::unique_ptr<DictDescriptor>>;
using DashboardDescriptorMap =
    std::map<std::string, std::shared_ptr<DashboardDescriptor>>;
using LinkDescriptorMap = std::map<std::string, LinkDescriptor*>;
using LinkDescriptorMapById = std::map<int, LinkDescriptor*>;
using DeletedColumnPerTableMap =
    std::unordered_map<const TableDescriptor*, const ColumnDescriptor*>;
using ForeignServerMap =
    std::map<std::string, std::shared_ptr<foreign_storage::ForeignServer>>;
using ForeignServerMapById =
    std::map<int, std::shared_ptr<foreign_storage::ForeignServer>>;
using CustomExpressionMapById = std::map<int, std::unique_ptr<CustomExpression>>;
}  // namespace Catalog_Namespace
