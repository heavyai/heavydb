/*
 * Copyright 2020 OmniSci, Inc.
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

#pragma once

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

#include "Catalog/ColumnDescriptor.h"
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

}  // namespace Catalog_Namespace
