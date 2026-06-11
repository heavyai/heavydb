/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <arrow/api.h>
#include <memory>
#include <string>

#include "ForeignStorageInterface.h"

void registerArrowCsvForeignStorage(std::shared_ptr<ForeignStorageInterface> fsi);

void registerArrowForeignStorage(std::shared_ptr<ForeignStorageInterface> fsi);

void setArrowTable(std::string name, std::shared_ptr<arrow::Table> table);

void releaseArrowTable(std::string name);
