/*
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

#include "Shared/sqltypes.h"
#include "StringDictionary/StringDictionary.h"

#include <arrow/api.h>

std::shared_ptr<arrow::DataType> getArrowImportType(const SQLTypeInfo type);

SQLTypeInfo getOmnisciType(const arrow::DataType& type);

std::shared_ptr<arrow::ChunkedArray> replaceNullValues(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type,
    StringDictionary* dict = nullptr);

std::shared_ptr<arrow::ChunkedArray> convertDecimalToInteger(
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type);

std::shared_ptr<arrow::ChunkedArray> createDictionaryEncodedColumn(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type);

std::shared_ptr<arrow::ChunkedArray> convertArrowDictionary(
    StringDictionary* dict,
    std::shared_ptr<arrow::ChunkedArray> arr,
    const SQLTypeInfo& type);
