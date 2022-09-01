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

#include "SchemaJson.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

namespace {

int toCalciteTypeId(const hdk::ir::Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kNull:
      return kNULLT;
    case hdk::ir::Type::kBoolean:
      return kBOOLEAN;
    case hdk::ir::Type::kInteger:
      switch (type->size()) {
        case 1:
          return kTINYINT;
        case 2:
          return kSMALLINT;
        case 4:
          return kINT;
        case 8:
          return kBIGINT;
        default:
          break;
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          return kFLOAT;
        case hdk::ir::FloatingPointType::kDouble:
          return kDOUBLE;
        default:
          break;
      }
      break;
    case hdk::ir::Type::kDecimal:
      return kDECIMAL;
    case hdk::ir::Type::kVarChar:
      return kVARCHAR;
    case hdk::ir::Type::kText:
      return kTEXT;
    case hdk::ir::Type::kDate:
      return kDATE;
    case hdk::ir::Type::kTime:
      return kTIME;
    case hdk::ir::Type::kTimestamp:
      return kTIMESTAMP;
    case hdk::ir::Type::kInterval:
      switch (type->as<hdk::ir::IntervalType>()->unit()) {
        case hdk::ir::TimeUnit::kMonth:
          return kINTERVAL_YEAR_MONTH;
        case hdk::ir::TimeUnit::kMilli:
          return kINTERVAL_DAY_TIME;
        default:
          break;
      }
      break;
    case hdk::ir::Type::kFixedLenArray:
    case hdk::ir::Type::kVarLenArray:
      return kARRAY;
    case hdk::ir::Type::kExtDictionary:
      if (type->as<hdk::ir::ExtDictionaryType>()->elemType()->isString()) {
        return kTEXT;
      }
      break;
    case hdk::ir::Type::kColumn:
      return kCOLUMN;
    case hdk::ir::Type::kColumnList:
      return kCOLUMN_LIST;
    default:
      break;
  }
  throw std::runtime_error("Cannot map to Calcite type system: " + type->toString());
}

int toCalciteSubtypeId(const hdk::ir::Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kFixedLenArray:
    case hdk::ir::Type::kVarLenArray:
      return toCalciteTypeId(type->as<hdk::ir::ArrayBaseType>()->elemType());
    case hdk::ir::Type::kColumn:
      return toCalciteTypeId(type->as<hdk::ir::ColumnType>()->columnType());
    case hdk::ir::Type::kColumnList:
      return toCalciteTypeId(type->as<hdk::ir::ColumnListType>()->columnType());
    default:
      return kNULLT;
  }
}

int getCalciteDimension(const hdk::ir::Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kDecimal:
      return type->as<hdk::ir::DecimalType>()->precision();
    case hdk::ir::Type::kVarChar:
      return type->as<hdk::ir::VarCharType>()->maxLength();
    case hdk::ir::Type::kTimestamp:
      switch (type->as<hdk::ir::TimestampType>()->unit()) {
        case hdk::ir::TimeUnit::kSecond:
          return 0;
        case hdk::ir::TimeUnit::kMilli:
          return 3;
        case hdk::ir::TimeUnit::kMicro:
          return 6;
        case hdk::ir::TimeUnit::kNano:
          return 9;
        default:
          break;
      }
      break;
    case hdk::ir::Type::kFixedLenArray:
    case hdk::ir::Type::kVarLenArray:
      return getCalciteDimension(type->as<hdk::ir::ArrayBaseType>()->elemType());
    case hdk::ir::Type::kExtDictionary:
      return getCalciteDimension(type->as<hdk::ir::ExtDictionaryType>()->elemType());
    case hdk::ir::Type::kColumnList:
      return type->as<hdk::ir::ColumnListType>()->length();
    default:
      return 0;
  }
  throw std::runtime_error("Cannot map to Calcite type system: " + type->toString());
}

int getCalciteScale(const hdk::ir::Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kDecimal:
      return type->as<hdk::ir::DecimalType>()->scale();
    case hdk::ir::Type::kFixedLenArray:
    case hdk::ir::Type::kVarLenArray:
      return getCalciteScale(type->as<hdk::ir::ArrayBaseType>()->elemType());
    default:
      return 0;
  }
  throw std::runtime_error("Cannot map to Calcite type system: " + type->toString());
}

}  // namespace

std::string schema_to_json(SchemaProviderPtr schema_provider) {
  auto dbs = schema_provider->listDatabases();
  if (dbs.empty()) {
    return "{}";
  }
  // Current JSON format supports a single database only.
  CHECK_EQ(dbs.size(), (size_t)1);
  auto tables = schema_provider->listTables(dbs.front());

  rapidjson::Document doc(rapidjson::kObjectType);

  for (auto tinfo : tables) {
    rapidjson::Value table(rapidjson::kObjectType);
    table.AddMember("name",
                    rapidjson::Value().SetString(rapidjson::StringRef(tinfo->name)),
                    doc.GetAllocator());
    table.AddMember("id", rapidjson::Value().SetInt(tinfo->table_id), doc.GetAllocator());
    table.AddMember(
        "columns", rapidjson::Value(rapidjson::kArrayType), doc.GetAllocator());

    auto columns = schema_provider->listColumns(*tinfo);
    for (const auto& col_info : columns) {
      rapidjson::Value column(rapidjson::kObjectType);
      column.AddMember("name",
                       rapidjson::Value().SetString(rapidjson::StringRef(col_info->name)),
                       doc.GetAllocator());
      column.AddMember("coltype",
                       rapidjson::Value().SetInt(toCalciteTypeId(col_info->type)),
                       doc.GetAllocator());
      column.AddMember("colsubtype",
                       rapidjson::Value().SetInt(toCalciteSubtypeId(col_info->type)),
                       doc.GetAllocator());
      column.AddMember("coldim",
                       rapidjson::Value().SetInt(getCalciteDimension(col_info->type)),
                       doc.GetAllocator());
      column.AddMember("colscale",
                       rapidjson::Value().SetInt(getCalciteScale(col_info->type)),
                       doc.GetAllocator());
      column.AddMember("is_notnull",
                       rapidjson::Value().SetBool(!col_info->type->nullable()),
                       doc.GetAllocator());
      column.AddMember("is_systemcol",
                       rapidjson::Value().SetBool(col_info->is_rowid),
                       doc.GetAllocator());
      column.AddMember("is_virtualcol",
                       rapidjson::Value().SetBool(col_info->is_rowid),
                       doc.GetAllocator());
      column.AddMember(
          "is_deletedcol", rapidjson::Value().SetBool(false), doc.GetAllocator());
      table["columns"].PushBack(column, doc.GetAllocator());
    }
    doc.AddMember(rapidjson::StringRef(tinfo->name), table, doc.GetAllocator());
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  return std::string(buffer.GetString());
}
