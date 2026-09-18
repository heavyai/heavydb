/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <list>
#include "Catalog/CatalogFwd.h"
#include "Catalog/UserMapping.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "Shared/sqltypes.h"

namespace foreign_storage {

class OdbcDriversConfigException : public ForeignStorageException {
 public:
  OdbcDriversConfigException(const std::string& error_message)
      : ForeignStorageException(error_message) {}
};

struct RemoteData {
  int8_t* data_ptr;
  std::vector<std::optional<size_t>> null_or_strlen;

  // Note: odbc_base_type holds the integral value corresponding to
  // the underlying odbc type.  In the ODBC API this is defined
  // as a SQLSMALLINT. As this level of abstraction doesn't have
  // access to the ODBC macros its defined as a int16_t
  int16_t odbc_base_type;
  int32_t odbc_octet_transfer_size;
  bool is_unsigned;
};

using ResultSetProcessor = std::function<
    void(size_t num_rows, size_t data_size, size_t row_offset, RemoteData& remote_data)>;

// struct to describe remote db columns
// in terms of omnisci sql types
// Means the ODBC sql types are not
// exposed out of odbc_utils.cpp

struct RemoteColumnDescription {
  SQLTypes omnisci_type;
  int32_t decimal_digits;
  int32_t odbc_octet_transfer_size;
  std::string omnisci_type_name;
  std::string column_name;
  int omnisci_column_id;
  bool is_unsigned;

  // To support type coercion, it's
  // necessary  to record the base
  // odbc type.
  int32_t odbc_base_type;

 public:
  bool isMappingAllowed(const SQLTypeInfo& omnisci_typeinfo) const;
  int32_t getDecimalPrecision() const;
};

struct ColumnRemoteData {
  RemoteColumnDescription remote_column_description;
  std::unique_ptr<int8_t[]> data_ptr;
  std::vector<std::optional<size_t>> null_or_strlen;
  size_t row_count;
};

int64_t temporal_conversion_utility(int8_t const* data_buffer_ptr,
                                    const SQLTypeInfo& sql_type_info);

std::string temporal_to_string(int8_t const* data_buffer_ptr,
                               const SQLTypeInfo& sql_type_info);

struct OdbcConnectionInfo {
  std::optional<std::string> data_source_name;
  std::optional<std::string> connection_string;
};

struct OdbcSelectDescriptor {
  OdbcSelectDescriptor(int64_t offset,
                       size_t limit,
                       std::string select_command,
                       std::string order_by_command,
                       const ColumnDescriptor& column,
                       const RemoteColumnDescription& remote_column)
      : offset(offset)
      , limit(limit)
      , select_command(select_command)
      , order_by_command(order_by_command)
      , column(column)
      , remote_column(remote_column) {}

  OdbcSelectDescriptor(std::string select_command,
                       std::string order_by_command,
                       const ColumnDescriptor& column,
                       const RemoteColumnDescription& remote_column)
      : offset(0)
      , limit(std::numeric_limits<size_t>::max())
      , select_command(select_command)
      , order_by_command(order_by_command)
      , column(column)
      , remote_column(remote_column) {}

  enum class SelectQueryType {
    kSelect = 0,
    kMaxLength,
    kCountNotNull,
    kMinValue,
    kMaxValue,
  };

  // Uses the optional `select_type` member to get the select statement. This
  // is used in cases where the parameters of the `OdbcSelectDescriptor` change
  // between when this object is instantiated and when it is used to obtain the
  // select statement.
  std::string getSelectStmtFromType() const;

  std::string getSelectStmt() const;
  std::string getSelectMaxLengthStmt() const;
  std::string getCountNotNullStmt() const;
  std::string getMinValueStmt() const;
  std::string getMaxValueStmt() const;

  size_t getExpectedNumResultRows() const;

  int64_t offset;
  size_t limit;
  std::string select_command;
  std::string order_by_command;
  const ColumnDescriptor& column;
  const RemoteColumnDescription& remote_column;
  std::optional<SelectQueryType> select_type;

 private:
  // Returned query is not terminated by semi-colon
  std::string getSelectStmtSubquery() const;
};

class OdbcConnection {
 public:
  static std::unique_ptr<OdbcConnection> create(
      const OdbcConnectionInfo& connection_info,
      const UserMapping* user_mapping = nullptr);
  virtual size_t getRecordCnt(const std::string& sql_select) = 0;
  virtual void getRemoteColumnDetailsValidateRemoteMetaData(
      std::string sql_command,
      std::list<ColumnDescriptor const*>& catalog_sqltypes,
      std::vector<foreign_storage::RemoteColumnDescription>& remote_column_details) = 0;

  virtual ~OdbcConnection() = 0;

  virtual void runSelectCmd(const OdbcSelectDescriptor& select_desc,
                            size_t buffer_byte_size,
                            ResultSetProcessor) = 0;
  virtual std::vector<ColumnRemoteData> runDataPreviewMultiColumnSelectCmd(
      const std::string& select_command,
      size_t row_count) = 0;
  virtual void runSqlAllowSuccessWithInfo(const std::string& sql) = 0;
  virtual void runSql(const std::string& sql) = 0;

  virtual size_t getChunkNullCount(const OdbcSelectDescriptor& select_desc) = 0;

  static std::set<std::string> getInstalledDrivers();

  static size_t max_buffer_resize_limit_;
};

bool is_decimal_odbc_type(int16_t sqltype);
}  // namespace foreign_storage
