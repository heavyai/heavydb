/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "odbc_utils.h"

#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <odbcinst.h>
#include <sql.h>
#include <sqlext.h>
#include <boost/algorithm/string.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/range/adaptor/indexed.hpp>
// Note this sqltypes is from the odbc include directory
// and should not be mixed up with the omnisci shared/sqltypes.h
#include <sqltypes.h>

#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "ImportExport/CopyParams.h"
#include "OdbcDataWrapper.h"
#include "Shared/DateTimeParser.h"
#include "Shared/SysDefinitions.h"

namespace {
bool has_variable_length_input(const SQLTypeInfo type_info) {
  // SQLTypeInfo is a heavy.ai data structure.  This
  // function reports on a column in a heavy table.
  return type_info.is_string() || type_info.is_geometry();
}

std::ostream& operator<<(std::ostream& os, const SQL_DATE_STRUCT& date_struct) {
  os << "Year = [" << date_struct.year << "], month = [" << date_struct.month
     << "], day [" << date_struct.day << "]";
  return os;
}
std::ostream& operator<<(std::ostream& os, const SQL_TIME_STRUCT& time_struct) {
  os << "Hour = [" << time_struct.hour << "], minute = [" << time_struct.minute
     << "], second [" << time_struct.second << "]";
  return os;
}
std::ostream& operator<<(std::ostream& os, const SQL_TIMESTAMP_STRUCT& timestamp_struct) {
  os << "Year = [" << timestamp_struct.year << "], month = [" << timestamp_struct.month
     << "], day [" << timestamp_struct.day << "], hour = [" << timestamp_struct.hour
     << "], minute = [" << timestamp_struct.minute << "], second ["
     << timestamp_struct.second << "], sub seconds [" << timestamp_struct.fraction << "]";
  return os;
}

template <class TEMPORAL_TYPE>
class TemporalConverter {
 public:
  static_assert(
      std::is_same<TEMPORAL_TYPE, SQL_DATE_STRUCT>::value ||
          std::is_same<TEMPORAL_TYPE, SQL_TIME_STRUCT>::value ||
          std::is_same<TEMPORAL_TYPE, SQL_TIMESTAMP_STRUCT>::value,
      "Only SQL_DATE_STRUCT | SQL_TIME_STRUCT |SQL_TIMESTAMP_STRUCT may be used "
      "for this template");

  TemporalConverter(int8_t const* raw_sql_tm, int precision = 0) : precision_(precision) {
    auto* sql_tm = reinterpret_cast<TEMPORAL_TYPE const*>(raw_sql_tm);

    setDateTimeStruct(sql_tm);
    if (validateDateTime()) {
      std::stringstream e_stream;
      e_stream << "Time conversion error for input structure " << *sql_tm
               << ". Internal structure " << date_time_;
      throw foreign_storage::ForeignStorageException(e_stream.str());
    }

    epoch_ = date_time_.getTime(precision_);
  }

  int64_t getEpoch() { return epoch_; }

  std::string toString(SQLTypes type) {
    std::stringstream ss;
    if (type == kTIME) {
      ss << std::setw(2) << std::setfill('0') << date_time_.H << ":" << std::setw(2)
         << std::setfill('0') << date_time_.M << ":" << std::setw(2) << std::setfill('0')
         << date_time_.S;
    } else if (type == kDATE) {
      ss << date_time_.Y << "-" << std::setw(2) << std::setfill('0') << date_time_.m
         << "-" << std::setw(2) << std::setfill('0') << date_time_.d;
    } else if (type == kTIMESTAMP) {
      ss << date_time_.Y << "-" << std::setw(2) << std::setfill('0') << date_time_.m
         << "-" << std::setw(2) << std::setfill('0') << date_time_.d << " "
         << std::setw(2) << std::setfill('0') << date_time_.H << ":" << std::setw(2)
         << std::setfill('0') << date_time_.M << ":" << std::setw(2) << std::setfill('0')
         << date_time_.S;
    } else {
      UNREACHABLE() << "Unexpected type in TemporalConverter::toString(). Type: "
                    << toString(type);
    }
    return ss.str();
  }

 private:
  bool validateDateTime() {
    return date_time_.m > 12 || date_time_.d > 31 || date_time_.H > 23 ||
           date_time_.M > 59 || date_time_.S > 59;
  }
  void setDateTimeStruct(SQL_TIMESTAMP_STRUCT const* source) {
    date_time_.Y = source->year;
    date_time_.m = source->month;
    date_time_.d = source->day;
    date_time_.H = source->hour;
    date_time_.M = source->minute;
    date_time_.S = source->second;
    date_time_.n = source->fraction;
  }

  void setDateTimeStruct(SQL_TIME_STRUCT const* source) {
    date_time_.H = source->hour;
    date_time_.M = source->minute;
    date_time_.S = source->second;
  }

  void setDateTimeStruct(SQL_DATE_STRUCT const* source) {
    date_time_.Y = source->year;
    date_time_.m = source->month;
    date_time_.d = source->day;
  }

  DateTimeParser::DateTime date_time_;
  int64_t epoch_ = 0;
  int precision_ = 0;
};

}  // namespace

namespace foreign_storage {

std::string OdbcSelectDescriptor::getSelectStmtFromType() const {
  CHECK(select_type.has_value())
      << " getSelectStmtFromType called with no type set, unsupported case";
  switch (select_type.value()) {
    case SelectQueryType::kSelect:
      return getSelectStmt();
    case SelectQueryType::kCountNotNull:
      return getCountNotNullStmt();
    case SelectQueryType::kMaxLength:
      return getSelectMaxLengthStmt();
    case SelectQueryType::kMaxValue:
      return getMaxValueStmt();
    case SelectQueryType::kMinValue:
      return getMinValueStmt();
  }
  UNREACHABLE();
  return {};
}

std::string OdbcSelectDescriptor::getSelectStmt() const {
  return getSelectStmtSubquery() + ";";
}

std::string OdbcSelectDescriptor::getSelectMaxLengthStmt() const {
  return std::string{"select MAX(LENGTH(" + remote_column.column_name + ")) from (" +
                     getSelectStmtSubquery() + ") as alias_name;"};
}

std::string OdbcSelectDescriptor::getCountNotNullStmt() const {
  return std::string{"select COUNT(" + remote_column.column_name + ")  from (" +
                     getSelectStmtSubquery() + ") as alias_name;"};
}

std::string OdbcSelectDescriptor::getMaxValueStmt() const {
  if (column.columnType.is_boolean()) {
    return std::string("select MAX( CAST(" + remote_column.column_name +
                       " AS INT))  from (" + getSelectStmtSubquery() + ") as alias_name");
  }
  return std::string("select MAX( " + remote_column.column_name + " )  from (" +
                     getSelectStmtSubquery() + ") as alias_name");
}

std::string OdbcSelectDescriptor::getMinValueStmt() const {
  if (column.columnType.is_boolean()) {
    return std::string("select MIN( CAST(" + remote_column.column_name +
                       " AS INT))  from (" + getSelectStmtSubquery() + ") as alias_name");
  }
  return std::string("select MIN( " + remote_column.column_name + " )  from (" +
                     getSelectStmtSubquery() + ") as alias_name");
}

std::string OdbcSelectDescriptor::getSelectStmtSubquery() const {
  return std::string{"select " + remote_column.column_name + " from (" + select_command +
                     ") as alias_name order by " + order_by_command +
                     (limit < std::numeric_limits<size_t>::max()
                          ? " limit " + std::to_string(limit)
                          : "") +
                     (offset > 0 ? " offset " + std::to_string(offset) : "")};
}

size_t OdbcSelectDescriptor::getExpectedNumResultRows() const {
  CHECK(select_type.has_value())
      << " getSelectStmtFromType called with no type set, unsupported case";
  switch (select_type.value()) {
    case SelectQueryType::kSelect:
      return limit;
    case SelectQueryType::kCountNotNull:
      return 1;
    case SelectQueryType::kMaxLength:
      return 1;
    case SelectQueryType::kMaxValue:
      return 1;
    case SelectQueryType::kMinValue:
      return 1;
  }
  UNREACHABLE();
  return {};
}

int64_t temporal_conversion_utility(int8_t const* data_buffer_ptr,
                                    const SQLTypeInfo& sql_type_info) {
  time_t value = 0;

  if (sql_type_info.get_type() == SQLTypes::kTIME) {
    TemporalConverter<SQL_TIME_STRUCT> ts_s(data_buffer_ptr);
    value = ts_s.getEpoch();
  } else if (sql_type_info.get_type() == SQLTypes::kDATE) {
    TemporalConverter<SQL_DATE_STRUCT> td_s(data_buffer_ptr);
    value = td_s.getEpoch();
  } else if (sql_type_info.get_type() == SQLTypes::kTIMESTAMP) {
    TemporalConverter<SQL_TIMESTAMP_STRUCT> ts_s(data_buffer_ptr,
                                                 sql_type_info.get_dimension());
    value = ts_s.getEpoch();
  } else {
    LOG(FATAL) << "Invalid database type used with temporal conversion";
  }

  return value;
}

std::string temporal_to_string(int8_t const* data_buffer_ptr,
                               const SQLTypeInfo& sql_type_info) {
  std::string value;
  if (sql_type_info.get_type() == SQLTypes::kTIME) {
    TemporalConverter<SQL_TIME_STRUCT> ts_s(data_buffer_ptr);
    value = ts_s.toString(sql_type_info.get_type());
  } else if (sql_type_info.get_type() == SQLTypes::kDATE) {
    TemporalConverter<SQL_DATE_STRUCT> td_s(data_buffer_ptr);
    value = td_s.toString(sql_type_info.get_type());
  } else if (sql_type_info.get_type() == SQLTypes::kTIMESTAMP) {
    TemporalConverter<SQL_TIMESTAMP_STRUCT> ts_s(data_buffer_ptr,
                                                 sql_type_info.get_dimension());
    value = ts_s.toString(sql_type_info.get_type());
  } else {
    UNREACHABLE() << "Unexpected type in temporal string conversion. Type: "
                  << toString(sql_type_info.get_type());
  }
  return value;
}
bool is_decimal_odbc_type(int16_t sqltype) {
  return (sqltype == SQL_DECIMAL || sqltype == SQL_NUMERIC);
}
}  // namespace foreign_storage

namespace {

using OdbcStatus = std::tuple<bool, std::string>;

// Struct passed to SQLDescribeCol to get column info,
struct OdbcColumnDesc {
  // Depending on the type of dbms, column names are around 128 long
  // With scheme qualification they should still fit into 2048.
  // If the field is too small SQLDescribeCol should return
  // SQL_SUCCESS_WITH_INFO. providing the correct length in a
  // parameter.  TODO detect this situation and increase col_name.
  // NOTE postgres behaves this way, while sqlite does not.
  std::string col_name = std::string(2048, '\0');
  SQLSMALLINT col_type = 0;
  SQLULEN column_size = 0;
  SQLSMALLINT decimal_digits = 0;
  SQLSMALLINT nullable = 0;
  bool is_unsigned = true;

  operator std::string() const {
    std::stringstream ss;
    SQLSMALLINT dimension;
    if (col_type == SQL_DECIMAL || col_type == SQL_NUMERIC) {
      dimension = column_size;
    } else {
      dimension = decimal_digits;
    }
    ss << "Column name [" << col_name << "], dimension [" << dimension << "], type ["
       << col_type << "]";
    return ss.str();
  }
  bool isVariableWidthColumn() {
    // Remote db column type
    return col_type == SQL_VARCHAR || col_type == SQL_LONGVARCHAR ||
           col_type == SQL_WLONGVARCHAR || col_type == SQL_WVARCHAR;
  }
};

std::string quoted_identifier(const std::string& quoted_char,
                              const std::string& identifier) {
  return quoted_char + identifier + quoted_char;
}

void updateRemoteColumnDetails(foreign_storage::RemoteColumnDescription& rcd,
                               const OdbcColumnDesc& odbc_cd,
                               const std::string& quoted_char) {
  static const std::map<SQLSMALLINT, SQLLEN> octet_size_lookup{
      // Note variable length types are not included in this map
      {SQL_BIT, 1},
      {SQL_TINYINT, 1},
      {SQL_SMALLINT, 2},
      {SQL_INTEGER, 4},
      {SQL_REAL, 4},
      {SQL_BIGINT, 8},
      {SQL_DOUBLE, 8},
      {SQL_FLOAT, 8},
      {SQL_TYPE_TIME, 6},
      {SQL_TYPE_DATE, 6},
      {SQL_TYPE_TIMESTAMP, 16},
      // clang-off
      // Based on
      // https://docs.microsoft.com/en-us/sql/odbc/reference/appendixes/transfer-octet-length?view=sql-server-ver15
      // clang-on
  };

  rcd.column_name = quoted_identifier(quoted_char, odbc_cd.col_name);
  rcd.odbc_base_type = odbc_cd.col_type;
  // Some db odbc drivers do not accurately report the
  // transfer size of their columns (notably sqlite).
  // The size is well documented and is defined in octet_size_lookup
  if (auto iter = octet_size_lookup.find(rcd.odbc_base_type);
      iter != octet_size_lookup.end()) {
    // Note the lookup is the base ODBC type.  This is particularly important
    // in the case where one type of column is coerced to a different type.
    rcd.odbc_octet_transfer_size = iter->second;
  } else {
    rcd.odbc_octet_transfer_size = odbc_cd.column_size;
    if (rcd.odbc_base_type == SQL_DECIMAL || rcd.odbc_base_type == SQL_NUMERIC) {
      // add space for decimal point, sign and null terminator
      rcd.odbc_octet_transfer_size += 3;
    }
  }
  rcd.decimal_digits = odbc_cd.decimal_digits;
  rcd.is_unsigned = odbc_cd.is_unsigned;
}

std::string stringifyCatDescriptor(const ColumnDescriptor* const cd) {
  std::stringstream ss;
  ss << "HeavyDB Catalog: column name [" << cd->columnName
     << "], type [" + cd->columnType.get_type_name() + "]"
     << ", type code [" << cd->columnType.get_type() << "], dimension ["
     << cd->columnType.get_dimension() << "]";
  return ss.str();
}

/*
 * CHECK_STATUS and THROW_STATUS - Wrapping CHECK and throw in these
 * macros allows the  correct line number is reported in the logs
 */
#define CHECK_STATUS(odbc_status) \
  CHECK(std::get<0>(odbc_status)) << std::get<1>(odbc_status);

#define THROW_STATUS(odbc_status)                                               \
  {                                                                             \
    if (!std::get<0>(odbc_status)) {                                            \
      LOG(ERROR) << std::get<1>(odbc_status);                                   \
      throw foreign_storage::ForeignStorageException(std::get<1>(odbc_status)); \
    }                                                                           \
  }
#define VALIDATE_WITH_DETAIL(return_code, message, detail, disable_logging) \
  {                                                                         \
    if (!return_code) {                                                     \
      message += " .Extra details [" + detail + "]";                        \
      if (!disable_logging) {                                               \
        LOG(ERROR) << message;                                              \
      }                                                                     \
      throw foreign_storage::ForeignStorageException(message);              \
    }                                                                       \
  }
/**
 *  Utility functions to process extract
 * diagnositc odbc information.
 */
std::string get_sql_details(SQLSMALLINT handle_type, SQLHANDLE raw_handle) {
  std::string state(SQL_SQLSTATE_SIZE, '\0');
  std::string details(SQL_MAX_MESSAGE_LENGTH, '\0');

  std::stringstream msg;

  SQLSMALLINT i = 1;
  SQLSMALLINT length;
  SQLINTEGER sqlcode = 0;
  while (SQLGetDiagRec(handle_type,
                       raw_handle,
                       i++,
                       reinterpret_cast<SQLCHAR*>(&state[0]),
                       &sqlcode,
                       reinterpret_cast<SQLCHAR*>(&details[0]),
                       SQL_MAX_MESSAGE_LENGTH + 1,
                       &length) == SQL_SUCCESS) {
    state.resize(strlen(state.c_str()));
    details.resize(strlen(details.c_str()));
    msg << " [Odbc error SQLSTATE = [" << state << "]. Native Error Code = [" << sqlcode
        << "]. Details:\"" << details << "\"]";
  }
  return msg.str();
}

/**
 ***   OdbcBaseHandle ***l
 */
class OdbcBaseHandle {
  /*
   * Base (abstract) class to manage the raw_handle for
   * all odbc handles.  An odbc raw handle will be required
   * in all ODBC api calls; the type of the handle depends
   * on the call
   */
 protected:
  OdbcBaseHandle(SQLSMALLINT handle_type, SQLPOINTER raw_parent_handle)
      : raw_handle_type_(handle_type) {
    THROW_STATUS(sqlCallValidation(
        SQLAllocHandle(raw_handle_type_, raw_parent_handle, &raw_handle_)));
  }

 public:
  SQLPOINTER getRawHandle() { return raw_handle_; }

  SQLSMALLINT getType() const { return raw_handle_type_; }

  virtual ~OdbcBaseHandle() = 0;

  OdbcStatus sqlCallValidation(RETCODE odbc_status) {
    // SQL_SUCCESS_WITH_INFO may be reporting a situation/error that is recoverable - for
    // example: Error SQLSTATE = [01004]. Native Error Code = [-2]. Details:"The buffer
    // was too small for the colName."
    // TODO change this interface to allow the calling code to recover in these situation;
    // for now treat SQL_SUCCESS_WITH_INFO as an error and report the messages.
    if (SQL_SUCCEEDED(odbc_status)) {
      if (odbc_status == SQL_SUCCESS_WITH_INFO) {
        // treat as error - see above
        return {false, get_sql_details(getType(), getRawHandle())};
      }
      return {true, ""};
    }
    std::stringstream err_msg;
    err_msg << "Error: recieved code [" << odbc_status << "]. Expected [" << SQL_SUCCESS
            << "] or [" << SQL_SUCCESS_WITH_INFO << "]";
    // Error message contains the number for odbc_status.
    // The case statment simply provides the status name.
    switch (odbc_status) {
      case SQL_INVALID_HANDLE:
        err_msg << "\n:Type[SQL_INVALID HANDLE]";
        break;
      case SQL_ERROR:
        err_msg << "\n:Type[SQL_ERROR]";
        break;
      case SQL_NO_DATA_FOUND:
        err_msg << "\n:Type[SQL_NO_DATA_FOUND]";
        break;
      default:
        err_msg << "\n:Type[UNKNOWN]";
        break;
    }
    err_msg << get_sql_details(getType(), getRawHandle());
    return {false, err_msg.str()};
  }

 protected:
  SQLSMALLINT raw_handle_type_ = 0;
  SQLPOINTER raw_handle_ = nullptr;

 private:
  OdbcBaseHandle(const OdbcBaseHandle&) = delete;
  OdbcBaseHandle& operator=(const OdbcBaseHandle&) = delete;
};

OdbcBaseHandle::~OdbcBaseHandle() {
  auto [status, msg] = sqlCallValidation(SQLFreeHandle(raw_handle_type_, raw_handle_));
  CHECK(status) << "Error in ~OdbcBaseHandle allocation [" << msg << "]";
}

/*
 ***  OdbcEnvironmentHandle ***
 */

class OdbcEnvironmentHandle final : public OdbcBaseHandle {
  /*
   * The Environment handle is the first handle to be created and as
   * such doesn't have a 'parent' handle.
   *
   * It can be configured with a range of attributes
   * TODO - as different remote dbs are used the setting of the attibute(s)
   * via SQLSetEnvAttr may need to be configurable.
   */

  // Note regarding 'void* ValuePtr' argument for SQLSetEnvAttr.
  // From the Microsoft ODBC documantation for a windows implementation.
  //
  // ValuePtr - [Input] Pointer to the value to be associated with Attribute.
  // Depending on the value of Attribute, ValuePtr will be a 32-bit integer
  // value or point to a null-terminated character string
  //
  // Leaving aside the different sizes of int and longs on different operating systems
  // From the iODBC headers SQLPOINTER is a typedef for void* and SQL_OV_ODBC3 is
  // the unsigned long constant 3UL.  In this case the function expects
  // ValuePtr to be  void* ptr that contains the intergral vale 3UL.
  //

 public:
  OdbcEnvironmentHandle() : OdbcBaseHandle(SQL_HANDLE_ENV, nullptr) {
    THROW_STATUS(
        sqlCallValidation(SQLSetEnvAttr(getRawHandle(),
                                        SQL_ATTR_ODBC_VERSION,
                                        reinterpret_cast<SQLPOINTER>(SQL_OV_ODBC3),
                                        SQL_IS_UINTEGER)));
  }
};

/*
 ***  OdbcDatabaseHandle ***
 */

class OdbcDatabaseHandle final : public OdbcBaseHandle {
  /*
   * Usually the second handle created (taking an environment handle as its parent).
   * The Database handle is used to manage the connection to the remote db.  It is
   * used as the parent for subsquent handles that perform actions on the db.
   */
 public:
  OdbcDatabaseHandle(OdbcEnvironmentHandle& parent_environment_handle)
      : OdbcBaseHandle(SQL_HANDLE_DBC, parent_environment_handle.getRawHandle()) {}
  ~OdbcDatabaseHandle() override {
    auto [status, msg] = sqlCallValidation(SQLDisconnect(raw_handle_));
    if (!status) {
      LOG(ERROR) << "Error in ~OdbcDatabaseHandle allocation [" << msg << "]";
    }
  }

  std::string getDBMSInfoString(SQLUSMALLINT info_type) {
    static constexpr int32_t dbms_name_max_size = 1024;
    std::string dbms_name(dbms_name_max_size, '\0');
    SQLSMALLINT actual_buffer_size;
    // If the 1024 wasn't big enough SQLGetInfo will return success with info
    // with the actual length returned in actual_buffer_size.
    // As with the other places this happens the check/validation code treats this
    // as an error.  TODO check for this condition and reallocate the buffer.
    CHECK_STATUS(sqlCallValidation(SQLGetInfo(
        raw_handle_, info_type, &dbms_name[0], dbms_name.length(), &actual_buffer_size)));

    dbms_name.resize(actual_buffer_size);
    return dbms_name;
  }

  SQLUSMALLINT getDBMSInfoSmallInt(SQLUSMALLINT info_type) {
    SQLUSMALLINT retval;
    CHECK_STATUS(
        sqlCallValidation(SQLGetInfo(raw_handle_, info_type, &retval, 0, nullptr)));
    return retval;
  }

  void validateQuotedIdentifierCaseSensitivity() {
    auto result = getDBMSInfoSmallInt(SQL_QUOTED_IDENTIFIER_CASE);
    switch (result) {
      case SQL_IC_UPPER:
        LOG(WARNING) << "Non-standard RDBMS quoted identifier behaviour detected: Quoted "
                        "identifiers in SQL are not case-sensitive and are stored in "
                        "uppercase in the system catalog.";
        break;
      case SQL_IC_LOWER:
        LOG(WARNING) << "Non-standard RDBMS quoted identifier behaviour detected: Quoted "
                        "identifiers in SQL are not case-sensitive and are stored in "
                        "lowercase in the system catalog.";
        break;
      case SQL_IC_MIXED:
        LOG(WARNING) << "Non-standard RDBMS quoted identifier behaviour detected: Quoted "
                        "identifiers in SQL are not case-sensitive and are stored in "
                        "mixed case in the system catalog.";
        break;
    }
  }
};

/*
 ***  template a function map for binding block of memory to columns  ***
 */

template <SQLSMALLINT TARGET_DATA_TYPE>
RETCODE data_binder(int8_t* datablock,
                    SQLPOINTER handle,
                    SQLSMALLINT column_index,
                    size_t column_size,
                    SQLLEN* strlen_or_indptr) {
  return SQLBindCol(
      handle, column_index, TARGET_DATA_TYPE, datablock, column_size, strlen_or_indptr);
}

static const std::map<
    int16_t,
    std::function<RETCODE(int8_t*, SQLPOINTER, SQLSMALLINT, size_t, SQLLEN*)>>
    data_binding_function_map = {{SQL_DECIMAL, &data_binder<SQL_C_CHAR>},
                                 {SQL_NUMERIC, &data_binder<SQL_C_CHAR>},
                                 {SQL_TYPE_DATE, &data_binder<SQL_C_TYPE_DATE>},
                                 {SQL_TYPE_TIME, &data_binder<SQL_C_TYPE_TIME>},
                                 {SQL_TYPE_TIMESTAMP, &data_binder<SQL_C_TYPE_TIMESTAMP>},
                                 {SQL_INTEGER, &data_binder<SQL_INTEGER>},
                                 {SQL_SMALLINT, &data_binder<SQL_SMALLINT>},
                                 {SQL_TINYINT, &data_binder<SQL_TINYINT>},
                                 {SQL_BIGINT, &data_binder<SQL_C_SBIGINT>},
                                 {SQL_DOUBLE, &data_binder<SQL_DOUBLE>},
                                 {SQL_CHAR, &data_binder<SQL_C_CHAR>},
                                 {SQL_WCHAR, &data_binder<SQL_C_CHAR>},
                                 {SQL_VARCHAR, &data_binder<SQL_C_CHAR>},
                                 {SQL_WVARCHAR, &data_binder<SQL_C_CHAR>},
                                 {SQL_LONGVARCHAR, &data_binder<SQL_C_CHAR>},
                                 {SQL_WLONGVARCHAR, &data_binder<SQL_C_CHAR>},
                                 {SQL_BIT, &data_binder<SQL_C_BIT>},
                                 {SQL_REAL, &data_binder<SQL_REAL>},
                                 {SQL_FLOAT, &data_binder<SQL_DOUBLE>}};

/*
 ***  OdbcSelectHandle ***
 */

class OdbcSelectHandle : public OdbcBaseHandle {
  /*
   * An abstract class that manages all 'actions' on the remote db, such as preparing and
   * executing sql statements.
   *
   * Inherited by OdbcMultipleRecordSelectHandle and  OdbcSingleRecordSelectHandle
   */

 public:
  OdbcSelectHandle(OdbcDatabaseHandle& parent_environment_handle)
      : OdbcBaseHandle(SQL_HANDLE_STMT, parent_environment_handle.getRawHandle()) {}

  void bindColumnData(SQLSMALLINT odbc_column_type,
                      int8_t* datablock,
                      SQLSMALLINT column_index,
                      size_t bind_size,
                      SQLLEN* strlen_or_indptr) {
    auto function_itr = data_binding_function_map.find(odbc_column_type);
    CHECK(function_itr != data_binding_function_map.end())
        << "For ODBC type " << odbc_column_type;
    CHECK_STATUS(sqlCallValidation(function_itr->second(
        datablock, getRawHandle(), column_index, bind_size, strlen_or_indptr)));
  }

  void prepare(const std::string& sql_command) {
    auto [return_code, message] = sqlCallValidation(
        SQLPrepare(getRawHandle(),
                   reinterpret_cast<SQLCHAR*>(const_cast<char*>(&sql_command[0])),
                   sql_command.length()));
    VALIDATE_WITH_DETAIL(return_code, message, sql_command, false);
  }

  void execDirectSql(const std::string& sql_command, bool disable_logging = false) {
    auto [return_code, message] = sqlCallValidation(
        SQLExecDirect(getRawHandle(),
                      reinterpret_cast<SQLCHAR*>(const_cast<char*>(&sql_command[0])),
                      SQL_NTS));
    VALIDATE_WITH_DETAIL(return_code, message, sql_command, disable_logging);
  }

  void execSqlAllowSuccessWithInfo(const std::string& sql_command) {
    // For use with test code only
    auto [return_code, message] = sqlCallValidation(
        SQLExecDirect(getRawHandle(),
                      reinterpret_cast<SQLCHAR*>(const_cast<char*>(&sql_command[0])),
                      SQL_NTS));
    if (return_code != SQL_SUCCESS && return_code != SQL_SUCCESS_WITH_INFO) {
      message += " .Extra details [" + sql_command + "]";
      LOG(ERROR) << message;
      throw foreign_storage::ForeignStorageException(message);
    }
  }

  OdbcColumnDesc describeCol(SQLSMALLINT column_index) {
    OdbcColumnDesc cd;
    SQLSMALLINT actual_column_name_len = 0;
    // SQLDescribeCol returns the display size of a column
    // (which isn't needed) rather than its size in bytes
    // or the octet_transfer_size.  To get the size in bytes
    // make an extra call to SQLColAttribute using SQL_DESC_OCTET.
    // However, some drivers don't honour this call (sqlite) returning
    // the display size.  For the moment a hardwired table is used
    // instead
    CHECK_STATUS(sqlCallValidation(
        SQLDescribeCol(getRawHandle(),
                       column_index,  // Note Odbc columns start numbering from 1
                       reinterpret_cast<SQLCHAR*>(&(cd.col_name)[0]),
                       cd.col_name.length(),
                       &actual_column_name_len,
                       &(cd.col_type),
                       &(cd.column_size),
                       &(cd.decimal_digits),
                       &(cd.nullable))));

    cd.col_name.resize(actual_column_name_len);
    cd.is_unsigned = columnUnsigned(column_index);
    return cd;
  }

  std::string columnType(SQLSMALLINT column_index) {
    // As long as SQL_SUCCESS_WITH_INFO is treated as an error,
    // if column type is too big for 1024 SQLColAttribute will error
    SQLSMALLINT column_type_length{1024};
    std::string column_type(column_type_length, ' ');

    THROW_STATUS(
        sqlCallValidation(SQLColAttribute(getRawHandle(),
                                          column_index,
                                          SQL_COLUMN_TYPE_NAME,
                                          reinterpret_cast<SQLCHAR*>(&column_type[0]),
                                          column_type.length(),
                                          &column_type_length,
                                          nullptr)));

    column_type.resize(column_type_length);
    return column_type;
  }

  bool columnUnsigned(SQLSMALLINT column_index) {
    int64_t is_unsigned = 0;

    THROW_STATUS(sqlCallValidation(SQLColAttribute(getRawHandle(),
                                                   column_index,
                                                   SQL_DESC_UNSIGNED,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   &is_unsigned)))

    return static_cast<bool>(is_unsigned);
  }

  ~OdbcSelectHandle() override = 0;

  // TODO add a set of phases to ensure calls are made in the appropriate order as in -
  // OdbcSelectPhase getPhase() { return phase_; }
};

OdbcSelectHandle::~OdbcSelectHandle() {}

/*
 ***  OdbcMultipleRecordSelectHandle ***
 */

class OdbcMultipleRecordSelectHandle final : public OdbcSelectHandle {
  /*
   * Uses its constructor to set options to read allow multiple rows
   * to be read in a single efficient amd provides methods to return
   * blocks of records.
   */
 public:
  OdbcMultipleRecordSelectHandle(OdbcDatabaseHandle& parent_environment_handle,
                                 int64_t num_rows_to_select)
      : OdbcSelectHandle(parent_environment_handle)
      , numRowsRequested_(num_rows_to_select) {
    CHECK_GT(numRowsRequested_, 0);
    CHECK_STATUS(sqlCallValidation(SQLSetStmtAttr(
        getRawHandle(), SQL_ATTR_PARAM_BIND_TYPE, SQL_PARAM_BIND_BY_COLUMN, 0)));
    // Note set the number of rows to fetch to numRowsRequested_
    THROW_STATUS(
        sqlCallValidation(SQLSetStmtAttr(getRawHandle(),
                                         SQL_ATTR_ROW_ARRAY_SIZE,
                                         reinterpret_cast<SQLPOINTER>(numRowsRequested_),
                                         0)));
    // Note differes from call above.  Sets the location for the ODBC layer to set
    // the number of rows that were fetched.
    CHECK_STATUS(sqlCallValidation(SQLSetStmtAttr(
        getRawHandle(), SQL_ATTR_ROWS_FETCHED_PTR, (SQLPOINTER)&numRowsFetched_, 0)));
  }

  void execDirectSqlAndFetchRecords(const std::string& sql_command) {
    execDirectSql(sql_command);
    fetchRecordNextBlock();
    CHECK_LE(numRowsFetched_, numRowsRequested_);
    validateIndividualRowReturnCodes();
  }

  size_t execDirectSqlAndFetchRecordsInBatches(
      const std::string& sql_command,
      foreign_storage::ResultSetProcessor result_set_processor,
      int8_t* results,
      SQLLEN* null_or_strlen_ptr,
      size_t data_size,
      int64_t row_offset,
      const foreign_storage::RemoteColumnDescription& remote_column_descriptor) {
    execDirectSql(sql_command);

    size_t total_num_rows_fetched = 0;

    foreign_storage::RemoteData remote_data;
    remote_data.data_ptr = results;
    remote_data.odbc_base_type = remote_column_descriptor.odbc_base_type;
    remote_data.odbc_octet_transfer_size =
        remote_column_descriptor.odbc_octet_transfer_size;
    remote_data.is_unsigned = remote_column_descriptor.is_unsigned;
    RETCODE ret;
    while ((ret = SQLFetchScroll(getRawHandle(), SQL_FETCH_NEXT, 0)) != SQL_NO_DATA) {
      THROW_STATUS(sqlCallValidation(ret));  // check return code

      remote_data.null_or_strlen.resize(numRowsFetched_);
      std::transform(null_or_strlen_ptr,
                     null_or_strlen_ptr + numRowsFetched_,
                     remote_data.null_or_strlen.data(),
                     [](const SQLLEN size) -> std::optional<size_t> {
                       if (size == SQL_NULL_DATA) {
                         return std::nullopt;
                       }
                       CHECK(size >= 0);
                       return size;
                     });

      validateIndividualRowReturnCodes();

      // process batch
      result_set_processor(
          numRowsFetched_, data_size, row_offset + total_num_rows_fetched, remote_data);
      total_num_rows_fetched += numRowsFetched_;
    }
    return total_num_rows_fetched;
  }

  SQLUINTEGER getNumRowsFetched() { return (numRowsFetched_); }

  void setStmtAttrRowStatusPtr(int64_t num_rows) {
    rowValidationReturnCode_.resize(num_rows);
    CHECK_STATUS(sqlCallValidation(
        SQLSetStmtAttr(getRawHandle(),
                       SQL_ATTR_ROW_STATUS_PTR,
                       reinterpret_cast<SQLPOINTER>(&rowValidationReturnCode_[0]),
                       0)));
  }

  void validateIndividualRowReturnCodes() {
    if (rowValidationReturnCode_.size() == 0) {
      return;
    }
    auto itr = std::find_if(rowValidationReturnCode_.begin(),
                            rowValidationReturnCode_.begin() + numRowsFetched_,
                            [](auto return_code) { return !SQL_SUCCEEDED(return_code); });
    CHECK(itr == rowValidationReturnCode_.begin() + numRowsFetched_)
        << "Invalid row at index [" << (itr - rowValidationReturnCode_.begin())
        << "] returned.";
  }

 private:
  void fetchRecordNextBlock() {
    // The number of records in the block has been pre-calculated It should be the
    // same size as the number of records expected by the chunk being filled.  We should
    // always get the exact number of record back; SQL_NO_DATA is therefore considered an
    // error.
    THROW_STATUS(sqlCallValidation(SQLFetchScroll(getRawHandle(), SQL_FETCH_NEXT, 0)));
  }

  std::vector<SQLUSMALLINT> rowValidationReturnCode_;
  SQLUINTEGER numRowsFetched_{0};
  int64_t numRowsRequested_;
};

/*
 ***  OdbcSingleRecordSelectHandle ***
 */

class OdbcSingleRecordSelectHandle final : public OdbcSelectHandle {
 public:
  OdbcSingleRecordSelectHandle(OdbcDatabaseHandle& parent_environment_handle)
      : OdbcSelectHandle(parent_environment_handle) {}

  void fetchRecord() { THROW_STATUS(sqlCallValidation(SQLFetch(getRawHandle()))); }
};

/*
 * End of Odbc handle classes.
 */

// clang-format off
static std::map<SQLSMALLINT, std ::list<foreign_storage::RemoteColumnDescription>>
    /* Note the byte size values listed in the reference map below come from
     * https://docs.microsoft.com/en-us/sql/odbc/reference/appendixes/transfer-octet-length?view=sql-server-ver15
     *
     * The size should also be available by querying the remote db using SQLColAttribute
     * with SQL_DESC_OCTET_LENGTH as the FieldIdentifier parameter.
     *
     * Unfortunately while the SQLColAttribute approach works with postgres the current
     * sqlite odbc driver returns an incorrect value (the display size rather than the
     * byte size).
     *
     * TODO ensure this works with any new ODBC connections and perhaps look at making it
     * a configuration option based on the db type.
     */
    odbc_to_omnisci_reference_map__{
        {{{SQL_LONGVARCHAR},
          {
              {SQLTypes::kTEXT, 0, 0, "kTEXT"},
              {SQLTypes::kPOINT, 0, 0, "kPOINT"},
              {SQLTypes::kMULTIPOINT, 0, 0, "kMULTIPOINT"},
              {SQLTypes::kLINESTRING, 0, 0, "kLINESTRING"},
              {SQLTypes::kMULTILINESTRING, 0, 0, "kMULTILINESTRING"},
              {SQLTypes::kPOLYGON, 0, 0, "kPOLYGON"},
              {SQLTypes::kMULTIPOLYGON, 0, 0, "kMULTIPOLYGON"},
          }},
         {{SQL_VARCHAR},
          {
              {SQLTypes::kTEXT, 0, 0, "kTEXT"},
              {SQLTypes::kPOINT, 0, 0, "kPOINT"},
              {SQLTypes::kMULTIPOINT, 0, 0, "kMULTIPOINT"},
              {SQLTypes::kLINESTRING, 0, 0, "kLINESTRING"},
              {SQLTypes::kMULTILINESTRING, 0, 0, "kMULTILINESTRING"},
              {SQLTypes::kPOLYGON, 0, 0, "kPOLYGON"},
              {SQLTypes::kMULTIPOLYGON, 0, 0, "kMULTIPOLYGON"},
          }},
         {{SQL_NUMERIC},
          {
              {SQLTypes::kDECIMAL, 0, 0, "kDECIMAL"},
              {SQLTypes::kTINYINT, 0, 0, "kTINYINT"},
              {SQLTypes::kSMALLINT, 0, 0, "kSMALLINT"},
              {SQLTypes::kINT, 0, 0, "kINT"},
              {SQLTypes::kBIGINT, 0, 0, "kBIGINT"},
          }},
         {{SQL_DECIMAL},
          {
              {SQLTypes::kDECIMAL, 0, 0, "kDECIMAL"},
              {SQLTypes::kTINYINT, 0, 0, "kTINYINT"},
              {SQLTypes::kSMALLINT, 0, 0, "kSMALLINT"},
              {SQLTypes::kINT, 0, 0, "kDECIMAL"},
              {SQLTypes::kBIGINT, 0, 0, "kBIGINT"},
          }},
         {{SQL_TYPE_DATE}, {{SQLTypes::kDATE, 0, 0, "kDATE"}}},
         {{SQL_TYPE_TIME}, {{SQLTypes::kTIME, 0, 0, "kTIME"}}},
         {{SQL_TYPE_TIMESTAMP}, {{SQLTypes::kTIMESTAMP, 0, 0, "kTIMESTAMP"}}},
         {{SQL_BIGINT},
          {
              {SQLTypes::kBIGINT, 0, 0, "kBIGINT"},
              {SQLTypes::kINT, 0, 0, "kINT"},
              {SQLTypes::kSMALLINT, 0, 0, "kSMALLINT"},
              {SQLTypes::kTINYINT, 0, 0, "kTINYINT"},
          }},
         {{SQL_INTEGER},
          {
              {SQLTypes::kINT, 0, 0, "kINT"},
              {SQLTypes::kSMALLINT, 0, 0, "kSMALLINT"},
              {SQLTypes::kTINYINT, 0, 0, "kTINYINT"},
              {SQLTypes::kBIGINT, 0, 0, "kBIGINT"},
          }},
         {{SQL_SMALLINT},
          {
              {SQLTypes::kSMALLINT, 0, 0, "kSMALLINT"},
              {SQLTypes::kTINYINT, 0, 0, "kTINYINT"},
              {SQLTypes::kINT, 0, 0, "kINT"},
          }},
         {{SQL_TINYINT},
          {
              {SQLTypes::kTINYINT, 0, 0, "kTINYINT"},
              {SQLTypes::kSMALLINT, 0, 0, "kSMALLINT"},
          }},
         {{SQL_DOUBLE},
          {
              {SQLTypes::kDOUBLE, 0, 0, "kDOUBLE"},
              {SQLTypes::kFLOAT, 0, 0, "kFLOAT"},
          }},
         {{SQL_FLOAT},
          {
              {SQLTypes::kDOUBLE, 0, 0, "kDOUBLE"},
              {SQLTypes::kFLOAT, 0, 0, "kFLOAT"},
          }},
         {{SQL_REAL},
          {{SQLTypes::kFLOAT, 0, 0, "kFLOAT"}}},
         {{SQL_CHAR}, {{SQLTypes::kTEXT, 0, 0, "kTEXT"}}},
         {{SQL_WCHAR}, {{SQLTypes::kTEXT, 0, 0, "kTEXT"}}},
         {{SQL_WVARCHAR},
          {
              {SQLTypes::kTEXT, 0, 0, "kTEXT"},
              {SQLTypes::kPOINT, 0, 0, "kPOINT"},
              {SQLTypes::kMULTIPOINT, 0, 0, "kMULTIPOINT"},
              {SQLTypes::kLINESTRING, 0, 0, "kLINESTRING"},
              {SQLTypes::kMULTILINESTRING, 0, 0, "kMULTILINESTRING"},
              {SQLTypes::kPOLYGON, 0, 0, "kPOLYGON"},
              {SQLTypes::kMULTIPOLYGON, 0, 0, "kMULTIPOLYGON"},
          }},
         {{SQL_WLONGVARCHAR},
          {
              {SQLTypes::kTEXT, 0, 0, "kTEXT"},
              {SQLTypes::kPOINT, 0, 0, "kPOINT"},
              {SQLTypes::kMULTIPOINT, 0, 0, "kMULTIPOINT"},
              {SQLTypes::kLINESTRING, 0, 0, "kLINESTRING"},
              {SQLTypes::kMULTILINESTRING, 0, 0, "kMULTILINESTRING"},
              {SQLTypes::kPOLYGON, 0, 0, "kPOLYGON"},
              {SQLTypes::kMULTIPOLYGON, 0, 0, "kMULTIPOLYGON"},
          }},
         {{SQL_BIT},
          {{SQLTypes::kBOOLEAN, 0, 0, "kBOOLEAN"}}}}};
// clang-format on
}  // namespace

namespace foreign_storage {

bool RemoteColumnDescription::isMappingAllowed(
    const SQLTypeInfo& omnisci_typeinfo) const {
  if (omnisci_typeinfo.get_type() != omnisci_type) {
    return false;
  }
  if (omnisci_type == SQLTypes::kTIMESTAMP) {
    if (omnisci_typeinfo.get_dimension() == 0) {
      return true;
    }
    int32_t to_precision;
    switch (decimal_digits) {
      case 1:
      case 2:
      case 3:
        to_precision = 3;
        break;
      case 4:
      case 5:
      case 6:
        to_precision = 6;
        break;
      case 7:
      case 8:
      case 9:
        to_precision = 9;
        break;
      default:
        return false;
    }
    return omnisci_typeinfo.get_dimension() == to_precision;
  }
  if (IS_INTEGER(omnisci_type)) {
    // Widening coercions for Integer types only allowed from unsigned integer types
    switch (odbc_base_type) {
      case SQL_BIGINT:
        if (is_unsigned) {
          return false;
        }
        break;
      case SQL_INTEGER:
        if (omnisci_type == SQLTypes::kBIGINT && !is_unsigned) {
          return false;
        }
        break;
      case SQL_SMALLINT:
        if (omnisci_type == SQLTypes::kINT && !is_unsigned) {
          return false;
        }
        break;
      case SQL_TINYINT:
        if (omnisci_type == SQLTypes::kSMALLINT && !is_unsigned) {
          return false;
        }
        break;
    }
  }
  if (omnisci_typeinfo.is_decimal()) {
    if (omnisci_typeinfo.get_precision() > getDecimalPrecision() ||
        omnisci_typeinfo.get_scale() > decimal_digits) {
      return false;
    }
  }
  return true;
}

int32_t RemoteColumnDescription::getDecimalPrecision() const {
  CHECK_EQ(omnisci_type, kDECIMAL);
  // Remove added size for decimal place and null terminator and sign.
  return odbc_octet_transfer_size - 3;
}

/*
 * OdbcConnectionImpl class.  Manages ODBC api calls.
 * The class has an environment handle and db connection handle
 * Statement handles are created as locals with methods.
 */

class OdbcConnectionImpl : public OdbcConnection {
 public:
  OdbcConnectionImpl(OdbcConnectionInfo connection_info, const UserMapping* user_mapping);
  size_t getRecordCnt(const std::string& sql_select) override;

  void getRemoteColumnDetailsValidateRemoteMetaData(
      std::string sql_command,
      std::list<ColumnDescriptor const*>& catalog_sqltypes,
      std::vector<foreign_storage::RemoteColumnDescription>& remote_column_details)
      override;

  void runSelectCmd(const OdbcSelectDescriptor& select_desc,
                    const size_t buffer_byte_size,
                    ResultSetProcessor result_set_processor) override;

  std::vector<ColumnRemoteData> runDataPreviewMultiColumnSelectCmd(
      const std::string& select_command,
      size_t row_count) override;

  // Used for testing to run arbitrary sql commands.
  void runSqlAllowSuccessWithInfo(const std::string& sql) override;
  void runSql(const std::string& sql) override;

  ~OdbcConnectionImpl() override {}

  size_t getChunkNullCount(const OdbcSelectDescriptor& select_desc) override;

  static std::set<std::string> getInstalledDrivers();

  void validateDbmsCompatible();

  size_t getStringWidthForColumn(SQLSMALLINT column_index,
                                 const std::string& select_command,
                                 const std::string& col_name,
                                 size_t row_count,
                                 int64_t row_offset = 0);

 private:
  std::pair<size_t, size_t> getColumnWidthAndBatchSize(
      const OdbcSelectDescriptor& select_desc,
      const int64_t buffer_byte_size,
      const size_t max_buffer_size);
  size_t getMaxStringLength(const OdbcSelectDescriptor& select_desc);
  OdbcEnvironmentHandle environment_handle_{};
  OdbcDatabaseHandle db_connectionHandle_;
  std::string dbms_type_name_{};  // will be set to something like SQLite, PostgreSQL
  std::string dbms_type_vers_{};
  std::string dbms_quoted_identifier_string_{};
};

namespace {
std::string get_option(const UserMapping* user_mapping, const std::string& attribute) {
  if (user_mapping != nullptr) {
    auto options = user_mapping->getUnencryptedOptions();
    if (auto it = options.find(attribute); it != options.end()) {
      return it->second;
    }
  }
  return "";
}

char* get_ptr_if_not_empty(const std::string& str) {
  if (!str.empty()) {
    return const_cast<char*>(&str[0]);
  }
  return nullptr;
}

}  // namespace

OdbcConnectionImpl::OdbcConnectionImpl(OdbcConnectionInfo connection_info,
                                       const UserMapping* user_mapping)
    : db_connectionHandle_{environment_handle_} {
  if (connection_info.data_source_name != std::nullopt) {
    // Username and password must be returned as a std::string and not a raw char* pointer
    // here, otherwise code optimization results in unintelligble strings being passed
    // into SQLConnect.
    const auto& username =
        get_option(user_mapping, foreign_storage::OdbcDataWrapper::ODBC_USERNAME);
    const auto& password =
        get_option(user_mapping, foreign_storage::OdbcDataWrapper::ODBC_PASSWORD);

    THROW_STATUS(db_connectionHandle_.sqlCallValidation(SQLConnect(
        db_connectionHandle_.getRawHandle(),
        reinterpret_cast<SQLCHAR*>(&connection_info.data_source_name.value()[0]),
        connection_info.data_source_name.value().length(),
        reinterpret_cast<SQLCHAR*>(get_ptr_if_not_empty(username)),
        username.length(),
        reinterpret_cast<SQLCHAR*>(get_ptr_if_not_empty(password)),
        password.length())));
  } else {
    CHECK(connection_info.connection_string != std::nullopt);
    auto& full_connection_string = connection_info.connection_string.value();
    full_connection_string +=
        ((full_connection_string.back() != ';') ? ";" : "") +
        get_option(user_mapping, foreign_storage::OdbcDataWrapper::ODBC_CREDENTIAL);
    THROW_STATUS(db_connectionHandle_.sqlCallValidation(
        SQLDriverConnect(db_connectionHandle_.getRawHandle(),
                         nullptr,
                         reinterpret_cast<SQLCHAR*>(&full_connection_string[0]),
                         full_connection_string.length(),
                         nullptr,
                         0,
                         nullptr,
                         SQL_DRIVER_NOPROMPT)));
  }
  // NOTE: for DBMS that do not support quoted identifiers, an empty string is
  // returned for SQL_IDENTIFIER_QUOTE_CHAR below, according to documentation
  dbms_quoted_identifier_string_ =
      db_connectionHandle_.getDBMSInfoString(SQL_IDENTIFIER_QUOTE_CHAR);
  dbms_type_name_ = db_connectionHandle_.getDBMSInfoString(SQL_DBMS_NAME);
  dbms_type_vers_ = db_connectionHandle_.getDBMSInfoString(SQL_DBMS_VER);
  validateDbmsCompatible();
  // Check for quoted identifier case sensitivity and print log warnings for non-standard
  // cases
  db_connectionHandle_.validateQuotedIdentifierCaseSensitivity();
}

size_t OdbcConnectionImpl::getChunkNullCount(const OdbcSelectDescriptor& select_desc) {
  auto remote_column_descriptor = select_desc.remote_column;
  auto column = select_desc.column;
  // Do an initial count to determine the number of NULLs and if there are any valid
  // entries
  OdbcSingleRecordSelectHandle count_statement_handle(db_connectionHandle_);

  auto count_command = select_desc.getCountNotNullStmt();

  SQLINTEGER num_non_null_rows = 0;
  SQLLEN null_or_strlen;
  count_statement_handle.bindColumnData(SQL_INTEGER,
                                        reinterpret_cast<int8_t*>(&num_non_null_rows),
                                        1,
                                        sizeof(SQLINTEGER),
                                        &null_or_strlen);
  count_statement_handle.execDirectSql(count_command);
  count_statement_handle.fetchRecord();

  CHECK_LE(static_cast<size_t>(num_non_null_rows), select_desc.limit);
  return select_desc.limit - static_cast<size_t>(num_non_null_rows);
}

size_t OdbcConnectionImpl::getMaxStringLength(const OdbcSelectDescriptor& select_desc) {
  OdbcSingleRecordSelectHandle query_statement_handle(db_connectionHandle_);
  SQLINTEGER max_length = 0;
  SQLLEN null_or_strlen;
  query_statement_handle.bindColumnData(SQL_INTEGER,
                                        reinterpret_cast<int8_t*>(&max_length),
                                        1,
                                        sizeof(SQLINTEGER),
                                        &null_or_strlen);

  query_statement_handle.execDirectSql(select_desc.getSelectMaxLengthStmt());

  query_statement_handle.fetchRecord();

  if (null_or_strlen == SQL_NULL_DATA) {
    max_length = 0;
  }

  CHECK_GE(max_length, 0);

  return max_length + 1;  // account for the null character
}

size_t OdbcConnectionImpl::getRecordCnt(const std::string& sql_select) {
  // Note postgres requires the 'as name' for the subquery.
  std::string sql_counter = "select count(1) from (" + sql_select + ") as alias_name";

  OdbcSingleRecordSelectHandle query_statement_handle(db_connectionHandle_);
  size_t num_rows = 0;
  query_statement_handle.bindColumnData(
      SQL_INTEGER, reinterpret_cast<int8_t*>(&num_rows), 1, sizeof(size_t), nullptr);
  query_statement_handle.execDirectSql(sql_counter);

  query_statement_handle.fetchRecord();

  return num_rows;
}

namespace {
inline void throw_unsupported_column_matching(const std::string& remote_col_name,
                                              const std::string& col_name,
                                              const std::string& remote_col_details,
                                              const std::string& col_details) {
  throw ForeignStorageException("Remote database column '" + remote_col_name +
                                "' mapped onto HeavyDB column '" + col_name +
                                "' not currently supported by ODBC foreign storage "
                                "interface. Remote column type = [" +
                                remote_col_details + "] HeavyDB column type = [" +
                                col_details + "]");
}
}  // namespace

void OdbcConnectionImpl::getRemoteColumnDetailsValidateRemoteMetaData(
    std::string sql_command,
    std::list<ColumnDescriptor const*>& cat_columns,
    std::vector<foreign_storage::RemoteColumnDescription>& remote_column_details) {
  OdbcSingleRecordSelectHandle query_statement_handle(db_connectionHandle_);

  query_statement_handle.prepare(sql_command);
  SQLSMALLINT num_columns = 0;

  THROW_STATUS(query_statement_handle.sqlCallValidation(
      SQLNumResultCols(query_statement_handle.getRawHandle(), &num_columns)));

  int num_logical_columns = cat_columns.size();

  if (num_columns != boost::numeric_cast<SQLSMALLINT>(num_logical_columns)) {
    throw ForeignStorageException{"Error.  Number of columns [" +
                                  std::to_string(num_columns) +
                                  "] returned for remote db query [" + sql_command +
                                  "] does not match HeavyDB catalog column size of " +
                                  std::to_string(num_logical_columns)};
  }

  for (auto desc : cat_columns | boost::adaptors::indexed(0)) {
    // column_definition_in_odbc_types holds the column details in terms of ODBC types
    // which are are only 'understood' at this layer.

    const ColumnDescriptor* catalog_column_descriptor = desc.value();
    CHECK(!catalog_column_descriptor->isGeoPhyCol);

    OdbcColumnDesc column_definition_in_odbc_types =
        query_statement_handle.describeCol(desc.index() + 1);

    auto omnisci_ref_type =
        odbc_to_omnisci_reference_map__.find(column_definition_in_odbc_types.col_type);

    auto mapping_found = omnisci_ref_type != odbc_to_omnisci_reference_map__.end();
    const foreign_storage::RemoteColumnDescription*
        remote_column_definition_omnisci_types_ptr = nullptr;
    if (mapping_found) {
      const auto& omnisci_ref_type_list = omnisci_ref_type->second;
      auto iter = std::find_if(
          omnisci_ref_type_list.begin(),
          omnisci_ref_type_list.end(),
          [&catalog_column_descriptor](
              const foreign_storage::RemoteColumnDescription& remote_column_descriptor) {
            return remote_column_descriptor.omnisci_type ==
                   catalog_column_descriptor->columnType.get_type();
          });
      if (iter != omnisci_ref_type_list.end()) {
        mapping_found = true;
        remote_column_definition_omnisci_types_ptr = &(*iter);
      } else {
        mapping_found = false;
      }
    }

    // TODO rather than throwing on the first error would be better to check all columns
    // accumulating errors and then throw (if any).
    if (!mapping_found) {
      throw_unsupported_column_matching(
          column_definition_in_odbc_types.col_name,
          catalog_column_descriptor->columnName,
          static_cast<std::string>(column_definition_in_odbc_types),
          stringifyCatDescriptor(catalog_column_descriptor));
    }

    // Create a remote column description using omnisci types
    // and populate it with the data returned in odbc types
    CHECK(remote_column_definition_omnisci_types_ptr);
    foreign_storage::RemoteColumnDescription remote_column_definition_omnisci_types =
        *remote_column_definition_omnisci_types_ptr;

    updateRemoteColumnDetails(remote_column_definition_omnisci_types,
                              column_definition_in_odbc_types,
                              dbms_quoted_identifier_string_);
    remote_column_definition_omnisci_types.omnisci_column_id = desc.value()->columnId;

    if (!remote_column_definition_omnisci_types.isMappingAllowed(
            catalog_column_descriptor->columnType)) {
      throw_unsupported_column_matching(
          column_definition_in_odbc_types.col_name,
          catalog_column_descriptor->columnName,
          static_cast<std::string>(column_definition_in_odbc_types),
          stringifyCatDescriptor(catalog_column_descriptor));
    }

    remote_column_details.emplace_back(remote_column_definition_omnisci_types);
  }
  return;
}

void OdbcConnectionImpl::runSqlAllowSuccessWithInfo(const std::string& sql) {
  OdbcSingleRecordSelectHandle query_statement_handle(db_connectionHandle_);
  query_statement_handle.execSqlAllowSuccessWithInfo(sql);
}

void OdbcConnectionImpl::runSql(const std::string& sql) {
  OdbcSingleRecordSelectHandle query_statement_handle(db_connectionHandle_);
  query_statement_handle.execDirectSql(sql, true);
}

std::pair<size_t, size_t> OdbcConnectionImpl::getColumnWidthAndBatchSize(
    const OdbcSelectDescriptor& select_desc,
    const int64_t buffer_byte_size,
    const size_t max_buffer_size) {
  auto remote_column_detail = select_desc.remote_column;
  auto column_descriptor = select_desc.column;
  size_t data_size;
  // Note this makes a decision based on the target db column being
  // variable length rather than the source column and assumes we're
  //  map variable length columns to variable length columns
  if (has_variable_length_input(column_descriptor.columnType)) {
    data_size = getMaxStringLength(select_desc);
  } else {
    CHECK_GT(remote_column_detail.odbc_octet_transfer_size, 0);
    data_size = remote_column_detail.odbc_octet_transfer_size;
  }

  size_t batch_size = buffer_byte_size / data_size;
  // for columns with variable length inputs, attempt resizing buffer up to a limit
  if (has_variable_length_input(column_descriptor.columnType) && batch_size < 1 &&
      data_size <= max_buffer_size) {
    batch_size = 1;
  }
  if (batch_size < 1) {
    throw ForeignStorageException(
        "`BUFFER_SIZE` specified is too small to fetch at least one element of required "
        "size " +
        std::to_string(data_size) + " bytes in HeavyDB column '" +
        column_descriptor.columnName +
        "', please specify larger `BUFFER_SIZE` currently it is " +
        std::to_string(buffer_byte_size) + " bytes.");
  }

  batch_size = std::min<size_t>(batch_size, select_desc.limit);

  return {data_size, batch_size};
}

void OdbcConnectionImpl::runSelectCmd(const OdbcSelectDescriptor& select_desc,
                                      size_t buffer_byte_size,
                                      ResultSetProcessor result_set_processor) {
  auto remote_column_descriptor = select_desc.remote_column;
  auto [data_size, batch_size] = getColumnWidthAndBatchSize(
      select_desc, buffer_byte_size, OdbcConnection::max_buffer_resize_limit_);

  CHECK_GT(batch_size, 0UL);
  OdbcMultipleRecordSelectHandle query_statement_handle(db_connectionHandle_, batch_size);

  auto data_buffer_ptr = std::make_unique<int8_t[]>(data_size * batch_size);
  auto strlen_or_indptr = std::make_unique<SQLLEN[]>(batch_size);
  query_statement_handle.bindColumnData(remote_column_descriptor.odbc_base_type,
                                        data_buffer_ptr.get(),
                                        1,
                                        data_size,
                                        strlen_or_indptr.get());

  // We need to be able to make multiple selects (for multiple chunks) and assume we'll
  // get back consistent record sets as in records 0 - 32k, 32k to 64k etc, which is why
  // the limit statements are added.  However, for this to work the base sql command must
  // have an order by on a unique set of fields in the records returned.
  auto augmented_select_cmd = select_desc.getSelectStmtFromType();

  query_statement_handle.setStmtAttrRowStatusPtr(batch_size);
  auto total_num_rows_fetched =
      query_statement_handle.execDirectSqlAndFetchRecordsInBatches(
          augmented_select_cmd,
          result_set_processor,
          data_buffer_ptr.get(),
          strlen_or_indptr.get(),
          data_size,
          select_desc.offset,
          remote_column_descriptor);

  size_t expected_num_rows_fetched = select_desc.getExpectedNumResultRows();
  if (total_num_rows_fetched != expected_num_rows_fetched) {
    throw_unexpected_number_of_items(
        expected_num_rows_fetched, total_num_rows_fetched, "records");
  }
}

namespace {

void set_column_metadata(std::vector<ColumnRemoteData>& column_remote_data_vec,
                         OdbcSingleRecordSelectHandle& single_record_select_handle,
                         SQLSMALLINT num_columns,
                         const std::string& select_command,
                         OdbcConnectionImpl& odbc_connection,
                         size_t row_count,
                         const std::string& dbms_quoted_identifier_string) {
  for (SQLSMALLINT i = 1; i <= num_columns; i++) {
    OdbcColumnDesc column_definition_in_odbc_types =
        single_record_select_handle.describeCol(i);

    auto omnisci_ref_type =
        odbc_to_omnisci_reference_map__.find(column_definition_in_odbc_types.col_type);

    if (column_definition_in_odbc_types.isVariableWidthColumn()) {
      column_definition_in_odbc_types.column_size =
          odbc_connection.getStringWidthForColumn(
              i, select_command, column_definition_in_odbc_types.col_name, row_count);
    }

    auto mapping_found = omnisci_ref_type != odbc_to_omnisci_reference_map__.end();
    if (!mapping_found) {
      const auto& column_name = column_definition_in_odbc_types.col_name;
      throw ForeignStorageException(
          "Column \"" + column_name +
          "\" cannot be mapped to a valid database column type.");
    }
    // TODO the allocation of omnisci_ref_type->second presumes that
    // first value in the returned vector is the correct value.  This may
    // not be the case.
    foreign_storage::RemoteColumnDescription remote_column_definition_omnisci_types =
        omnisci_ref_type->second.front();
    updateRemoteColumnDetails(remote_column_definition_omnisci_types,
                              column_definition_in_odbc_types,
                              dbms_quoted_identifier_string);
    column_remote_data_vec.emplace_back();
    column_remote_data_vec.back().remote_column_description =
        remote_column_definition_omnisci_types;
  }
}

void fetch_rows(const std::string& select_command,
                std::vector<ColumnRemoteData>& column_remote_data_vec,
                size_t row_count,
                OdbcDatabaseHandle& db_connectionHandle) {
  OdbcMultipleRecordSelectHandle multiple_record_select_handle(db_connectionHandle,
                                                               row_count);
  std::vector<std::unique_ptr<SQLLEN[]>> strlen_or_ind_ptrs;
  for (size_t i = 0; i < column_remote_data_vec.size(); i++) {
    auto data_size =
        column_remote_data_vec[i].remote_column_description.odbc_octet_transfer_size;
    auto& data_ptr = column_remote_data_vec[i].data_ptr;
    data_ptr = std::make_unique<int8_t[]>(data_size * row_count);
    strlen_or_ind_ptrs.emplace_back(std::make_unique<SQLLEN[]>(row_count));
    multiple_record_select_handle.bindColumnData(
        column_remote_data_vec[i].remote_column_description.odbc_base_type,
        data_ptr.get(),
        i + 1,
        data_size,
        strlen_or_ind_ptrs.back().get());
  }
  multiple_record_select_handle.setStmtAttrRowStatusPtr(row_count);
  multiple_record_select_handle.execDirectSqlAndFetchRecords(select_command);
  auto num_rows_fetched = multiple_record_select_handle.getNumRowsFetched();

  CHECK_EQ(column_remote_data_vec.size(), strlen_or_ind_ptrs.size());
  for (size_t i = 0; i < column_remote_data_vec.size(); i++) {
    column_remote_data_vec[i].row_count = num_rows_fetched;
    auto& null_or_strlen = column_remote_data_vec[i].null_or_strlen;
    null_or_strlen.resize(num_rows_fetched);
    auto null_or_strlen_ptr = strlen_or_ind_ptrs[i].get();
    std::transform(null_or_strlen_ptr,
                   null_or_strlen_ptr + num_rows_fetched,
                   null_or_strlen.data(),
                   [](const SQLLEN size) -> std::optional<size_t> {
                     if (size == SQL_NULL_DATA) {
                       return std::nullopt;
                     }
                     CHECK_GT(size, static_cast<SQLLEN>(0));
                     return size;
                   });
  }
}
}  // namespace

size_t OdbcConnectionImpl::getStringWidthForColumn(SQLSMALLINT column_index,
                                                   const std::string& select_command,
                                                   const std::string& col_name,
                                                   size_t row_count,
                                                   int64_t row_offset /*0 default*/) {
  ColumnDescriptor column_descriptor;
  RemoteColumnDescription remote_column_descriptor{
      kINT, 0, 0, "not_needed", col_name, -1, false};

  OdbcSelectDescriptor odbc_select_descriptor{row_offset,
                                              row_count,
                                              select_command,
                                              col_name,
                                              column_descriptor,
                                              remote_column_descriptor};

  return getMaxStringLength(odbc_select_descriptor);
}

std::vector<ColumnRemoteData> OdbcConnectionImpl::runDataPreviewMultiColumnSelectCmd(
    const std::string& select_command,
    size_t row_count) {
  CHECK(row_count <= shared::kDefaultSampleRowsCount);
  OdbcSingleRecordSelectHandle single_record_select_handle(db_connectionHandle_);
  single_record_select_handle.prepare(select_command);

  SQLSMALLINT num_columns = 0;
  THROW_STATUS(single_record_select_handle.sqlCallValidation(
      SQLNumResultCols(single_record_select_handle.getRawHandle(), &num_columns)));

  std::vector<ColumnRemoteData> column_remote_data_vec;
  column_remote_data_vec.reserve(num_columns);

  set_column_metadata(column_remote_data_vec,
                      single_record_select_handle,
                      num_columns,
                      select_command,
                      *this,
                      row_count,
                      dbms_quoted_identifier_string_);
  fetch_rows(select_command, column_remote_data_vec, row_count, db_connectionHandle_);
  return column_remote_data_vec;
}

std::set<std::string> OdbcConnectionImpl::getInstalledDrivers() {
  std::set<std::string> drivers;
  constexpr size_t default_string_length{2048};
  std::string drivers_str;
  drivers_str.resize(default_string_length);
  WORD actual_length{0};
  if (SQLGetInstalledDrivers(drivers_str.data(), drivers_str.length(), &actual_length)) {
    const char null_terminator{0};
    // Remove trailing null terminators
    while (actual_length > 0 && drivers_str[actual_length - 1] == null_terminator) {
      actual_length--;
    }
    if (actual_length < drivers_str.length()) {
      drivers_str.resize(actual_length);
    }
    size_t start_pos{0};
    while (start_pos < drivers_str.length()) {
      auto found_pos = drivers_str.find(null_terminator, start_pos);
      if (found_pos != std::string::npos) {
        drivers.emplace(drivers_str.substr(start_pos, found_pos - start_pos));
        start_pos = found_pos + 1;
      } else {
        drivers.emplace(drivers_str.substr(start_pos, drivers_str.length() - start_pos));
        start_pos = drivers_str.length();
      }
    }
  } else {
    std::string error_message{
        "An error occurred while attempting to get installed ODBC drivers."};
    WORD i_error{0};
    DWORD pf_error_code{0};
    LPSTR lpsz_error_msg{nullptr};
    WORD pcb_error_msg{0};
    const auto response = SQLInstallerError(
        i_error, &pf_error_code, lpsz_error_msg, SQL_MAX_MESSAGE_LENGTH, &pcb_error_msg);
    if (response == SQL_SUCCESS && lpsz_error_msg && pcb_error_msg > 0) {
      error_message += " Error: " + std::string(lpsz_error_msg, pcb_error_msg);
    }
    throw foreign_storage::OdbcDriversConfigException{error_message};
  }
  return drivers;
}

namespace {
int version_token_to_int(const std::string& token) {
  int result;
  try {
    result = stoi(token);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in version_token_to_int() : " << e.what();
    throw ForeignStorageException{"Invalid version text returned by remote database."};
  }
  return result;
}

void pad_dbms_vers_with_zeroes(std::vector<std::string>& dbms_vers) {
  while (dbms_vers.size() < 3) {
    dbms_vers.emplace_back("0");
  }
}

void validate_dbms_version(const std::string& dbms_name,
                           const std::string& required_dbms_vers,
                           const std::string& detected_dbms_vers) {
  std::vector<std::string> tokenized_required_dbms_vers, tokenized_detected_dbms_vers;
  boost::split(tokenized_required_dbms_vers, required_dbms_vers, boost::is_any_of(". "));
  boost::split(tokenized_detected_dbms_vers, detected_dbms_vers, boost::is_any_of(". "));
  pad_dbms_vers_with_zeroes(tokenized_required_dbms_vers);
  pad_dbms_vers_with_zeroes(tokenized_detected_dbms_vers);

  size_t i = 0;
  while (i < 3) {
    int required = version_token_to_int(tokenized_required_dbms_vers[i]);
    int detected = version_token_to_int(tokenized_detected_dbms_vers[i]);
    if (detected < required) {
      throw ForeignStorageException{
          "Unsupported version of " + dbms_name + " detected. Only " + dbms_name + " " +
          required_dbms_vers + " or newer is currently supported."};
    } else if (detected > required) {
      break;
    } else {
      i++;
    }
  }
}
}  // namespace

void OdbcConnectionImpl::validateDbmsCompatible() {
  if (dbms_type_name_ == "Hive") {
    validate_dbms_version(dbms_type_name_, "3.0.0", dbms_type_vers_);
  }
}
/*
 *  Static entry point.
 */
std::unique_ptr<OdbcConnection> OdbcConnection::create(
    const OdbcConnectionInfo& connection_info,
    const UserMapping* user_mapping) {
  return std::make_unique<OdbcConnectionImpl>(connection_info, user_mapping);
}
OdbcConnection::~OdbcConnection() {}

std::set<std::string> OdbcConnection::getInstalledDrivers() {
  return OdbcConnectionImpl::getInstalledDrivers();
}

size_t OdbcConnection::max_buffer_resize_limit_ =
    import_export::max_import_buffer_resize_byte_size;

}  // namespace foreign_storage
