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

#include "Catalog/Catalog.h"
#include "Shared/sqltypes.h"

namespace ddl_utils {
class SqlType {
 public:
  /**
   * Encapsulates column definition type information.
   *
   * @param type - Column type.
   * @param param1 - For column types followed by parenthesis with more information,
   * this represents the first parameter in the parenthesis. For example, in
   * DECIMAL(5, 2), this would represent the precision value of 5. In GEOMETRY(POINT,
   * 4326), this would represent the integer value of the kPOINT enum.
   * @param param2 - For column types followed by parenthesis with more information,
   * this represents the second parameter in the parenthesis. For example, in
   * DECIMAL(5, 2), this would represent the scale value of 2. In GEOMETRY(POINT, 4326),
   * this would represent the coordinate type integer value of 4326.
   * @param is_array - Flag that indicates whether or not column type is an array.
   * @param array_size - For array column types, this is the specified size of the array.
   */
  SqlType(SQLTypes type, int param1, int param2, bool is_array, int array_size);

  virtual SQLTypes get_type() const;
  virtual int get_param1() const;
  virtual void set_param1(int param);
  virtual int get_param2() const;
  virtual bool get_is_array() const;
  virtual void set_is_array(bool a);
  virtual int get_array_size() const;
  virtual void set_array_size(int s);
  virtual std::string to_string() const;
  virtual void check_type();

 protected:
  SQLTypes type;
  int param1;  // e.g. for NUMERIC(10).  -1 means unspecified.
  int param2;  // e.g. for NUMERIC(10,3). 0 is default value.
  bool is_array;
  int array_size;
};

class Encoding {
 public:
  /**
   * Encapsulates column definition encoding information.
   *
   * @param encoding_name - Type of encoding. For example, "DICT", "FIXED", etc.
   * @param encoding_param - Encoding size.
   */
  Encoding(std::string* encoding_name, int encoding_param);
  virtual ~Encoding() {}

  virtual const std::string* get_encoding_name() const;
  virtual int get_encoding_param() const;

 protected:
  std::unique_ptr<std::string> encoding_name;
  int encoding_param;
};

enum class DataTransferType { IMPORT = 1, EXPORT };

class FilePathWhitelist {
 public:
  static void initialize(const std::string& data_dir,
                         const std::string& allowed_import_paths,
                         const std::string& allowed_export_paths);
  static void validateWhitelistedFilePath(
      const std::vector<std::string>& expanded_file_paths,
      const DataTransferType data_transfer_type);
  static void clear();

 private:
  static std::vector<std::string> whitelisted_import_paths_;
  static std::vector<std::string> whitelisted_export_paths_;
};

class FilePathBlacklist {
 public:
  static void addToBlacklist(const std::string& path);
  static bool isBlacklistedPath(const std::string& path);
  static void clear();

 private:
  static std::vector<std::string> blacklisted_paths_;
};

enum class TableType { TABLE = 1, VIEW };

void set_default_encoding(ColumnDescriptor& cd);

void validate_and_set_fixed_encoding(ColumnDescriptor& cd,
                                     int encoding_size,
                                     const SqlType* column_type);

void validate_and_set_dictionary_encoding(ColumnDescriptor& cd, int encoding_size);

void validate_and_set_none_encoding(ColumnDescriptor& cd);

void validate_and_set_sparse_encoding(ColumnDescriptor& cd, int encoding_size);

void validate_and_set_compressed_encoding(ColumnDescriptor& cd, int encoding_size);

void validate_and_set_date_encoding(ColumnDescriptor& cd, int encoding_size);

void validate_and_set_encoding(ColumnDescriptor& cd,
                               const Encoding* encoding,
                               const SqlType* column_type);

void validate_and_set_type(ColumnDescriptor& cd, SqlType* column_type);

void validate_and_set_array_size(ColumnDescriptor& cd, const SqlType* column_type);

void validate_and_set_default_value(ColumnDescriptor& cd,
                                    const std::string* default_value,
                                    bool not_null);

void set_column_descriptor(const std::string& column_name,
                           ColumnDescriptor& cd,
                           SqlType* column_type,
                           const bool not_null,
                           const Encoding* encoding,
                           const std::string* default_value);

void set_default_table_attributes(const std::string& table_name,
                                  TableDescriptor& td,
                                  const int32_t column_count);

void validate_non_duplicate_column(const std::string& column_name,
                                   std::unordered_set<std::string>& upper_column_names);

void validate_non_reserved_keyword(const std::string& column_name);

void validate_table_type(const TableDescriptor* td,
                         const TableType expected_table_type,
                         const std::string& command);

std::string table_type_enum_to_string(const TableType table_type);

/**
 * Validates that the given file path is allowed. Validation entails ensuring
 * that given path is not under a blacklisted root path and path is under a
 * whitelisted path, if whitelisted paths have been configured.
 * Also excludes the use of spaces and punctuation other than: . / _ + - = :
 *
 * @param file_path - file path to validate
 * @param data_transfer_type - enum indicating whether validation is for an import or
 * export use case
 * @param allow_wildcards - bool indicating if wildcards are allowed
 */
void validate_allowed_file_path(const std::string& file_path,
                                const DataTransferType data_transfer_type,
                                const bool allow_wildcards = false);
}  // namespace ddl_utils
