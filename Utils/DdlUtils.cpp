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

#include "DdlUtils.h"

#include <unordered_set>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "rapidjson/document.h"

#include "Fragmenter/FragmentDefaultValues.h"
#include "Geospatial/Types.h"
#include "Parser/ReservedKeywords.h"
#include "Shared/file_glob.h"
#include "Shared/misc.h"
#include "Shared/sqltypes.h"

bool g_use_date_in_days_default_encoding{true};

namespace ddl_utils {
SqlType::SqlType(SQLTypes type, int param1, int param2, bool is_array, int array_size)
    : type(type)
    , param1(param1)
    , param2(param2)
    , is_array(is_array)
    , array_size(array_size) {}

SQLTypes SqlType::get_type() const {
  return type;
}

int SqlType::get_param1() const {
  return param1;
}

void SqlType::set_param1(int param) {
  param1 = param;
}

int SqlType::get_param2() const {
  return param2;
}

bool SqlType::get_is_array() const {
  return is_array;
}

void SqlType::set_is_array(bool a) {
  is_array = a;
}

int SqlType::get_array_size() const {
  return array_size;
}

void SqlType::set_array_size(int s) {
  array_size = s;
}

std::string SqlType::to_string() const {
  std::string str;
  switch (type) {
    case kBOOLEAN:
      str = "BOOLEAN";
      break;
    case kCHAR:
      str = "CHAR(" + boost::lexical_cast<std::string>(param1) + ")";
      break;
    case kVARCHAR:
      str = "VARCHAR(" + boost::lexical_cast<std::string>(param1) + ")";
      break;
    case kTEXT:
      str = "TEXT";
      break;
    case kNUMERIC:
      str = "NUMERIC(" + boost::lexical_cast<std::string>(param1);
      if (param2 > 0) {
        str += ", " + boost::lexical_cast<std::string>(param2);
      }
      str += ")";
      break;
    case kDECIMAL:
      str = "DECIMAL(" + boost::lexical_cast<std::string>(param1);
      if (param2 > 0) {
        str += ", " + boost::lexical_cast<std::string>(param2);
      }
      str += ")";
      break;
    case kBIGINT:
      str = "BIGINT";
      break;
    case kINT:
      str = "INT";
      break;
    case kTINYINT:
      str = "TINYINT";
      break;
    case kSMALLINT:
      str = "SMALLINT";
      break;
    case kFLOAT:
      str = "FLOAT";
      break;
    case kDOUBLE:
      str = "DOUBLE";
      break;
    case kTIME:
      str = "TIME";
      if (param1 < 6) {
        str += "(" + boost::lexical_cast<std::string>(param1) + ")";
      }
      break;
    case kTIMESTAMP:
      str = "TIMESTAMP";
      if (param1 <= 9) {
        str += "(" + boost::lexical_cast<std::string>(param1) + ")";
      }
      break;
    case kDATE:
      str = "DATE";
      break;
    default:
      assert(false);
      break;
  }
  if (is_array) {
    str += "[";
    if (array_size > 0) {
      str += boost::lexical_cast<std::string>(array_size);
    }
    str += "]";
  }
  return str;
}

void SqlType::check_type() {
  switch (type) {
    case kCHAR:
    case kVARCHAR:
      if (param1 <= 0) {
        throw std::runtime_error("CHAR and VARCHAR must have a positive dimension.");
      }
      break;
    case kDECIMAL:
    case kNUMERIC:
      if (param1 <= 0) {
        throw std::runtime_error("DECIMAL and NUMERIC must have a positive precision.");
      } else if (param1 > 19) {
        throw std::runtime_error(
            "DECIMAL and NUMERIC precision cannot be larger than 19.");
      } else if (param1 <= param2) {
        throw std::runtime_error(
            "DECIMAL and NUMERIC must have precision larger than scale.");
      }
      break;
    case kTIMESTAMP:
      if (param1 == -1) {
        param1 = 0;  // set default to 0
      } else if (param1 != 0 && param1 != 3 && param1 != 6 &&
                 param1 != 9) {  // support ms, us, ns
        throw std::runtime_error(
            "Only TIMESTAMP(n) where n = (0,3,6,9) are supported now.");
      }
      break;
    case kTIME:
      if (param1 == -1) {
        param1 = 0;  // default precision is 0
      }
      if (param1 > 0) {  // @TODO(wei) support sub-second precision later.
        throw std::runtime_error("Only TIME(0) is supported now.");
      }
      break;
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      // Storing SRID in param1
      break;
    default:
      param1 = 0;
      break;
  }
}

Encoding::Encoding(std::string* encoding_name, int encoding_param)
    : encoding_name(encoding_name), encoding_param(encoding_param) {}

const std::string* Encoding::get_encoding_name() const {
  return encoding_name.get();
}

int Encoding::get_encoding_param() const {
  return encoding_param;
}

void set_default_encoding(ColumnDescriptor& cd) {
  // Change default TEXT column behaviour to be DICT encoded
  if (cd.columnType.is_string() || cd.columnType.is_string_array()) {
    // default to 32-bits
    cd.columnType.set_compression(kENCODING_DICT);
    cd.columnType.set_comp_param(32);
  } else if (cd.columnType.is_decimal() && cd.columnType.get_precision() <= 4) {
    cd.columnType.set_compression(kENCODING_FIXED);
    cd.columnType.set_comp_param(16);
  } else if (cd.columnType.is_decimal() && cd.columnType.get_precision() <= 9) {
    cd.columnType.set_compression(kENCODING_FIXED);
    cd.columnType.set_comp_param(32);
  } else if (cd.columnType.is_decimal() && cd.columnType.get_precision() > 18) {
    throw std::runtime_error(cd.columnName + ": Precision too high, max 18.");
  } else if (cd.columnType.is_geometry() && cd.columnType.get_output_srid() == 4326) {
    // default to GEOINT 32-bits
    cd.columnType.set_compression(kENCODING_GEOINT);
    cd.columnType.set_comp_param(32);
  } else if (cd.columnType.get_type() == kDATE && g_use_date_in_days_default_encoding) {
    // Days encoding for DATE
    cd.columnType.set_compression(kENCODING_DATE_IN_DAYS);
    cd.columnType.set_comp_param(0);
  } else {
    cd.columnType.set_compression(kENCODING_NONE);
    cd.columnType.set_comp_param(0);
  }
}

void validate_and_set_fixed_encoding(ColumnDescriptor& cd,
                                     int encoding_size,
                                     const SqlType* column_type) {
  auto type = cd.columnType.get_type();
  // fixed-bits encoding
  if (type == kARRAY) {
    type = cd.columnType.get_subtype();
    switch (type) {
      case kTINYINT:
      case kSMALLINT:
      case kINT:
      case kBIGINT:
      case kDATE:
        throw std::runtime_error(cd.columnName + ": Cannot apply FIXED encoding to " +
                                 column_type->to_string() + " type array.");
        break;
      default:
        break;
    }
  }

  if (!IS_INTEGER(type) && !is_datetime(type) &&
      !(type == kDECIMAL || type == kNUMERIC)) {
    throw std::runtime_error(
        cd.columnName +
        ": Fixed encoding is only supported for integer or time columns.");
  }

  switch (type) {
    case kSMALLINT:
      if (encoding_size != 8) {
        throw std::runtime_error(
            cd.columnName +
            ": Compression parameter for Fixed encoding on SMALLINT must be 8.");
      }
      break;
    case kINT:
      if (encoding_size != 8 && encoding_size != 16) {
        throw std::runtime_error(
            cd.columnName +
            ": Compression parameter for Fixed encoding on INTEGER must be 8 or 16.");
      }
      break;
    case kBIGINT:
      if (encoding_size != 8 && encoding_size != 16 && encoding_size != 32) {
        throw std::runtime_error(cd.columnName +
                                 ": Compression parameter for Fixed encoding on "
                                 "BIGINT must be 8 or 16 or 32.");
      }
      break;
    case kTIMESTAMP:
    case kTIME:
      if (encoding_size != 32) {
        throw std::runtime_error(cd.columnName +
                                 ": Compression parameter for Fixed encoding on "
                                 "TIME or TIMESTAMP must be 32.");
      } else if (cd.columnType.is_high_precision_timestamp()) {
        throw std::runtime_error("Fixed encoding is not supported for TIMESTAMP(3|6|9).");
      }
      break;
    case kDECIMAL:
    case kNUMERIC:
      if (encoding_size != 32 && encoding_size != 16) {
        throw std::runtime_error(cd.columnName +
                                 ": Compression parameter for Fixed encoding on "
                                 "DECIMAL must be 16 or 32.");
      }

      if (encoding_size == 32 && cd.columnType.get_precision() > 9) {
        throw std::runtime_error(cd.columnName +
                                 ": Precision too high for Fixed(32) encoding, max 9.");
      }

      if (encoding_size == 16 && cd.columnType.get_precision() > 4) {
        throw std::runtime_error(cd.columnName +
                                 ": Precision too high for Fixed(16) encoding, max 4.");
      }
      break;
    case kDATE:
      if (encoding_size != 32 && encoding_size != 16) {
        throw std::runtime_error(cd.columnName +
                                 ": Compression parameter for Fixed encoding on "
                                 "DATE must be 16 or 32.");
      }
      break;
    default:
      throw std::runtime_error(cd.columnName + ": Cannot apply FIXED encoding to " +
                               column_type->to_string());
  }
  if (type == kDATE) {
    cd.columnType.set_compression(kENCODING_DATE_IN_DAYS);
    cd.columnType.set_comp_param(16);
  } else {
    cd.columnType.set_compression(kENCODING_FIXED);
    cd.columnType.set_comp_param(encoding_size);
  }
}

void validate_and_set_dictionary_encoding(ColumnDescriptor& cd, int encoding_size) {
  if (!cd.columnType.is_string() && !cd.columnType.is_string_array()) {
    throw std::runtime_error(
        cd.columnName +
        ": Dictionary encoding is only supported on string or string array columns.");
  }
  int comp_param;
  if (encoding_size == 0) {
    comp_param = 32;  // default to 32-bits
  } else {
    comp_param = encoding_size;
  }
  if (cd.columnType.is_string_array() && comp_param != 32) {
    throw std::runtime_error(cd.columnName +
                             ": Compression parameter for string arrays must be 32");
  }
  if (comp_param != 8 && comp_param != 16 && comp_param != 32) {
    throw std::runtime_error(
        cd.columnName +
        ": Compression parameter for Dictionary encoding must be 8 or 16 or 32.");
  }
  // dictionary encoding
  cd.columnType.set_compression(kENCODING_DICT);
  cd.columnType.set_comp_param(comp_param);
}

void validate_and_set_none_encoding(ColumnDescriptor& cd) {
  if (!cd.columnType.is_string() && !cd.columnType.is_string_array() &&
      !cd.columnType.is_geometry()) {
    throw std::runtime_error(
        cd.columnName +
        ": None encoding is only supported on string, string array, or geo columns.");
  }
  cd.columnType.set_compression(kENCODING_NONE);
  cd.columnType.set_comp_param(0);
}

void validate_and_set_sparse_encoding(ColumnDescriptor& cd, int encoding_size) {
  // sparse column encoding with mostly NULL values
  if (cd.columnType.get_notnull()) {
    throw std::runtime_error(cd.columnName +
                             ": Cannot do sparse column encoding on a NOT NULL column.");
  }
  if (encoding_size == 0 || encoding_size % 8 != 0 || encoding_size > 48) {
    throw std::runtime_error(
        cd.columnName +
        "Must specify number of bits as 8, 16, 24, 32 or 48 as the parameter to "
        "sparse-column encoding.");
  }
  cd.columnType.set_compression(kENCODING_SPARSE);
  cd.columnType.set_comp_param(encoding_size);
  // throw std::runtime_error("SPARSE encoding not supported yet.");
}

void validate_and_set_compressed_encoding(ColumnDescriptor& cd, int encoding_size) {
  if (!cd.columnType.is_geometry() || cd.columnType.get_output_srid() != 4326) {
    throw std::runtime_error(
        cd.columnName + ": COMPRESSED encoding is only supported on WGS84 geo columns.");
  }
  int comp_param;
  if (encoding_size == 0) {
    comp_param = 32;  // default to 32-bits
  } else {
    comp_param = encoding_size;
  }
  if (comp_param != 32) {
    throw std::runtime_error(cd.columnName +
                             ": only 32-bit COMPRESSED geo encoding is supported");
  }
  // encoding longitude/latitude as integers
  cd.columnType.set_compression(kENCODING_GEOINT);
  cd.columnType.set_comp_param(comp_param);
}

void validate_and_set_date_encoding(ColumnDescriptor& cd, int encoding_size) {
  // days encoding for dates
  if (cd.columnType.get_type() == kARRAY && cd.columnType.get_subtype() == kDATE) {
    throw std::runtime_error(cd.columnName +
                             ": Cannot apply days encoding to date array.");
  }
  if (cd.columnType.get_type() != kDATE) {
    throw std::runtime_error(cd.columnName +
                             ": Days encoding is only supported for DATE columns.");
  }
  if (encoding_size != 32 && encoding_size != 16) {
    throw std::runtime_error(cd.columnName +
                             ": Compression parameter for Days encoding on "
                             "DATE must be 16 or 32.");
  }
  cd.columnType.set_compression(kENCODING_DATE_IN_DAYS);
  cd.columnType.set_comp_param((encoding_size == 16) ? 16 : 0);
}

void validate_and_set_encoding(ColumnDescriptor& cd,
                               const Encoding* encoding,
                               const SqlType* column_type) {
  if (encoding == nullptr) {
    set_default_encoding(cd);
  } else {
    const std::string& comp = *encoding->get_encoding_name();
    if (boost::iequals(comp, "fixed")) {
      validate_and_set_fixed_encoding(cd, encoding->get_encoding_param(), column_type);
    } else if (boost::iequals(comp, "rl")) {
      // run length encoding
      cd.columnType.set_compression(kENCODING_RL);
      cd.columnType.set_comp_param(0);
      // throw std::runtime_error("RL(Run Length) encoding not supported yet.");
    } else if (boost::iequals(comp, "diff")) {
      // differential encoding
      cd.columnType.set_compression(kENCODING_DIFF);
      cd.columnType.set_comp_param(0);
      // throw std::runtime_error("DIFF(differential) encoding not supported yet.");
    } else if (boost::iequals(comp, "dict")) {
      validate_and_set_dictionary_encoding(cd, encoding->get_encoding_param());
    } else if (boost::iequals(comp, "NONE")) {
      validate_and_set_none_encoding(cd);
    } else if (boost::iequals(comp, "sparse")) {
      validate_and_set_sparse_encoding(cd, encoding->get_encoding_param());
    } else if (boost::iequals(comp, "compressed")) {
      validate_and_set_compressed_encoding(cd, encoding->get_encoding_param());
    } else if (boost::iequals(comp, "days")) {
      validate_and_set_date_encoding(cd, encoding->get_encoding_param());
    } else {
      throw std::runtime_error(cd.columnName + ": Invalid column compression scheme " +
                               comp);
    }
  }
}

void validate_and_set_type(ColumnDescriptor& cd, SqlType* column_type) {
  column_type->check_type();

  if (column_type->get_is_array()) {
    cd.columnType.set_type(kARRAY);
    cd.columnType.set_subtype(column_type->get_type());
  } else {
    cd.columnType.set_type(column_type->get_type());
  }
  if (IS_GEO(column_type->get_type())) {
    cd.columnType.set_subtype(static_cast<SQLTypes>(column_type->get_param1()));
    cd.columnType.set_input_srid(column_type->get_param2());
    cd.columnType.set_output_srid(column_type->get_param2());
  } else {
    cd.columnType.set_dimension(column_type->get_param1());
    cd.columnType.set_scale(column_type->get_param2());
  }
}

void validate_and_set_array_size(ColumnDescriptor& cd, const SqlType* column_type) {
  if (cd.columnType.is_string_array() &&
      cd.columnType.get_compression() != kENCODING_DICT) {
    throw std::runtime_error(
        cd.columnName +
        ": Array of strings must be dictionary encoded. Specify ENCODING DICT");
  }

  if (column_type->get_is_array()) {
    int s = -1;
    auto array_size = column_type->get_array_size();
    if (array_size > 0) {
      auto sti = cd.columnType.get_elem_type();
      s = array_size * sti.get_size();
      if (s <= 0) {
        throw std::runtime_error(cd.columnName + ": Unexpected fixed length array size");
      }
    }
    cd.columnType.set_size(s);

  } else {
    cd.columnType.set_fixed_size();
  }
}

namespace {

void validate_literal(const std::string& val,
                      SQLTypeInfo column_type,
                      const std::string& column_name) {
  if (to_upper(val) == "NULL") {
    return;
  }
  switch (column_type.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kFLOAT:
    case kDOUBLE:
    case kTIME:
    case kTIMESTAMP:
      StringToDatum(val, column_type);
      break;
    case kDATE: {
      auto d = StringToDatum(val, column_type);
      DateDaysOverflowValidator validator(column_type);
      validator.validate(d.bigintval);
      break;
    }
    case kDECIMAL:
    case kNUMERIC: {
      SQLTypeInfo ti(kNUMERIC, 0, 0, false);
      auto d = StringToDatum(val, ti);
      auto converted_val = convert_decimal_value_to_scale(d.bigintval, ti, column_type);
      DecimalOverflowValidator validator(column_type);
      validator.validate(converted_val);
      break;
    }
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (val.length() > StringDictionary::MAX_STRLEN) {
        throw std::runtime_error("String too long for column " + column_name + " was " +
                                 std::to_string(val.length()) + " max is " +
                                 std::to_string(StringDictionary::MAX_STRLEN));
      }
      break;
    case kARRAY: {
      if (val.front() != '{' || val.back() != '}') {
        throw std::runtime_error(column_name +
                                 ": arrays should start and end with curly braces");
      }
      std::vector<std::string> elements = split(val.substr(1, val.length() - 2), ", ");
      if (column_type.get_size() > 0) {
        auto sti = column_type.get_elem_type();
        size_t expected_size = column_type.get_size() / sti.get_size();
        size_t actual_size = elements.size();
        if (actual_size != expected_size) {
          throw std::runtime_error("Fixed length array column " + column_name +
                                   " expects " + std::to_string(expected_size) +
                                   " values, received " + std::to_string(actual_size));
        }
      }
      SQLTypeInfo element_ti = column_type.get_elem_type();
      for (const auto& element : elements) {
        if (to_upper(element) != "NULL") {
          validate_literal(element, element_ti, column_name);
        }
      }
      break;
    }
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      if (val.empty()) {
        return;
      }
      try {
        auto geo = Geospatial::GeoTypesFactory::createGeoType(val);
        if (!geo) {
          throw std::runtime_error("Unexpected geo literal '" + val + "' for column " +
                                   column_name);
        }
        if (!geo->transform(column_type)) {
          throw std::runtime_error("Cannot transform SRID for literal '" + val +
                                   "' for column " + column_name);
        } else {
          auto sql_type = column_type.get_type();
          auto geo_type = geo->getType();
          if ((geo_type == Geospatial::GeoBase::GeoType::kPOINT && sql_type != kPOINT) ||
              (geo_type == Geospatial::GeoBase::GeoType::kLINESTRING &&
               sql_type != kLINESTRING) ||
              (geo_type == Geospatial::GeoBase::GeoType::kPOLYGON &&
               sql_type != kPOLYGON) ||
              (geo_type == Geospatial::GeoBase::GeoType::kMULTIPOLYGON &&
               sql_type != kMULTIPOLYGON)) {
            throw std::runtime_error("Geo literal '" + val +
                                     "' doesn't match the type "
                                     "of column column " +
                                     column_name);
          }
        }
      } catch (Geospatial::GeoTypesError& e) {
        throw std::runtime_error("Unexpected geo literal '" + val + "' for column " +
                                 column_name + ": " + e.what());
      }
      break;
    default:
      CHECK(false) << "validate_literal() does not support type "
                   << column_type.get_type();
  }
}

}  // namespace

void validate_and_set_default_value(ColumnDescriptor& cd,
                                    const std::string* default_value,
                                    bool not_null) {
  bool is_null_literal =
      default_value && ((to_upper(*default_value) == "NULL") ||
                        (cd.columnType.is_geometry() && default_value->empty()));
  if (not_null && (is_null_literal)) {
    throw std::runtime_error(cd.columnName +
                             ": cannot set default value to NULL for "
                             "NOT NULL column");
  }
  if (!default_value || is_null_literal) {
    cd.default_value = std::nullopt;
    return;
  }
  const auto& column_type = cd.columnType;
  const auto& val = *default_value;
  validate_literal(val, column_type, cd.columnName);
  cd.default_value = std::make_optional(*default_value);
}

void set_column_descriptor(const std::string& column_name,
                           ColumnDescriptor& cd,
                           SqlType* column_type,
                           const bool not_null,
                           const Encoding* encoding,
                           const std::string* default_value) {
  cd.columnName = column_name;
  validate_and_set_type(cd, column_type);
  cd.columnType.set_notnull(not_null);
  validate_and_set_encoding(cd, encoding, column_type);
  validate_and_set_array_size(cd, column_type);
  cd.isSystemCol = false;
  cd.isVirtualCol = false;
  validate_and_set_default_value(cd, default_value, not_null);
}

void set_default_table_attributes(const std::string& table_name,
                                  TableDescriptor& td,
                                  const int32_t column_count) {
  td.tableName = table_name;
  td.nColumns = column_count;
  td.isView = false;
  td.fragmenter = nullptr;
  td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
  td.maxFragRows = DEFAULT_FRAGMENT_ROWS;
  td.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;
  td.fragPageSize = DEFAULT_PAGE_SIZE;
  td.maxRows = DEFAULT_MAX_ROWS;
}

void validate_non_duplicate_column(const std::string& column_name,
                                   std::unordered_set<std::string>& upper_column_names) {
  const auto upper_column_name = boost::to_upper_copy<std::string>(column_name);
  const auto insert_it = upper_column_names.insert(upper_column_name);
  if (!insert_it.second) {
    throw std::runtime_error("Column '" + column_name + "' defined more than once");
  }
}

void validate_non_reserved_keyword(const std::string& column_name) {
  const auto upper_column_name = boost::to_upper_copy<std::string>(column_name);
  if (reserved_keywords.find(upper_column_name) != reserved_keywords.end()) {
    throw std::runtime_error("Cannot create column with reserved keyword '" +
                             column_name + "'");
  }
}

void validate_table_type(const TableDescriptor* td,
                         const TableType expected_table_type,
                         const std::string& command) {
  if (td->isView) {
    if (expected_table_type != TableType::VIEW) {
      throw std::runtime_error(td->tableName + " is a view. Use " + command + " VIEW.");
    }
  } else if (td->storageType == StorageType::FOREIGN_TABLE) {
    if (expected_table_type != TableType::FOREIGN_TABLE) {
      throw std::runtime_error(td->tableName + " is a foreign table. Use " + command +
                               " FOREIGN TABLE.");
    }
  } else if (expected_table_type != TableType::TABLE) {
    throw std::runtime_error(td->tableName + " is a table. Use " + command + " TABLE.");
  }
}

std::string table_type_enum_to_string(const TableType table_type) {
  if (table_type == ddl_utils::TableType::TABLE) {
    return "Table";
  }
  if (table_type == ddl_utils::TableType::FOREIGN_TABLE) {
    return "ForeignTable";
  }
  if (table_type == ddl_utils::TableType::VIEW) {
    return "View";
  }
  throw std::runtime_error{"Unexpected table type"};
}

std::string get_malformed_config_error_message(const std::string& config_key) {
  return "Configuration value for \"" + config_key +
         "\" is malformed. Value should be a list of paths with format: [ "
         "\"root-path-1\", \"root-path-2\", ... ]";
}

void validate_expanded_file_path(const std::string& file_path,
                                 const std::vector<std::string>& whitelisted_root_paths) {
  const auto& canonical_file_path = boost::filesystem::canonical(file_path);
  for (const auto& root_path : whitelisted_root_paths) {
    if (boost::istarts_with(canonical_file_path.string(), root_path)) {
      return;
    }
  }
  if (canonical_file_path == boost::filesystem::absolute(file_path)) {
    throw std::runtime_error{"File or directory path \"" + file_path +
                             "\" is not whitelisted."};
  }
  throw std::runtime_error{"File or directory path \"" + file_path +
                           "\" (resolved to \"" + canonical_file_path.string() +
                           "\") is not whitelisted."};
}

std::vector<std::string> get_expanded_file_paths(
    const std::string& file_path,
    const DataTransferType data_transfer_type) {
  std::vector<std::string> file_paths;
  if (data_transfer_type == DataTransferType::IMPORT) {
    std::set<std::string> file_paths_set;
    file_paths_set = shared::glob_local_recursive_files(file_path);
    file_paths = std::vector<std::string>(file_paths_set.begin(), file_paths_set.end());
  } else {
    std::string path;
    if (!boost::filesystem::exists(file_path)) {
      // For exports, it is possible to provide a path to a new (nonexistent) file. In
      // this case, validate using the parent path.
      path = boost::filesystem::path(file_path).parent_path().string();
      if (!boost::filesystem::exists(path)) {
        throw std::runtime_error{"File or directory \"" + file_path +
                                 "\" does not exist."};
      }
    } else {
      path = file_path;
    }
    file_paths = {path};
  }
  return file_paths;
}

void validate_allowed_file_path(const std::string& file_path,
                                const DataTransferType data_transfer_type,
                                const bool allow_wildcards) {
  // Reject any punctuation characters except for a few safe ones.
  // Some punctuation characters present a security risk when passed
  // to subprocesses. Don't change this without a security review.
  static const std::string safe_punctuation{"./_+-=:~"};
  for (const auto& ch : file_path) {
    if (std::ispunct(ch) && safe_punctuation.find(ch) == std::string::npos &&
        !(allow_wildcards && ch == '*')) {
      throw std::runtime_error(std::string("Punctuation \"") + ch +
                               "\" is not allowed in file path: " + file_path);
    }
  }

  // Enforce our whitelist and blacklist for file paths.
  const auto& expanded_file_paths =
      get_expanded_file_paths(file_path, data_transfer_type);
  for (const auto& path : expanded_file_paths) {
    if (FilePathBlacklist::isBlacklistedPath(path)) {
      const auto& canonical_file_path = boost::filesystem::canonical(file_path);
      if (canonical_file_path == boost::filesystem::absolute(file_path)) {
        throw std::runtime_error{"Access to file or directory path \"" + file_path +
                                 "\" is not allowed."};
      }
      throw std::runtime_error{"Access to file or directory path \"" + file_path +
                               "\" (resolved to \"" + canonical_file_path.string() +
                               "\") is not allowed."};
    }
  }
  FilePathWhitelist::validateWhitelistedFilePath(expanded_file_paths, data_transfer_type);
}

void set_whitelisted_paths(const std::string& config_key,
                           const std::string& config_value,
                           std::vector<std::string>& whitelisted_paths) {
  rapidjson::Document whitelisted_root_paths;
  whitelisted_root_paths.Parse(config_value);
  if (!whitelisted_root_paths.IsArray()) {
    throw std::runtime_error{get_malformed_config_error_message(config_key)};
  }
  for (const auto& root_path : whitelisted_root_paths.GetArray()) {
    if (!root_path.IsString()) {
      throw std::runtime_error{get_malformed_config_error_message(config_key)};
    }
    if (!boost::filesystem::exists(root_path.GetString())) {
      throw std::runtime_error{"Whitelisted root path \"" +
                               std::string{root_path.GetString()} + "\" does not exist."};
    }
    whitelisted_paths.emplace_back(
        boost::filesystem::canonical(root_path.GetString()).string());
  }
  LOG(INFO) << "Parsed " << config_key << ": "
            << shared::printContainer(whitelisted_paths);
}

void FilePathWhitelist::initialize(const std::string& data_dir,
                                   const std::string& allowed_import_paths,
                                   const std::string& allowed_export_paths) {
  CHECK(!data_dir.empty());
  CHECK(boost::filesystem::is_directory(data_dir));

  auto data_dir_path = boost::filesystem::canonical(data_dir);
  CHECK(whitelisted_import_paths_.empty());
  whitelisted_import_paths_.emplace_back((data_dir_path / "mapd_import").string());

  CHECK(whitelisted_export_paths_.empty());
  whitelisted_export_paths_.emplace_back((data_dir_path / "mapd_export").string());

  if (!allowed_import_paths.empty()) {
    set_whitelisted_paths(
        "allowed-import-paths", allowed_import_paths, whitelisted_import_paths_);
  }
  if (!allowed_export_paths.empty()) {
    set_whitelisted_paths(
        "allowed-export-paths", allowed_export_paths, whitelisted_export_paths_);
  }
}

void FilePathWhitelist::validateWhitelistedFilePath(
    const std::vector<std::string>& expanded_file_paths,
    const DataTransferType data_transfer_type) {
  for (const auto& path : expanded_file_paths) {
    if (data_transfer_type == DataTransferType::IMPORT) {
      validate_expanded_file_path(path, whitelisted_import_paths_);
    } else if (data_transfer_type == DataTransferType::EXPORT) {
      validate_expanded_file_path(path, whitelisted_export_paths_);
    } else {
      UNREACHABLE();
    }
  }
}

void FilePathWhitelist::clear() {
  whitelisted_import_paths_.clear();
  whitelisted_export_paths_.clear();
}

std::vector<std::string> FilePathWhitelist::whitelisted_import_paths_{};
std::vector<std::string> FilePathWhitelist::whitelisted_export_paths_{};

void FilePathBlacklist::addToBlacklist(const std::string& path) {
  CHECK(!path.empty());
  blacklisted_paths_.emplace_back(path);
}

bool FilePathBlacklist::isBlacklistedPath(const std::string& path) {
  const auto canonical_path = boost::filesystem::canonical(path).string();
  for (const auto& blacklisted_path : blacklisted_paths_) {
    std::string full_path;
    try {
      full_path = boost::filesystem::canonical(blacklisted_path).string();
    } catch (...) {
      /**
       * boost::filesystem::canonical throws an exception if provided path
       * does not exist. This may happen for use cases like license path
       * where the path may not necessarily contain a file. Fallback to
       * boost::filesystem::absolute in this case.
       */
      full_path = boost::filesystem::absolute(blacklisted_path).string();
    }
    if (boost::istarts_with(canonical_path, full_path)) {
      return true;
    }
  }
  return false;
}

void FilePathBlacklist::clear() {
  blacklisted_paths_.clear();
}

std::vector<std::string> FilePathBlacklist::blacklisted_paths_{};
}  // namespace ddl_utils
