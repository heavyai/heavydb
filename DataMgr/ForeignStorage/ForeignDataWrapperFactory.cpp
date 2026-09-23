/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ForeignDataWrapperFactory.h"

#include "CsvDataWrapper.h"
#if defined(HAVE_AWS_S3)
#include "DataMgr/ForeignStorage/S3SelectDataWrapper.h"
#endif
#include "DataMgr/ForeignStorage/RasterDataWrapper.h"
#include "ForeignDataWrapper.h"
#include "InternalCatalogDataWrapper.h"
#include "InternalExecutorStatsDataWrapper.h"
#include "InternalLogsDataWrapper.h"
#include "InternalMLModelMetadataDataWrapper.h"
#include "InternalMemoryStatsDataWrapper.h"
#include "InternalStorageStatsDataWrapper.h"
#ifdef EE_FSI_ODBC
#include "ODBC/OdbcDataWrapper.h"
#endif
#ifdef ENABLE_IMPORT_PARQUET
#include "ParquetDataWrapper.h"
#include "ParquetImporter.h"
#endif
#include "RegexParserDataWrapper.h"
#include "Shared/JsonUtils.h"
#include "Shared/SysDefinitions.h"
#include "Shared/file_path_util.h"
#include "Shared/misc.h"
#include "Shared/thread_count.h"

namespace {
std::string get_data_wrapper_type(const import_export::CopyParams& copy_params) {
  std::string data_wrapper_type;
  if (copy_params.source_type == import_export::SourceType::kDelimitedFile) {
    data_wrapper_type = foreign_storage::DataWrapperType::CSV;
  } else if (copy_params.source_type == import_export::SourceType::kRegexParsedFile) {
    data_wrapper_type = foreign_storage::DataWrapperType::REGEX_PARSER;
#ifdef ENABLE_IMPORT_PARQUET
  } else if (copy_params.source_type == import_export::SourceType::kParquetFile) {
    data_wrapper_type = foreign_storage::DataWrapperType::PARQUET;
#endif
#ifdef EE_FSI_ODBC
  } else if (copy_params.source_type == import_export::SourceType::kOdbc) {
    data_wrapper_type = foreign_storage::DataWrapperType::ODBC;
#endif
  } else if (copy_params.source_type == import_export::SourceType::kRasterFile) {
    data_wrapper_type = foreign_storage::DataWrapperType::RASTER;
  } else {
    UNREACHABLE();
  }
  return data_wrapper_type;
}
}  // namespace

namespace foreign_storage {
std::tuple<std::unique_ptr<foreign_storage::ForeignServer>,
           std::unique_ptr<foreign_storage::UserMapping>,
           std::unique_ptr<foreign_storage::ForeignTable>>
create_proxy_fsi_objects(const std::string& copy_from_source,
                         const import_export::CopyParams& copy_params,
                         const int db_id,
                         const TableDescriptor* table,
                         const int32_t user_id) {
  auto server = foreign_storage::ForeignDataWrapperFactory::createForeignServerProxy(
      db_id, user_id, copy_from_source, copy_params);

  CHECK(server);
  server->validate();

  auto user_mapping =
      foreign_storage::ForeignDataWrapperFactory::createUserMappingProxyIfApplicable(
          db_id, user_id, copy_from_source, copy_params, server.get());

  if (user_mapping) {
    user_mapping->validate(server.get());
  }

  auto foreign_table =
      foreign_storage::ForeignDataWrapperFactory::createForeignTableProxy(
          db_id, table, copy_from_source, copy_params, server.get());

  CHECK(foreign_table);
  foreign_table->validateOptionValues();

  return {std::move(server), std::move(user_mapping), std::move(foreign_table)};
}

std::tuple<std::unique_ptr<foreign_storage::ForeignServer>,
           std::unique_ptr<foreign_storage::UserMapping>,
           std::unique_ptr<foreign_storage::ForeignTable>>
create_proxy_fsi_objects(const std::string& copy_from_source,
                         const import_export::CopyParams& copy_params,
                         const TableDescriptor* table) {
  return create_proxy_fsi_objects(copy_from_source, copy_params, -1, table, -1);
}

}  // namespace foreign_storage

namespace {
#if defined(HAVE_AWS_S3)
std::string get_s3_bucket_from_uri(const std::string& uri) {
  const std::string bucket_prefix = "://";
  auto base_pos = uri.find(bucket_prefix);
  CHECK(base_pos != std::string::npos && base_pos < uri.length());
  auto start_pos = base_pos + bucket_prefix.length();
  auto end_pos = uri.find("/", start_pos);
  return uri.substr(start_pos, end_pos - start_pos);
}

std::string get_s3_key_from_uri(const std::string& uri) {
  const std::string bucket_prefix = "://";
  auto base_pos = uri.find(bucket_prefix);
  CHECK(base_pos != std::string::npos && base_pos < uri.length());
  auto bucket_start_pos = base_pos + bucket_prefix.length();
  auto bucket_end_pos = uri.find("/", bucket_start_pos);
  return uri.substr(bucket_end_pos + 1);
}
#endif

const foreign_storage::UserMapping* get_user_mapping(
    const int db_id,
    const foreign_storage::ForeignTable* foreign_table) {
  const auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  return catalog->getUserMapping(shared::kRootUserId, foreign_table->foreign_server->id);
}

bool is_valid_data_wrapper(const std::string& data_wrapper_type) {
  return
#ifdef ENABLE_IMPORT_PARQUET
      data_wrapper_type == foreign_storage::DataWrapperType::PARQUET ||
#endif
#ifdef EE_FSI_ODBC
      data_wrapper_type == foreign_storage::DataWrapperType::ODBC ||
#endif
      data_wrapper_type == foreign_storage::DataWrapperType::RASTER ||
      data_wrapper_type == foreign_storage::DataWrapperType::CSV ||
      data_wrapper_type == foreign_storage::DataWrapperType::REGEX_PARSER;
}

}  // namespace

namespace foreign_storage {

void validate_regex_parser_options(const import_export::CopyParams& copy_params) {
  if (copy_params.line_regex.empty()) {
    throw std::runtime_error{"Regex parser options must contain a line regex."};
  }
}

#ifdef EE_FSI_ODBC
void validate_odbc_options(const import_export::CopyParams& copy_params) {
  if (copy_params.sql_select.empty()) {
    throw std::runtime_error{"ODBC options must contain a SQL select statement."};
  }
  if (copy_params.sql_order_by.empty()) {
    throw std::runtime_error{"ODBC options must contain a SQL ORDER BY statement."};
  }
  if (copy_params.dsn.empty() && copy_params.connection_string.empty()) {
    throw std::runtime_error{
        "ODBC options must contain either a data source name or a connection string."};
  }
  if (!copy_params.dsn.empty() && !copy_params.connection_string.empty()) {
    throw std::runtime_error{
        "ODBC options must contain only one of data source name or connection string."};
  }
  validate_odbc_credential_options(copy_params.dsn,
                                   copy_params.connection_string,
                                   copy_params.credential_string,
                                   copy_params.username,
                                   copy_params.password,
                                   true);
  if ((!copy_params.username.empty() || !copy_params.password.empty()) &&
      copy_params.dsn.empty()) {
    throw std::runtime_error{
        "ODBC options must contain a data source name when username/password is used."};
  }
  if (!copy_params.credential_string.empty() && copy_params.connection_string.empty()) {
    throw std::runtime_error{
        "ODBC options must contain a connection string when a credential string is "
        "used."};
  }
}
#endif

bool is_valid_source_type(const import_export::CopyParams& copy_params) {
  return
#ifdef ENABLE_IMPORT_PARQUET
      copy_params.source_type == import_export::SourceType::kParquetFile ||
#endif
#ifdef EE_FSI_ODBC
      copy_params.source_type == import_export::SourceType::kOdbc ||
#endif
      copy_params.source_type == import_export::SourceType::kRasterFile ||
      copy_params.source_type == import_export::SourceType::kDelimitedFile ||
      copy_params.source_type == import_export::SourceType::kRegexParsedFile;
}

std::string bool_to_option_value(const bool value) {
  return value ? "TRUE" : "FALSE";
}

std::unique_ptr<ForeignDataWrapper> ForeignDataWrapperFactory::createForGeneralImport(
    const import_export::CopyParams& copy_params,
    const int db_id,
    const ForeignTable* foreign_table,
    const UserMapping* user_mapping) {
  auto data_wrapper_type = get_data_wrapper_type(copy_params);
  CHECK(is_valid_data_wrapper(data_wrapper_type));

  if (data_wrapper_type == DataWrapperType::CSV) {
    return std::make_unique<CsvDataWrapper>(
        db_id, foreign_table, user_mapping, /*disable_cache=*/true);
  } else if (data_wrapper_type == DataWrapperType::REGEX_PARSER) {
    return std::make_unique<RegexParserDataWrapper>(
        db_id, foreign_table, user_mapping, true);
  }
#ifdef ENABLE_IMPORT_PARQUET
  else if (data_wrapper_type == DataWrapperType::PARQUET) {
    return std::make_unique<ParquetDataWrapper>(db_id,
                                                foreign_table,
                                                user_mapping,
                                                /*do_metadata_stats_validation=*/false);
  }
#endif
#ifdef EE_FSI_ODBC
  else if (data_wrapper_type == DataWrapperType::ODBC) {
    return std::make_unique<OdbcDataWrapper>(db_id, foreign_table, user_mapping);
  }
#endif
  else if (data_wrapper_type == DataWrapperType::RASTER) {
    return std::make_unique<RasterDataWrapper>(db_id, foreign_table, user_mapping);
  } else {
    UNREACHABLE() << "Datawrapper does not support import";
  }
  return {};
}

std::unique_ptr<ForeignDataWrapper> ForeignDataWrapperFactory::createForImport(
    const std::string& data_wrapper_type,
    const int db_id,
    const ForeignTable* foreign_table,
    const UserMapping* user_mapping) {
#ifdef ENABLE_IMPORT_PARQUET
  // only supported for parquet import path currently
  CHECK(data_wrapper_type == DataWrapperType::PARQUET);
  return std::make_unique<ParquetImporter>(db_id, foreign_table, user_mapping);
#else
  return {};
#endif
}

std::unique_ptr<UserMapping>
ForeignDataWrapperFactory::createUserMappingProxyIfApplicable(
    const int db_id,
    const int user_id,
    const std::string& file_path,
    const import_export::CopyParams& copy_params,
    const ForeignServer* server) {
#if defined(HAVE_AWS_S3)
  if (!shared::is_s3_uri(file_path) &&
      copy_params.source_type != import_export::SourceType::kOdbc) {
    return {};
  }

  OptionsMap options;

  if (copy_params.source_type == import_export::SourceType::kOdbc) {
    if (copy_params.username.empty() && copy_params.password.empty() &&
        copy_params.credential_string.empty()) {
      return {};
    }

#ifdef EE_FSI_ODBC
    if (!copy_params.username.empty()) {
      options[OdbcDataWrapper::ODBC_USERNAME] = copy_params.username;
    }

    if (!copy_params.password.empty()) {
      options[OdbcDataWrapper::ODBC_PASSWORD] = copy_params.password;
    }

    if (!copy_params.credential_string.empty()) {
      options[OdbcDataWrapper::ODBC_CREDENTIAL] = copy_params.credential_string;
    }
#endif

  } else if (shared::is_s3_uri(file_path)) {
    if (copy_params.s3_config.access_key.empty() &&
        copy_params.s3_config.secret_key.empty() &&
        copy_params.s3_config.session_token.empty()) {
      return {};
    }

    if (!copy_params.s3_config.access_key.empty()) {
      options[AbstractFileStorageDataWrapper::S3_ACCESS_KEY] =
          copy_params.s3_config.access_key;
    }
    if (!copy_params.s3_config.secret_key.empty()) {
      options[AbstractFileStorageDataWrapper::S3_SECRET_KEY] =
          copy_params.s3_config.secret_key;
    }
    if (!copy_params.s3_config.session_token.empty()) {
      options[AbstractFileStorageDataWrapper::S3_SESSION_TOKEN] =
          copy_params.s3_config.session_token;
    }
  }

  auto user_mapping = std::make_unique<UserMapping>();
  user_mapping->setOptions(options);
  user_mapping->foreign_server_id = -1;
  user_mapping->user_id = shared::kRootUserId;
  user_mapping->type = UserMappingType::PUBLIC;
  user_mapping->validate(server);

  return user_mapping;
#else
  return {};
#endif
}

std::unique_ptr<ForeignServer> ForeignDataWrapperFactory::createForeignServerProxy(
    const int db_id,
    const int user_id,
    const std::string& file_path,
    const import_export::CopyParams& copy_params) {
  CHECK(is_valid_source_type(copy_params));

  auto foreign_server = std::make_unique<foreign_storage::ForeignServer>();

  foreign_server->id = -1;
  foreign_server->user_id = user_id;
  if (copy_params.source_type == import_export::SourceType::kDelimitedFile) {
    foreign_server->data_wrapper_type = DataWrapperType::CSV;
  } else if (copy_params.source_type == import_export::SourceType::kRegexParsedFile) {
    foreign_server->data_wrapper_type = DataWrapperType::REGEX_PARSER;
#ifdef ENABLE_IMPORT_PARQUET
  } else if (copy_params.source_type == import_export::SourceType::kParquetFile) {
    foreign_server->data_wrapper_type = DataWrapperType::PARQUET;
#endif
#ifdef EE_FSI_ODBC
  } else if (copy_params.source_type == import_export::SourceType::kOdbc) {
    foreign_server->data_wrapper_type = DataWrapperType::ODBC;
#endif
  } else if (copy_params.source_type == import_export::SourceType::kRasterFile) {
    foreign_server->data_wrapper_type = DataWrapperType::RASTER;
  } else {
    UNREACHABLE();
  }
  foreign_server->name = "import_proxy_server";

  if (copy_params.source_type == import_export::SourceType::kOdbc) {
#ifdef EE_FSI_ODBC
    if (!copy_params.connection_string.empty()) {
      foreign_server->options[OdbcDataWrapper::ODBC_CONNECTION_KEY] =
          copy_params.connection_string;
    }
    if (!copy_params.dsn.empty()) {
      foreign_server->options[OdbcDataWrapper::ODBC_DSN_KEY] = copy_params.dsn;
    }
#endif
  } else if (shared::is_s3_uri(file_path)) {
#if defined(HAVE_AWS_S3)
    foreign_server->options[AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY] =
        AbstractFileStorageDataWrapper::S3_STORAGE_TYPE;
    foreign_server->options[AbstractFileStorageDataWrapper::AWS_REGION_KEY] =
        copy_params.s3_config.region;
    foreign_server->options[AbstractFileStorageDataWrapper::S3_BUCKET_KEY] =
        get_s3_bucket_from_uri(file_path);
    foreign_server->options[AbstractFileStorageDataWrapper::S3_ENDPOINT] =
        copy_params.s3_config.endpoint;
    foreign_server
        ->options[AbstractFileStorageDataWrapper::S3_USE_VIRTUAL_ADDRESSING_KEY] =
        copy_params.s3_config.use_virtual_addressing ? "TRUE" : "FALSE";
#else
    throw std::runtime_error("AWS storage not supported");
#endif
  } else {
    foreign_server->options[AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY] =
        AbstractFileStorageDataWrapper::LOCAL_FILE_STORAGE_TYPE;
  }

  return foreign_server;
}

namespace {
void set_header_option(OptionsMap& options,
                       const import_export::ImportHeaderRow& has_header) {
  switch (has_header) {
    case import_export::ImportHeaderRow::kNoHeader:
      options[CsvFileBufferParser::HEADER_KEY] = "FALSE";
      break;
    case import_export::ImportHeaderRow::kHasHeader:
    case import_export::ImportHeaderRow::kAutoDetect:
      options[CsvFileBufferParser::HEADER_KEY] = "TRUE";
      break;
    default:
      CHECK(false);
  }
}
}  // namespace

std::unique_ptr<ForeignTable> ForeignDataWrapperFactory::createForeignTableProxy(
    const int db_id,
    const TableDescriptor* table,
    const std::string& copy_from_source,
    const import_export::CopyParams& copy_params,
    const ForeignServer* server) {
  CHECK(is_valid_source_type(copy_params));

  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  auto foreign_table = std::make_unique<ForeignTable>();

  *static_cast<TableDescriptor*>(foreign_table.get()) =
      *table;  // copy table related values

  CHECK(server);
  foreign_table->foreign_server = server;

  if (copy_params.source_type == import_export::SourceType::kRasterFile) {
    if (copy_params.raster_width.has_value()) {
      foreign_table->options[RasterDataWrapper::RASTER_WIDTH_KEY] =
          std::to_string(copy_params.raster_width.value());
    }
    if (copy_params.raster_height.has_value()) {
      foreign_table->options[RasterDataWrapper::RASTER_HEIGHT_KEY] =
          std::to_string(copy_params.raster_height.value());
    }
    foreign_table->options[RasterDataWrapper::RASTER_FILTER_BANDS_KEY] =
        copy_params.raster_import_bands;
    foreign_table->options[RasterDataWrapper::RASTER_POINT_TRANSFORM_KEY] =
        to_string(copy_params.raster_point_transform);
    foreign_table->options[RasterDataWrapper::BOUNDING_BOX_CLIP_KEY] =
        to_string(copy_params.bounding_box_clip);
    foreign_table->options[RasterDataWrapper::RASTER_DROP_IF_ALL_NULL_KEY] =
        bool_to_option_value(copy_params.raster_drop_if_all_null);
  }

  // enable geo validation in most/all source types
  if (copy_params.source_type == import_export::SourceType::kRegexParsedFile ||
      copy_params.source_type == import_export::SourceType::kDelimitedFile ||
      copy_params.source_type == import_export::SourceType::kParquetFile ||
      copy_params.source_type == import_export::SourceType::kOdbc) {
    foreign_table->options[ForeignTable::GEO_VALIDATE_GEOMETRY_KEY] =
        bool_to_option_value(copy_params.geo_validate_geometry);
  }

  // populate options for regex filtering of file-paths in supported data types
  if (copy_params.source_type == import_export::SourceType::kRegexParsedFile ||
      copy_params.source_type == import_export::SourceType::kDelimitedFile ||
      copy_params.source_type == import_export::SourceType::kParquetFile) {
    if (copy_params.regex_path_filter.has_value()) {
      foreign_table->options[AbstractFileStorageDataWrapper::REGEX_PATH_FILTER_KEY] =
          copy_params.regex_path_filter.value();
    }
    if (copy_params.file_sort_order_by.has_value()) {
      foreign_table->options[AbstractFileStorageDataWrapper::FILE_SORT_ORDER_BY_KEY] =
          copy_params.file_sort_order_by.value();
    }
    if (copy_params.file_sort_regex.has_value()) {
      foreign_table->options[AbstractFileStorageDataWrapper::FILE_SORT_REGEX_KEY] =
          copy_params.file_sort_regex.value();
    }
    foreign_table->options[AbstractFileStorageDataWrapper::THREADS_KEY] =
        std::to_string(import_export::num_import_threads(copy_params.threads));
  }

  if (copy_params.source_type == import_export::SourceType::kRegexParsedFile) {
    CHECK(!copy_params.line_regex.empty());
    foreign_table->options[RegexFileBufferParser::LINE_REGEX_KEY] =
        copy_params.line_regex;
    if (!copy_params.line_start_regex.empty()) {
      foreign_table->options[RegexFileBufferParser::LINE_START_REGEX_KEY] =
          copy_params.line_start_regex;
    }
    if (copy_params.has_header != import_export::ImportHeaderRow::kAutoDetect) {
      set_header_option(foreign_table->options, copy_params.has_header);
    }
  }

  // setup data source options based on various criteria
  if (copy_params.source_type == import_export::SourceType::kOdbc) {
#ifdef EE_FSI_ODBC
    foreign_table->options[OdbcDataWrapper::ODBC_SELECT_KEY] = copy_params.sql_select;
    foreign_table->options[OdbcDataWrapper::ODBC_ORDER_BY_KEY] = copy_params.sql_order_by;
    foreign_table->options[OdbcDataWrapper::ODBC_BUFFER_SIZE_KEY] =
        std::to_string(copy_params.buffer_size);
#endif
  } else if (shared::is_s3_uri(copy_from_source)) {
#if defined(HAVE_AWS_S3)
    foreign_table->options["FILE_PATH"] = get_s3_key_from_uri(copy_from_source);
#else
    throw std::runtime_error("AWS storage not supported");
#endif
  } else {
    foreign_table->options["FILE_PATH"] = copy_from_source;
  }

  // for CSV import
  if (copy_params.source_type == import_export::SourceType::kDelimitedFile) {
    foreign_table->options[CsvFileBufferParser::DELIMITER_KEY] = copy_params.delimiter;
    foreign_table->options[CsvFileBufferParser::NULLS_KEY] = copy_params.null_str;
    set_header_option(foreign_table->options, copy_params.has_header);
    foreign_table->options[CsvFileBufferParser::QUOTED_KEY] =
        bool_to_option_value(copy_params.quoted);
    foreign_table->options[CsvFileBufferParser::QUOTE_KEY] = copy_params.quote;
    foreign_table->options[CsvFileBufferParser::ESCAPE_KEY] = copy_params.escape;
    foreign_table->options[CsvFileBufferParser::LINE_DELIMITER_KEY] =
        copy_params.line_delim;
    foreign_table->options[CsvFileBufferParser::ARRAY_DELIMITER_KEY] =
        copy_params.array_delim;
    const std::array<char, 3> array_marker{
        copy_params.array_begin, copy_params.array_end, 0};
    foreign_table->options[CsvFileBufferParser::ARRAY_MARKER_KEY] = array_marker.data();
    foreign_table->options[AbstractFileStorageDataWrapper::LONLAT_KEY] =
        bool_to_option_value(copy_params.lonlat);
    if (copy_params.geo_explode_collections) {
      throw std::runtime_error(
          "geo_explode_collections is not yet supported for FSI CSV import");
    }
    foreign_table->options[CsvFileBufferParser::GEO_EXPLODE_COLLECTIONS_KEY] =
        bool_to_option_value(copy_params.geo_explode_collections);
    foreign_table->options[CsvFileBufferParser::SOURCE_SRID_KEY] =
        std::to_string(copy_params.source_srid);

    foreign_table->options[TextFileBufferParser::BUFFER_SIZE_KEY] =
        std::to_string(copy_params.buffer_size);

    foreign_table->options[CsvFileBufferParser::TRIM_SPACES_KEY] =
        bool_to_option_value(copy_params.trim_spaces);
  }

  // for Parquet import
  if (copy_params.source_type == import_export::SourceType::kParquetFile) {
    foreign_table->options[AbstractFileStorageDataWrapper::LONLAT_KEY] =
        bool_to_option_value(copy_params.lonlat);
  }

  foreign_table->initializeOptions();
  return foreign_table;
}

std::unique_ptr<ForeignDataWrapper> ForeignDataWrapperFactory::create(
    const std::string& data_wrapper_type,
    const int db_id,
    const ForeignTable* foreign_table) {
  return create(
      data_wrapper_type, db_id, foreign_table, get_user_mapping(db_id, foreign_table));
}

std::unique_ptr<ForeignDataWrapper> ForeignDataWrapperFactory::create(
    const std::string& data_wrapper_type,
    const int db_id,
    const ForeignTable* foreign_table,
    const UserMapping* user_mapping) {
  std::unique_ptr<ForeignDataWrapper> data_wrapper;
  if (data_wrapper_type == DataWrapperType::CSV) {
    if (CsvDataWrapper::validateAndGetIsS3Select(foreign_table)) {
#if defined(HAVE_AWS_S3)
      data_wrapper =
          std::make_unique<S3SelectDataWrapper>(db_id, foreign_table, user_mapping);
#else
      UNREACHABLE();
#endif
    } else {
      data_wrapper = std::make_unique<CsvDataWrapper>(db_id, foreign_table, user_mapping);
    }
#ifdef ENABLE_IMPORT_PARQUET
  } else if (data_wrapper_type == DataWrapperType::PARQUET) {
    data_wrapper =
        std::make_unique<ParquetDataWrapper>(db_id, foreign_table, user_mapping);
#endif
#ifdef EE_FSI_ODBC
  } else if (data_wrapper_type == DataWrapperType::ODBC) {
    data_wrapper = std::make_unique<OdbcDataWrapper>(db_id, foreign_table, user_mapping);
#endif
  } else if (data_wrapper_type == DataWrapperType::RASTER) {
    data_wrapper =
        std::make_unique<RasterDataWrapper>(db_id, foreign_table, user_mapping);
  } else if (data_wrapper_type == DataWrapperType::REGEX_PARSER) {
    data_wrapper = std::make_unique<RegexParserDataWrapper>(
        db_id, foreign_table, get_user_mapping(db_id, foreign_table));
  } else if (data_wrapper_type == DataWrapperType::INTERNAL_CATALOG) {
    data_wrapper = std::make_unique<InternalCatalogDataWrapper>(db_id, foreign_table);
  } else if (data_wrapper_type == DataWrapperType::INTERNAL_EXECUTOR_STATS) {
    data_wrapper =
        std::make_unique<InternalExecutorStatsDataWrapper>(db_id, foreign_table);
  } else if (data_wrapper_type == DataWrapperType::INTERNAL_ML_MODEL_METADATA) {
    data_wrapper =
        std::make_unique<InternalMLModelMetadataDataWrapper>(db_id, foreign_table);
  } else if (data_wrapper_type == DataWrapperType::INTERNAL_MEMORY_STATS) {
    data_wrapper = std::make_unique<InternalMemoryStatsDataWrapper>(db_id, foreign_table);
  } else if (data_wrapper_type == DataWrapperType::INTERNAL_STORAGE_STATS) {
    data_wrapper =
        std::make_unique<InternalStorageStatsDataWrapper>(db_id, foreign_table);
  } else if (data_wrapper_type == DataWrapperType::INTERNAL_LOGS) {
    data_wrapper = std::make_unique<InternalLogsDataWrapper>(db_id, foreign_table);
  } else {
    throw std::runtime_error("Unsupported data wrapper");
  }
  return data_wrapper;
}

const ForeignDataWrapper* ForeignDataWrapperFactory::createForValidation(
    const std::string& data_wrapper_type,
    const ForeignTable* foreign_table) {
  bool is_s3_select_wrapper{false};
  std::string data_wrapper_type_key{data_wrapper_type};
  constexpr const char* S3_SELECT_WRAPPER_KEY = "CSV_S3_SELECT";
  if (foreign_table && data_wrapper_type == DataWrapperType::CSV &&
      CsvDataWrapper::validateAndGetIsS3Select(foreign_table)) {
    is_s3_select_wrapper = true;
    data_wrapper_type_key = S3_SELECT_WRAPPER_KEY;
  }

  auto [itr, is_new] = validation_data_wrappers_.emplace(data_wrapper_type_key, nullptr);
  if (is_new) {
    if (data_wrapper_type == DataWrapperType::CSV) {
      if (is_s3_select_wrapper) {
#if defined(HAVE_AWS_S3)
        CHECK_EQ(S3_SELECT_WRAPPER_KEY, data_wrapper_type_key);
        itr->second = std::make_unique<S3SelectDataWrapper>();
#else
        UNREACHABLE();
#endif
      } else {
        itr->second = std::make_unique<CsvDataWrapper>();
      }
#ifdef ENABLE_IMPORT_PARQUET
    } else if (data_wrapper_type == DataWrapperType::PARQUET) {
      itr->second = std::make_unique<ParquetDataWrapper>();
#endif
#ifdef EE_FSI_ODBC
    } else if (data_wrapper_type == DataWrapperType::ODBC) {
      itr->second = std::make_unique<OdbcDataWrapper>();
#endif
    } else if (data_wrapper_type == DataWrapperType::REGEX_PARSER) {
      itr->second = std::make_unique<RegexParserDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::INTERNAL_CATALOG) {
      itr->second = std::make_unique<InternalCatalogDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::INTERNAL_EXECUTOR_STATS) {
      itr->second = std::make_unique<InternalExecutorStatsDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::INTERNAL_ML_MODEL_METADATA) {
      itr->second = std::make_unique<InternalMLModelMetadataDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::INTERNAL_MEMORY_STATS) {
      itr->second = std::make_unique<InternalMemoryStatsDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::INTERNAL_STORAGE_STATS) {
      itr->second = std::make_unique<InternalStorageStatsDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::INTERNAL_LOGS) {
      itr->second = std::make_unique<InternalLogsDataWrapper>();
    } else if (data_wrapper_type == DataWrapperType::RASTER) {
      itr->second = std::make_unique<RasterDataWrapper>();
    } else {
      UNREACHABLE();
    }
  }
  return itr->second.get();
}

void ForeignDataWrapperFactory::validateDataWrapperType(
    const std::string& data_wrapper_type) {
  const auto& supported_wrapper_types = DataWrapperType::supported_data_wrapper_types;
  if (std::find(supported_wrapper_types.begin(),
                supported_wrapper_types.end(),
                data_wrapper_type) == supported_wrapper_types.end()) {
    std::vector<std::string_view> user_facing_wrapper_types;
    for (const auto& type : supported_wrapper_types) {
      if (!shared::contains(DataWrapperType::INTERNAL_DATA_WRAPPERS, type)) {
        user_facing_wrapper_types.emplace_back(type);
      }
    }
    throw std::runtime_error{"Invalid data wrapper type \"" + data_wrapper_type +
                             "\". Data wrapper type must be one of the following: " +
                             join(user_facing_wrapper_types, ", ") + "."};
  }
}

std::map<std::string, std::unique_ptr<ForeignDataWrapper>>
    ForeignDataWrapperFactory::validation_data_wrappers_;
}  // namespace foreign_storage
