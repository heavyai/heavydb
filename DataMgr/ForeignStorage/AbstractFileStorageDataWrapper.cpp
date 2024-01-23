#include "AbstractFileStorageDataWrapper.h"

#include <codecvt>
#include <locale>

#include "Catalog/ForeignServer.h"
#include "Catalog/ForeignTable.h"
#include "Shared/StringTransform.h"
#include "Shared/misc.h"
#include "Shared/thread_count.h"
#include "Utils/DdlUtils.h"

extern bool g_enable_s3_fsi;

namespace foreign_storage {

size_t get_num_threads(const ForeignTable& table) {
  auto num_threads = 0;
  if (auto opt = table.getOption(AbstractFileStorageDataWrapper::THREADS_KEY);
      opt.has_value()) {
    num_threads = std::stoi(opt.value());
  }
  return import_export::num_import_threads(num_threads);
}

AbstractFileStorageDataWrapper::AbstractFileStorageDataWrapper() {}

void AbstractFileStorageDataWrapper::validateServerOptions(
    const ForeignServer* foreign_server) const {
  const auto& options = foreign_server->options;
  for (const auto& entry : options) {
    if (!shared::contains(supported_server_options_, entry.first)) {
      throw std::runtime_error{"Invalid foreign server option \"" + entry.first +
                               "\". Option must be one of the following: " +
                               join(supported_server_options_, ", ") + "."};
    }
  }

  if (options.find(STORAGE_TYPE_KEY) == options.end()) {
    throw std::runtime_error{"Foreign server options must contain \"" + STORAGE_TYPE_KEY +
                             "\"."};
  }
  const auto& storage_type = options.find(STORAGE_TYPE_KEY)->second;
  if (!shared::contains(supported_storage_types, storage_type)) {
    throw std::runtime_error{"Invalid \"" + STORAGE_TYPE_KEY +
                             "\" option value. Value must be one of the following: " +
                             join(supported_storage_types, ", ") + "."};
  }

  if (!g_enable_s3_fsi && storage_type == S3_STORAGE_TYPE) {
    throw std::runtime_error{
        "Foreign server storage type value of \"" + std::string{S3_STORAGE_TYPE} +
        "\" is not allowed because FSI S3 support is currently disabled."};
  }
}

void AbstractFileStorageDataWrapper::validateTableOptions(
    const ForeignTable* foreign_table) const {
  validateFilePathOptionKey(foreign_table);
  validateFilePath(foreign_table);
  shared::validate_sort_options(getFilePathOptions(foreign_table));
  validateFileRollOffOption(foreign_table);
}

const std::set<std::string_view>&
AbstractFileStorageDataWrapper::getSupportedTableOptions() const {
  return supported_table_options_;
}

void AbstractFileStorageDataWrapper::validateUserMappingOptions(
    const UserMapping* user_mapping,
    const ForeignServer* foreign_server) const {
  throw std::runtime_error{"User mapping for the \"" + foreign_server->data_wrapper_type +
                           "\" data wrapper can only be created for AWS S3 backed "
                           "foreign servers. AWS S3 support is currently disabled."};
}

const std::set<std::string_view>&
AbstractFileStorageDataWrapper::getSupportedUserMappingOptions() const {
  return supported_user_mapping_options_;
}

void AbstractFileStorageDataWrapper::validateFilePath(const ForeignTable* foreign_table) {
  auto& server_options = foreign_table->foreign_server->options;
  if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
    ddl_utils::validate_allowed_file_path(
        getFullFilePath(foreign_table), ddl_utils::DataTransferType::IMPORT, true);
  }
}

namespace {
std::string append_file_path(const std::optional<std::string>& base,
                             const std::optional<std::string>& subdirectory) {
#ifdef _WIN32
  const std::wstring str_to_cov{boost::filesystem::path::preferred_separator};
  using convert_type = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_type, wchar_t> converter;
  std::string separator = converter.to_bytes(str_to_cov);
#else
  const std::string separator{boost::filesystem::path::preferred_separator};
#endif
  return std::regex_replace(
      (base ? *base + separator : "") + (subdirectory ? *subdirectory : ""),
      std::regex{separator + "{2,}"},
      separator);
}
}  // namespace

/**
  @brief Returns the path to the source file/dir of the table.  Depending on options
  this may result from a concatenation of server and table path options.
*/
std::string AbstractFileStorageDataWrapper::getFullFilePath(
    const ForeignTable* foreign_table) {
  auto file_path = foreign_table->getOption(FILE_PATH_KEY);
  std::optional<std::string> base_path{};
  auto foreign_server = foreign_table->foreign_server;
  auto storage_type = foreign_server->getOption(STORAGE_TYPE_KEY);
  CHECK(storage_type);

#ifdef _WIN32
  const std::wstring str_to_cov{boost::filesystem::path::preferred_separator};
  using convert_type = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_type, wchar_t> converter;
  std::string separator = converter.to_bytes(str_to_cov);
#else
  const std::string separator{boost::filesystem::path::preferred_separator};
#endif
  if (*storage_type == LOCAL_FILE_STORAGE_TYPE) {
    base_path = foreign_server->getOption(BASE_PATH_KEY);
  }

  // If both base_path and file_path are present, then concatenate.  Otherwise we are just
  // taking the one as the path.  One of the two must exist, or we have failed validation.
  CHECK(file_path || base_path);
  return append_file_path(base_path, file_path);
}

namespace {
void throw_file_path_error(const std::string_view& missing_path,
                           const std::string& table_name,
                           const std::string_view& file_path_key) {
  std::stringstream ss;
  ss << "No file_path found for Foreign Table \"" << table_name
     << "\". Table must have either set a \"" << file_path_key << "\" option, or its "
     << "parent server must have set a \"" << missing_path << "\" option.";
  throw std::runtime_error(ss.str());
}
}  // namespace

// A valid path is a concatenation of the file_path and the base_path (for local storage).
// One of the two must be present.
void AbstractFileStorageDataWrapper::validateFilePathOptionKey(
    const ForeignTable* foreign_table) {
  auto file_path = foreign_table->getOption(FILE_PATH_KEY);
  auto foreign_server = foreign_table->foreign_server;
  auto storage_type = foreign_server->getOption(STORAGE_TYPE_KEY);
  CHECK(storage_type) << "No storage type found in parent server. Server \""
                      << foreign_server->name << "\" is not valid.";
  if (!file_path) {
    if (*storage_type == LOCAL_FILE_STORAGE_TYPE) {
      if (!foreign_server->getOption(BASE_PATH_KEY)) {
        throw_file_path_error(BASE_PATH_KEY, foreign_table->tableName, FILE_PATH_KEY);
      }
    } else {
      UNREACHABLE() << "Unknown foreign storage type.";
    }
  }
}

shared::FilePathOptions AbstractFileStorageDataWrapper::getFilePathOptions(
    const ForeignTable* foreign_table) {
  return {
      foreign_table->getOption(AbstractFileStorageDataWrapper::REGEX_PATH_FILTER_KEY),
      foreign_table->getOption(AbstractFileStorageDataWrapper::FILE_SORT_ORDER_BY_KEY),
      foreign_table->getOption(AbstractFileStorageDataWrapper::FILE_SORT_REGEX_KEY)};
}

namespace {
std::optional<bool> get_file_roll_off_value(const ForeignTable* foreign_table) {
  auto option =
      foreign_table->getOption(AbstractFileStorageDataWrapper::ALLOW_FILE_ROLL_OFF_KEY);
  if (option.has_value()) {
    if (to_upper(option.value()) == "TRUE") {
      return true;
    } else if (to_upper(option.value()) == "FALSE") {
      return false;
    } else {
      return {};
    }
  }
  return false;
}
}  // namespace

void AbstractFileStorageDataWrapper::validateFileRollOffOption(
    const ForeignTable* foreign_table) {
  auto allow_file_roll_off = get_file_roll_off_value(foreign_table);
  if (allow_file_roll_off.has_value()) {
    if (allow_file_roll_off.value() && !foreign_table->isAppendMode()) {
      throw std::runtime_error{"The \"" + ALLOW_FILE_ROLL_OFF_KEY +
                               "\" option can only be set to 'true' for foreign tables "
                               "with append refresh updates."};
    }
  } else {
    throw std::runtime_error{
        "Invalid boolean value specified for \"" + ALLOW_FILE_ROLL_OFF_KEY +
        "\" foreign table option. Value must be either 'true' or 'false'."};
  }
}

bool AbstractFileStorageDataWrapper::allowFileRollOff(const ForeignTable* foreign_table) {
  auto allow_file_roll_off = get_file_roll_off_value(foreign_table);
  if (allow_file_roll_off.has_value()) {
    return allow_file_roll_off.value();
  } else {
    auto option = foreign_table->getOption(ALLOW_FILE_ROLL_OFF_KEY);
    CHECK(option.has_value());
    UNREACHABLE() << "Unexpected " << ALLOW_FILE_ROLL_OFF_KEY
                  << " value: " << option.value();
    return false;
  }
}

const std::set<std::string> AbstractFileStorageDataWrapper::getAlterableTableOptions()
    const {
  return {ALLOW_FILE_ROLL_OFF_KEY};
}

const std::set<std::string_view> AbstractFileStorageDataWrapper::supported_table_options_{
    FILE_PATH_KEY,
    REGEX_PATH_FILTER_KEY,
    FILE_SORT_ORDER_BY_KEY,
    FILE_SORT_REGEX_KEY,
    ALLOW_FILE_ROLL_OFF_KEY,
    THREADS_KEY};

const std::set<std::string_view>
    AbstractFileStorageDataWrapper::supported_server_options_{STORAGE_TYPE_KEY,
                                                              BASE_PATH_KEY};

const std::set<std::string_view>
    AbstractFileStorageDataWrapper::supported_user_mapping_options_{};
}  // namespace foreign_storage
