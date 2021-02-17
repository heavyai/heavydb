#include "AbstractFileStorageDataWrapper.h"

#include "Catalog/ForeignServer.h"
#include "Catalog/ForeignTable.h"
#include "ForeignDataWrapperShared.h"
#include "Shared/StringTransform.h"
#include "Shared/misc.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {
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
    ddl_utils::validate_allowed_file_path(getFullFilePath(foreign_table),
                                          ddl_utils::DataTransferType::IMPORT);
  }
}

/**
  @brief Returns the path to the source file/dir of the table.  Depending on options
  this may result from a concatenation of server and table path options.
*/
std::string AbstractFileStorageDataWrapper::getFullFilePath(
    const ForeignTable* foreign_table) {
  auto options_container = dynamic_cast<const OptionsContainer*>(foreign_table);
  auto file_path = options_container->getOption(FILE_PATH_KEY);
  std::optional<std::string> base_path{};
  auto foreign_server = foreign_table->foreign_server;
  auto storage_type = foreign_server->getOption(STORAGE_TYPE_KEY);
  CHECK(storage_type);

  if (*storage_type == LOCAL_FILE_STORAGE_TYPE) {
    base_path = foreign_server->getOption(BASE_PATH_KEY);
  }

  // If both base_path and file_path are present, then concatenate.  Otherwise we are just
  // taking the one as the path.  One of the two must exist, or we have failed validation.
  CHECK(file_path || base_path);
  const std::string separator{boost::filesystem::path::preferred_separator};
  return std::regex_replace(
      (base_path ? *base_path + separator : "") + (file_path ? *file_path : ""),
      std::regex{separator + "{2,}"},
      separator);
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
  auto options_container = dynamic_cast<const OptionsContainer*>(foreign_table);
  auto file_path = options_container->getOption(FILE_PATH_KEY);
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

const std::set<std::string_view> AbstractFileStorageDataWrapper::supported_table_options_{
    FILE_PATH_KEY};

const std::set<std::string_view>
    AbstractFileStorageDataWrapper::supported_server_options_{STORAGE_TYPE_KEY,
                                                              BASE_PATH_KEY};

const std::set<std::string_view>
    AbstractFileStorageDataWrapper::supported_user_mapping_options_{};
}  // namespace foreign_storage
