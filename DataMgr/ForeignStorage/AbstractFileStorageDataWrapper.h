#pragma once

#include "ForeignDataWrapper.h"

namespace foreign_storage {
class AbstractFileStorageDataWrapper : public ForeignDataWrapper {
 public:
  AbstractFileStorageDataWrapper();

  void validateServerOptions(const ForeignServer* foreign_server) const override;

  void validateTableOptions(const ForeignTable* foreign_table) const override;

  const std::set<std::string_view>& getSupportedTableOptions() const override;

  void validateUserMappingOptions(const UserMapping* user_mapping,
                                  const ForeignServer* foreign_server) const override;

  const std::set<std::string_view>& getSupportedUserMappingOptions() const override;

  inline static const std::string STORAGE_TYPE_KEY = "STORAGE_TYPE";
  inline static const std::string BASE_PATH_KEY = "BASE_PATH";
  inline static const std::string FILE_PATH_KEY = "FILE_PATH";
  inline static const std::string REGEX_PATH_FILTER_KEY = "REGEX_PATH_FILTER";
  inline static const std::string LOCAL_FILE_STORAGE_TYPE = "LOCAL_FILE";
  inline static const std::string S3_STORAGE_TYPE = "AWS_S3";

  inline static const std::array<std::string, 1> supported_storage_types{
      LOCAL_FILE_STORAGE_TYPE};

 protected:
  /**
  @brief Returns the path to the source file/dir of the table.  Depending on options
  this may result from a concatenation of server and table path options.
*/
  static std::string getFullFilePath(const ForeignTable* foreign_table);

 private:
  static void validateFilePath(const ForeignTable* foreign_table);

  // A valid path is a concatenation of the file_path and the base_path (for local
  // storage). One of the two must be present.
  static void validateFilePathOptionKey(const ForeignTable* foreign_table);

  static const std::set<std::string_view> supported_table_options_;
  static const std::set<std::string_view> supported_server_options_;
  static const std::set<std::string_view> supported_user_mapping_options_;
};
}  // namespace foreign_storage
