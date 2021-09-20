#include <optional>
#include <string>
#include <vector>

#include <aws/s3/model/Object.h>

// #include "DataMgr/OmniSciAwsSdk.h"
#include "Shared/DateTimeParser.h"
#include "Shared/file_path_util.h"

namespace foreign_storage {
std::vector<Aws::S3::Model::Object> s3_objects_filter_sort_files(
    const std::vector<Aws::S3::Model::Object>& file_paths,
    const std::optional<std::string>& filter_regex,
    const std::optional<std::string>& sort_by,
    const std::optional<std::string>& sort_regex);

using S3ObjectComparator =
    std::function<bool(const Aws::S3::Model::Object&, const Aws::S3::Model::Object&)>;
class FileOrderS3 : public shared::FileOrderBase<S3ObjectComparator> {
 public:
  FileOrderS3(const std::optional<std::string>& sort_regex,
              const std::optional<std::string>& sort_by)
      : FileOrderBase<S3ObjectComparator>(sort_regex, sort_by) {}

  virtual inline S3ObjectComparator getFileComparator() {
    auto comparator_pair = comparator_map_.find(getSortBy());
    CHECK(comparator_pair != comparator_map_.end());
    return comparator_pair->second;
  }

 protected:
  const std::map<std::string, S3ObjectComparator> comparator_map_{
      {shared::PATHNAME_ORDER_TYPE,
       [](const Aws::S3::Model::Object& lhs, const Aws::S3::Model::Object& rhs) -> bool {
         return lhs.GetKey() < rhs.GetKey();
       }},
      {shared::DATE_MODIFIED_ORDER_TYPE,
       [](const Aws::S3::Model::Object& lhs, const Aws::S3::Model::Object& rhs) -> bool {
         return lhs.GetLastModified() < rhs.GetLastModified();
       }},
      {shared::REGEX_ORDER_TYPE,
       [this](const Aws::S3::Model::Object& lhs,
              const Aws::S3::Model::Object& rhs) -> bool {
         auto lhs_name = lhs.GetKey();
         auto rhs_name = rhs.GetKey();
         return this->concatCaptureGroups(lhs_name) < this->concatCaptureGroups(rhs_name);
       }},
      {shared::REGEX_DATE_ORDER_TYPE,
       [this](const Aws::S3::Model::Object& lhs,
              const Aws::S3::Model::Object& rhs) -> bool {
         return shared::common_regex_date_comp_(this->concatCaptureGroups(lhs.GetKey()),
                                                this->concatCaptureGroups(rhs.GetKey()));
       }},
      {shared::REGEX_NUMBER_ORDER_TYPE,
       [this](const Aws::S3::Model::Object& lhs,
              const Aws::S3::Model::Object& rhs) -> bool {
         return shared::common_regex_number_comp_(
             this->concatCaptureGroups(lhs.GetKey()),
             this->concatCaptureGroups(rhs.GetKey()));
       }}};
};
}  // namespace foreign_storage