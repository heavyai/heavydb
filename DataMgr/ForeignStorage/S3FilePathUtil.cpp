#include "DataMgr/ForeignStorage/S3FilePathUtil.h"

namespace foreign_storage {
namespace {
std::vector<Aws::S3::Model::Object> s3_objects_regex_file_filter(
    const std::string& pattern,
    const std::vector<Aws::S3::Model::Object>& objects_list) {
  boost::regex regex_pattern(pattern);
  std::vector<Aws::S3::Model::Object> matched_objects_list;
  for (const auto& object : objects_list) {
    if (boost::regex_match(object.GetKey(), regex_pattern)) {
      matched_objects_list.emplace_back(object);
    }
  }
  if (matched_objects_list.empty()) {
    shared::throw_no_filter_match(pattern);
  }
  return matched_objects_list;
}
}  // namespace

std::vector<Aws::S3::Model::Object> s3_objects_filter_sort_files(
    const std::vector<Aws::S3::Model::Object>& file_paths,
    const shared::FilePathOptions& options) {
  auto result_files =
      options.filter_regex.has_value()
          ? s3_objects_regex_file_filter(options.filter_regex.value(), file_paths)
          : file_paths;
  // initial lexicographical order ensures a determinisitc ordering for files not matching
  // sort_regex
  shared::FilePathOptions temp_options;
  temp_options.sort_by = shared::PATHNAME_ORDER_TYPE;
  auto initial_file_order = FileOrderS3(temp_options);
  auto lexi_comp = initial_file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), lexi_comp);

  auto file_order = FileOrderS3(options);
  auto comp = file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), comp);
  return result_files;
}
}  // namespace foreign_storage