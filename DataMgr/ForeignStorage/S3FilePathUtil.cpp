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
    const std::optional<std::string>& filter_regex,
    const std::optional<std::string>& sort_by,
    const std::optional<std::string>& sort_regex) {
  auto result_files = filter_regex.has_value()
                          ? s3_objects_regex_file_filter(filter_regex.value(), file_paths)
                          : file_paths;
  // initial lexicographical order ensures a determinisitc ordering for files not matching
  // sort_regex
  auto initial_file_order = FileOrderS3(std::nullopt, shared::PATHNAME_ORDER_TYPE);
  auto lexi_comp = initial_file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), lexi_comp);

  auto file_order = FileOrderS3(sort_regex, sort_by);
  auto comp = file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), comp);
  return result_files;
}
}  // namespace foreign_storage