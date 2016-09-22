#ifndef SHARED_STRINGTRANSFORM_H
#define SHARED_STRINGTRANSFORM_H

#include <algorithm>
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>
#include <set>
#include <glog/logging.h>

inline std::string to_upper(const std::string& str) {
  auto str_uc = str;
  std::transform(str_uc.begin(), str_uc.end(), str_uc.begin(), ::toupper);
  return str_uc;
}

std::vector<std::pair<size_t, size_t>> find_string_literals(const std::string& query);

ssize_t inside_string_literal(const size_t start,
                              const size_t length,
                              const std::vector<std::pair<size_t, size_t>>& literal_positions);

void apply_shim(std::string& result,
                const boost::regex& reg_expr,
                const std::function<void(std::string&, const boost::smatch&)>& shim_fn);

#endif  // SHARED_STRINGTRANSFORM_H
