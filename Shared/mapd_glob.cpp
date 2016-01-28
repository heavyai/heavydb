#include "mapd_glob.h"
#include <glob.h>
#include <string>
#include <vector>

std::vector<std::string> mapd_glob(const std::string& pattern) {
  std::vector<std::string> results;
  glob_t glob_result;
  ::glob(pattern.c_str(), GLOB_BRACE | GLOB_TILDE, nullptr, &glob_result);
  for (size_t i = 0; i < glob_result.gl_pathc; i++) {
    results.push_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  return results;
}
