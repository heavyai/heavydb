//
// Created by mehmet on 2019-09-23.
//

#ifndef OMNISCI_CSVPARSERUTILS_H
#define OMNISCI_CSVPARSERUTILS_H

#include "CopyParams.h"

namespace Importer_NS {
class CsvParserUtils {
 public:
  static size_t find_beginning(const char* buffer,
                               size_t begin,
                               size_t end,
                               const CopyParams& copy_params);

  static size_t find_end(const char* buffer,
                         size_t size,
                         const CopyParams& copy_params,
                         unsigned int& num_rows_this_buffer);

  static const char* get_row(const char* buf,
                             const char* buf_end,
                             const char* entire_buf_end,
                             const Importer_NS::CopyParams& copy_params,
                             const bool* is_array,
                             std::vector<std::string>& row,
                             bool& try_single_thread);

  static void parseStringArray(const std::string& s,
                               const Importer_NS::CopyParams& copy_params,
                               std::vector<std::string>& string_vec);

  static const std::string trim_space(const char* field, const size_t len);
};
}  // namespace Importer_NS

#endif  // OMNISCI_CSVPARSERUTILS_H
