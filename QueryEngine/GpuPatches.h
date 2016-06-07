#include "Execute.h"
#include "../CudaMgr/CudaMgr.h"

inline bool need_patch_unnest_double(const SQLTypeInfo& ti, const bool is_maxwell, const bool mem_shared) {
  return is_maxwell && mem_shared && ti.is_fp() && ti.get_type() == kDOUBLE;
}

inline std::string patch_agg_fname(const std::string& agg_name) {
  const auto new_name = agg_name + "_slow";
  CHECK_EQ("agg_id_double_shared_slow", new_name);
  return new_name;
}
