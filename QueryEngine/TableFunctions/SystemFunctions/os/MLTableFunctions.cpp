#include "MLTableFunctions.hpp"

using namespace TableFunctions_Namespace;

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t supported_ml_frameworks__cpu_(TableFunctionManager& mgr,
                                      Column<TextEncodingDict>& output_ml_frameworks,
                                      Column<bool>& output_availability,
                                      Column<bool>& output_default) {
  const std::vector<std::string> ml_frameworks = {"onedal", "mlpack"};
  const int32_t num_frameworks = ml_frameworks.size();
  mgr.set_output_row_size(num_frameworks);
  const std::vector<int32_t> ml_framework_string_ids =
      output_ml_frameworks.string_dict_proxy_->getOrAddTransientBulk(ml_frameworks);

#if defined(HAVE_ONEDAL) || defined(HAVE_MLPACK)
  bool found_available_framework = false;
  auto framework_found_actions = [&output_availability,
                                  &output_default,
                                  &found_available_framework](const int64_t out_row_idx) {
    output_availability[out_row_idx] = true;
    if (!found_available_framework) {
      output_default[out_row_idx] = true;
      found_available_framework = true;
    } else {
      output_default[out_row_idx] = false;
    }
  };
#endif

#if !defined(HAVE_ONEDAL) || !defined(HAVE_MLPACK)
  auto framework_not_found_actions = [&output_availability,
                                      &output_default](const int64_t out_row_idx) {
    output_availability[out_row_idx] = false;
    output_default[out_row_idx] = false;
  };
#endif

  for (int32_t out_row_idx = 0; out_row_idx < num_frameworks; ++out_row_idx) {
    output_ml_frameworks[out_row_idx] = ml_framework_string_ids[out_row_idx];
    if (ml_frameworks[out_row_idx] == "onedal") {
#ifdef HAVE_ONEDAL
      framework_found_actions(out_row_idx);
#else
      framework_not_found_actions(out_row_idx);
#endif
    } else if (ml_frameworks[out_row_idx] == "mlpack") {
#ifdef HAVE_MLPACK
      framework_found_actions(out_row_idx);
#else
      framework_not_found_actions(out_row_idx);
#endif
    }
  }
  return num_frameworks;
}

#endif  // #ifndef __CUDACC__
