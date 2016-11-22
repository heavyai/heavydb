#ifndef RESULTSETTESTUTILS_H
#define RESULTSETTESTUTILS_H

#include "../QueryEngine/TargetValue.h"
#include "../Shared/TargetInfo.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

void fill_one_entry_baseline(int64_t* value_slots,
                             const int64_t v,
                             const std::vector<TargetInfo>& target_infos,
                             const bool empty = false,
                             const bool null_val = false);

size_t get_slot_count(const std::vector<TargetInfo>& target_infos);

template <class T>
inline T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

#endif  // RESULTSETTESTUTILS_H
