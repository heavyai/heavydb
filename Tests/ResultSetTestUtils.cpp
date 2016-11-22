#include "ResultSetTestUtils.h"
#include "../QueryEngine/ResultSetBufferAccessors.h"

void fill_one_entry_baseline(int64_t* value_slots,
                             const int64_t v,
                             const std::vector<TargetInfo>& target_infos,
                             const bool empty /* = false */,
                             const bool null_val /* = false */) {
  size_t target_slot = 0;
  int64_t vv = 0;
  for (const auto& target_info : target_infos) {
    bool isNullable = !target_info.sql_type.get_notnull();
    if ((isNullable && target_info.skip_null_val && null_val) || empty) {
      vv = null_val_bit_pattern(target_info.sql_type);
    } else {
      vv = v;
    }
    switch (target_info.sql_type.get_type()) {
      case kSMALLINT:
      case kINT:
      case kBIGINT:
        value_slots[target_slot] = vv;
        break;
      case kFLOAT:
      case kDOUBLE: {
        double di = vv;
        value_slots[target_slot] = null_val ? vv : *reinterpret_cast<int64_t*>(&di);
        break;
      }
      case kTEXT:
        value_slots[target_slot] = -(vv + 2);
        break;
      default:
        CHECK(false);
    }
    if (target_info.agg_kind == kAVG) {
      value_slots[target_slot + 1] = 1;
    }
    target_slot = advance_slot(target_slot, target_info);
  }
}

size_t get_slot_count(const std::vector<TargetInfo>& target_infos) {
  size_t count = 0;
  for (const auto& target_info : target_infos) {
    count = advance_slot(count, target_info);
  }
  return count;
}
