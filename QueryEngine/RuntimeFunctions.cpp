#include "RuntimeFunctions.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string.h>


// decoder implementations

extern "C" __attribute__((always_inline))
int64_t fixed_width_int_decode(
    const int8_t* byte_stream,
    const int32_t byte_width,
    const int64_t pos) {
  switch (byte_width) {
  case 1:
    return static_cast<int64_t>(byte_stream[pos * byte_width]);
  case 2:
    return *(reinterpret_cast<const int16_t*>(&byte_stream[pos * byte_width]));
  case 4:
    return *(reinterpret_cast<const int32_t*>(&byte_stream[pos * byte_width]));
  default:
    // TODO(alex)
    return std::numeric_limits<int64_t>::min() + 1;
  }
}

extern "C" __attribute__((always_inline))
int64_t diff_fixed_width_int_decode(
    const int8_t* byte_stream,
    const int32_t byte_width,
    const int64_t baseline,
    const int64_t pos) {
  return fixed_width_int_decode(byte_stream, byte_width, pos) + baseline;
}

// aggregator implementations

extern "C" __attribute__((always_inline))
void agg_count(int64_t* agg, const int64_t val) {
  ++*agg;;
}

extern "C" __attribute__((always_inline))
void agg_sum(int64_t* agg, const int64_t val) {
  *agg += val;
}

extern "C" __attribute__((always_inline))
void agg_max(int64_t* agg, const int64_t val) {
  *agg = std::max(*agg, val);
}

extern "C" __attribute__((always_inline))
void agg_min(int64_t* agg, const int64_t val) {
  *agg = std::min(*agg, val);
}

// placeholder functions -- either replaced by platform specific implementation
// at runtime or auto-generated

extern "C" int32_t pos_start();
extern "C" int32_t pos_step();
extern "C" void row_process(int64_t* out, const int64_t pos);

// x64 stride functions

extern "C" __attribute__((noinline))
int32_t pos_start_impl() {
  return 0;
}

extern "C" __attribute__((noinline))
int32_t pos_step_impl() {
  return 1;
}

// group by helpers

#define EMPTY_KEY std::numeric_limits<int64_t>::min()

extern "C"
void init_groups(int64_t* groups_buffer,
                 const int32_t groups_buffer_entry_count,
                 const int32_t key_qw_count,
                 const int64_t init_val) {
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + 1);
  for (int32_t i = 0; i < groups_buffer_entry_qw_count; ++i) {
    groups_buffer[i] = (i + 1) % (key_qw_count + 1) ? EMPTY_KEY : init_val;
  }
}

extern "C" __attribute__((always_inline))
int64_t* get_matching_group_value(int64_t* groups_buffer,
                                  const int32_t h,
                                  const int64_t* key,
                                  const int32_t key_qw_count) {
  auto off = h * (key_qw_count + 1);
  if (groups_buffer[off] == EMPTY_KEY) {
    memcpy(groups_buffer + off, key, key_qw_count * sizeof(*key));
    return groups_buffer + off + key_qw_count;
  }
  if (memcmp(groups_buffer + off, key, key_qw_count * sizeof(*key)) == 0) {
    return groups_buffer + off + key_qw_count;
  }
  return nullptr;
}

extern "C" __attribute__((always_inline))
int32_t key_hash(const int64_t* key, const int32_t key_qw_count, const int32_t groups_buffer_entry_count) {
  int32_t hash = 0;
  for (int32_t i = 0; i < key_qw_count; ++i) {
    hash = ((hash << 5) - hash + key[i]) % groups_buffer_entry_count;
  }
  return hash;
}

extern "C"
int64_t* get_group_value(int64_t* groups_buffer,
                         const int32_t groups_buffer_entry_count,
                         const int64_t* key,
                         const int32_t key_qw_count) {
  auto h = key_hash(key, key_qw_count, groups_buffer_entry_count);
  auto matching_group = get_matching_group_value(groups_buffer, h, key, key_qw_count);
  if (matching_group) {
    return matching_group;
  }
  auto h_probe = h + 1;
  while (h_probe != h) {
    matching_group = get_matching_group_value(groups_buffer, h_probe, key, key_qw_count);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  // TODO(alex): handle error by resizing?
  return nullptr;
}
