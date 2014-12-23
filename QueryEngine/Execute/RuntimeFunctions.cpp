#include <algorithm>
#include <cstdint>
#include <limits>


extern "C" __attribute__((always_inline))
int64_t fixed_width_int64_decode(
    const int8_t* byte_stream,
    const int32_t byte_width,
    const int32_t pos) {
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


// query templates

extern "C"
int64_t filter_placeholder(const int32_t pos, const int8_t** byte_stream);
extern "C" int32_t pos_start();
extern "C" int32_t pos_step();

extern "C" __attribute__((noinline))
int32_t pos_start_impl() {
  return 0;
}

extern "C" __attribute__((noinline))
int32_t pos_step_impl() {
  return 1;
}

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

extern "C" int64_t agg_placeholder(int64_t* agg, const int32_t pos, const int8_t* byte_stream);

extern "C"
void filter_and_agg_template(const int8_t** byte_stream,
                             const int32_t* row_count_ptr,
                             const int64_t* agg_init_val,
                             int32_t* out) {
  auto row_count = *row_count_ptr;
  auto result = *agg_init_val;
  const int32_t start = pos_start();
  int32_t step = pos_step();
  for (int32_t pos = start; pos < row_count; pos += step) {
    if (filter_placeholder(pos, byte_stream)) {
      agg_placeholder(&result, pos, nullptr);
    }
  }
  out[start] = result;
}
