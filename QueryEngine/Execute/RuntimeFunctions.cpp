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

extern "C"
void filter_and_count_template(const int8_t** byte_stream,
                               const int32_t* row_count_ptr,
                               int32_t* out) {
  auto row_count = *row_count_ptr;
  int64_t result = 0;
  for (int32_t pos = 0; pos < row_count; ++pos) {
    if (filter_placeholder(pos, byte_stream)) {
      ++result;
    }
  }
  *out = result;
}
