// TODO(alex): re-organize runtime to get rid of this file

#include <cstdint>

extern "C" void row_process(int64_t* out, const int64_t pos) {}
extern "C" int32_t pos_start() { return 0; };
extern "C" int32_t pos_step() { return 0; };
