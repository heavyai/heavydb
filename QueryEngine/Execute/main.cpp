#include "Translator.h"

#include <glog/logging.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>


template<typename TimeT = std::chrono::milliseconds>
struct measure
{
  template<typename F, typename ...Args>
  static typename TimeT::rep execution(F func, Args&&... args)
  {
    auto start = std::chrono::system_clock::now();
    func(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);
    return duration.count();
  }
};

int main() {
  AggQueryCodeGenerator cgen(
    std::make_shared<OpGt>(
      std::make_shared<FetchInt64Col>(0, std::make_shared<FixedWidthInt64>(1)),
      std::make_shared<ImmInt64>(15)
    ),
    "filter_and_count_template",
    "filter_placeholder",
    "agg_placeholder",
    "pos_start",
    "pos_step",
    -1,
    std::make_shared<FixedWidthInt64>(1),
    "max"
  );

  int32_t N = 300 * 1000 * 1000;
  int8_t* byte_stream_col_0 = new int8_t[N];
  memset(byte_stream_col_0, 42, N);
  const int8_t* byte_stream[] = { byte_stream_col_0 };
  typedef int32_t (*agg_query)(const int8_t** byte_stream, const int32_t* row_count, int32_t* out);
  int32_t out;
  LOG(INFO) << measure<>::execution([&]() {
    reinterpret_cast<agg_query>(cgen.getNativeCode())(byte_stream, &N, &out);
    LOG(INFO) << out;
  });
  return 0;
}
