
#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

namespace threading_tbb {

using tbb::blocked_range;
using tbb::task_arena;
using tbb::task_group;
namespace this_task_arena {
using namespace tbb::this_task_arena;
}
extern tbb::task_arena g_tbb_arena;

template <typename... X>
void parallel_for(X&&... x) {
  this_task_arena::isolate([&] { tbb::parallel_for(std::forward<X>(x)...); });
}

template <typename... X>
auto parallel_reduce(X&&... x) -> decltype(tbb::parallel_reduce(std::forward<X>(x)...)) {
  return this_task_arena::isolate(
      [&] { return tbb::parallel_reduce(std::forward<X>(x)...); });
}

template <typename T>
struct tbb_packaged_task : tbb::task_group {
  T value_;
  tbb_packaged_task() : value_(T()) {}
};

template <>
struct tbb_packaged_task<void> : tbb::task_group {};

template <typename T>
struct future {
  std::unique_ptr<tbb_packaged_task<T>> task_;
  future() = default;
  future(future&&) = default;
  future(std::unique_ptr<tbb_packaged_task<T>>&& p) : task_(std::move(p)) {}
  void wait() {
    g_tbb_arena.execute([this] { task_->wait(); });
  }
  T& get() {
    wait();
    return task_->value_;
  }
};

template <>
struct future<void> {
  std::unique_ptr<tbb_packaged_task<void>> task_;
  future() = default;
  future(future&&) = default;
  future(std::unique_ptr<tbb_packaged_task<void>>&& p) : task_(std::move(p)) {}
  void wait() {
    g_tbb_arena.execute([this] { task_->wait(); });
  }
  void get() { wait(); }
};

template <typename Fn,
          typename... Args,
          typename Result = std::result_of_t<Fn && (Args && ...)>>
future<Result> async(Fn&& fn, Args&&... args) {
  auto f = std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);
  auto ptask = std::make_unique<tbb_packaged_task<Result>>();
#if TBB_INTERFACE_VERSION >= 12040
  g_tbb_arena.enqueue(ptask->defer(f));
#else
  g_tbb_arena.execute([&] { ptask->run(f); });
#endif
  return future<Result>(std::move(ptask));
}

}  // namespace threading_tbb
