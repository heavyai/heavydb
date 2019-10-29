#pragma once

#include <arrow/util/thread-pool.h>
#include <future>

namespace utils {
template <typename Fn,
          typename... Args,
          typename Result = std::result_of_t<Fn && (Args && ...)>>
std::future<Result> async(Fn&& fn, Args&&... args) {
  // TODO: replace it with arrow::ThreadPool::Submit when arrow 0.4.2 release
  auto pool = arrow::internal::GetCpuThreadPool();
  using PackagedTask = std::packaged_task<Result()>;
  auto task = PackagedTask(std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
  auto fut = task.get_future();

  auto st = pool->Spawn(
      arrow::internal::detail::packaged_task_wrapper<Result>(std::move(task)));
  if (!st.ok()) {
    // This happens when Submit() is called after Shutdown()
    std::cerr << st.ToString() << std::endl;
    std::abort();
  }
  return fut;
}
}  // namespace utils
