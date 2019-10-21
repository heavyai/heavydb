#pragma once

#include <arrow/util/thread-pool.h>
#include <future>

namespace utils {
template <typename Fn, typename... Args, typename Result = std::result_of_t<Fn && (Args && ...)>>
std::future<Result> async(std::launch policy, Fn&& fn, Args&&... args) {
  auto pool = arrow::internal::GetCpuThreadPool();
  using PackagedTask = std::packaged_task<Result()>;
  auto task =
      PackagedTask(std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
  auto fut = task.get_future();

  auto st = pool->Spawn(arrow::internal::detail::packaged_task_wrapper<Result>(std::move(task)));
  if (!st.ok()) {
    // This happens when Submit() is called after Shutdown()
    std::cerr << st.ToString() << std::endl;
    std::abort();
  }
  return fut;
  // return utils::async(std::launch::async, std::forward<Fn>(fn),
  //                                                  std::forward<Args>(args)...);
  // return utils::async(policy, std::forward<Fn>(fn), std::forward<Args>(args)...);
}
}  // namespace utils
