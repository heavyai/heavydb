/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

namespace threading_tbb {

using tbb::blocked_range;
using tbb::task_arena;
#ifdef HAVE_TSAN
class task_group {
  class task_state {
   public:
    virtual ~task_state() = default;
    virtual void execute() = 0;

    void complete() { completed_.store(true, std::memory_order_release); }

    void acquireCompletion() const { (void)completed_.load(std::memory_order_acquire); }

   private:
    std::atomic<bool> completed_{false};
  };

  template <typename Function>
  class task_state_impl final : public task_state {
   public:
    explicit task_state_impl(Function function) : function_(std::move(function)) {}

    void execute() override { function_(); }

   private:
    Function function_;
  };

  struct task_trampoline {
    task_group* owner;
    task_state* state;

    // oneTBB is not TSAN-instrumented, so copying this two-pointer adapter through
    // its queue has no observable publication edge. The instrumented execute()
    // method establishes that edge before invoking the user callable.
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((no_sanitize_thread, noinline))
#endif
    void
    operator()() const {
      owner->execute(state);
    }
  };

 public:
  template <typename Function>
  void run(Function&& function) {
    using state_type = task_state_impl<std::decay_t<Function>>;
    auto state = std::make_unique<state_type>(std::forward<Function>(function));
    auto state_ptr = state.get();
    {
      std::lock_guard<std::mutex> lock(sync_mutex_);
      task_states_.emplace_back(std::move(state));
    }
    tasks_.run(task_trampoline{this, state_ptr});
  }

  void cancel() {
    tasks_.cancel();
  }

  void wait() {
    try {
      tasks_.wait();
    } catch (...) {
      releaseTaskStates();
      throw;
    }
    releaseTaskStates();
  }

 private:
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((noinline))
#endif
  void
  execute(task_state* state) {
    {
      // Pair with run()'s lock so TSAN observes publication of the callable
      // before the uninstrumented oneTBB scheduler invokes it.
      std::lock_guard<std::mutex> lock(sync_mutex_);
    }
    try {
      state->execute();
    } catch (...) {
      state->complete();
      throw;
    }
    state->complete();
  }

  void releaseTaskStates() {
    std::lock_guard<std::mutex> lock(sync_mutex_);
    // tasks_.wait() guarantees that no callable is still active. Acquire each
    // callable that ran; canceled callables remain false and published no output.
    for (const auto& state : task_states_) {
      state->acquireCompletion();
    }
    task_states_.clear();
  }

  std::mutex sync_mutex_;
  std::vector<std::unique_ptr<task_state>> task_states_;
  tbb::task_group tasks_;
};
#else
using tbb::task_group;
#endif
namespace this_task_arena {
using namespace tbb::this_task_arena;
}
extern tbb::task_arena g_tbb_arena;

#ifdef HAVE_TSAN
inline size_t tsan_worker_count(const size_t work_size) {
  const auto max_parallelism =
      tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
  return std::min(std::max<size_t>(max_parallelism, size_t(1)), work_size);
}

template <typename Int, typename Body, typename Partitioner = tbb::auto_partitioner>
void parallel_for(const blocked_range<Int>& range,
                  const Body& body,
                  const Partitioner& = Partitioner()) {
  if (range.empty()) {
    return;
  }
  const auto worker_count = tsan_worker_count(range.size());
  const auto stride = (range.size() + worker_count - 1) / worker_count;
  std::vector<std::future<void>> workers;
  workers.reserve(worker_count);
  for (auto start = range.begin(); start < range.end(); start += stride) {
    const auto end = std::min<Int>(start + stride, range.end());
    workers.emplace_back(std::async(
        std::launch::async, body, blocked_range<Int>(start, end, range.grainsize())));
  }
  for (auto& worker : workers) {
    worker.get();
  }
}

template <typename Index, typename Function, typename Partitioner = tbb::auto_partitioner>
void parallel_for(const Index first,
                  const Index last,
                  const Function& function,
                  const Partitioner& partitioner = Partitioner()) {
  threading_tbb::parallel_for(
      blocked_range<Index>(first, last),
      [&function](const blocked_range<Index>& range) {
        for (auto index = range.begin(); index < range.end(); ++index) {
          function(index);
        }
      },
      partitioner);
}
#else
template <typename... Args>
void parallel_for(Args&&... args) {
  g_tbb_arena.execute([&] {
    this_task_arena::isolate([&] { tbb::parallel_for(std::forward<Args>(args)...); });
  });
}
#endif

#ifdef HAVE_TSAN
template <typename Int,
          typename Value,
          typename Body,
          typename Reduction,
          typename Partitioner = tbb::auto_partitioner>
Value parallel_reduce(const blocked_range<Int>& range,
                      const Value& identity,
                      const Body& body,
                      const Reduction& reduction,
                      const Partitioner& = Partitioner()) {
  if (range.empty()) {
    return identity;
  }
  const auto worker_count = tsan_worker_count(range.size());
  const auto stride = (range.size() + worker_count - 1) / worker_count;
  std::vector<std::future<Value>> workers;
  workers.reserve(worker_count);
  for (auto start = range.begin(); start < range.end(); start += stride) {
    const auto end = std::min<Int>(start + stride, range.end());
    workers.emplace_back(std::async(std::launch::async,
                                    body,
                                    blocked_range<Int>(start, end, range.grainsize()),
                                    identity));
  }
  auto value = identity;
  for (auto& worker : workers) {
    value = reduction(value, worker.get());
  }
  return value;
}
#else
template <typename... Args>
auto parallel_reduce(Args&&... args)
    -> decltype(tbb::parallel_reduce(std::forward<Args>(args)...)) {
  return g_tbb_arena.execute([&] {
    return this_task_arena::isolate(
        [&] { return tbb::parallel_reduce(std::forward<Args>(args)...); });
  });
}
#endif

#ifdef HAVE_TSAN
using std::future;
#else
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
#endif

template <typename Fn,
          typename... Args,
          typename Result = std::invoke_result_t<Fn&&, Args&&...>>
future<Result> async(Fn&& fn, Args&&... args) {
#ifdef HAVE_TSAN
  return std::async(
      std::launch::async, std::forward<Fn>(fn), std::forward<Args>(args)...);
#else
  auto f = std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...);
  auto ptask = std::make_unique<tbb_packaged_task<Result>>();
#if TBB_INTERFACE_VERSION >= 12040
  g_tbb_arena.enqueue(ptask->defer(f));
#else
  g_tbb_arena.execute([&] { ptask->run(f); });
#endif
  return future<Result>(std::move(ptask));
#endif
}

}  // namespace threading_tbb
