#include "threading_std.h"

namespace threading_serial {

using namespace threading_common;
using std::future;

template <typename Fn,
          typename... Args,
          typename Result = std::result_of_t<Fn && (Args && ...)>>
future<Result> async(Fn&& fn, Args&&... args) {
  std::promise<Result> pr;
  if constexpr (std::is_same<void, Result>::value) {
    fn(std::forward<Args>(args)...);
    pr.set_value();
  } else {
    pr.set_value(fn(std::forward<Args>(args)...));
  }
  return pr.get_future();
}

class task_group {
 public:
  template <typename F>
  void run(F&& f) {
    f();
  }
  void cancel() { /*not implemented*/
  }
  void wait() {}
};  // class task_group

template <typename Int, typename Body, typename Partitioner = auto_partitioner>
void parallel_for(const blocked_range<Int>& range,
                  const Body& body,
                  const Partitioner& p = Partitioner()) {
  const Int worker_count = cpu_threads();

  for (Int i = 0,
           start_entry = range.begin(),
           stop_entry = range.end(),
           stride = (range.size() + worker_count - 1) / worker_count;
       i < worker_count && start_entry < stop_entry;
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, stop_entry);
    body(blocked_range<Int>(start_entry, end_entry));
  }
}

//! Parallel iteration over a range of integers with a default step value and default
//! partitioner
template <typename Index, typename Function, typename Partitioner = auto_partitioner>
void parallel_for(Index first,
                  Index last,
                  const Function& f,
                  const Partitioner& p = Partitioner()) {
  parallel_for(
      blocked_range<Index>(first, last),
      [&f](const blocked_range<Index>& r) {
        for (auto i = r.begin(), e = r.end(); i < e; i++) {
          f(i);
        }
      },
      p);
}

//! Parallel iteration with reduction
/** @ingroup algorithms **/
template <typename Int,
          typename Value,
          typename RealBody,
          typename Reduction,
          typename Partitioner = auto_partitioner>
Value parallel_reduce(const blocked_range<Int>& range,
                      const Value& identity,
                      const RealBody& real_body,
                      const Reduction& reduction,
                      const Partitioner& p = Partitioner()) {
  const size_t worker_count = cpu_threads();
  std::vector<Value> worker_threads;
  worker_threads.reserve(worker_count);

  for (Int i = 0,
           start_entry = range.begin(),
           stop_entry = range.end(),
           stride = (range.size() + worker_count - 1) / worker_count;
       i < worker_count && start_entry < stop_entry;
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, stop_entry);
    // TODO grainsize?
    worker_threads.emplace_back(
        real_body(blocked_range<Int>(start_entry, end_entry), Value{}));
  }
  Value v = identity;
  for (auto& child : worker_threads) {
    v = reduction(v, child);
  }

  return v;
}

}  // namespace threading_serial
