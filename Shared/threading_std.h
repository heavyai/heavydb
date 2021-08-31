#include <cassert>
#include <cstddef>
#include <future>
#include <type_traits>
#include <vector>
#include "thread_count.h"

#ifndef THREADING_STD_LAUNCH
#define THREADING_STD_LAUNCH async  // async or deferred
#endif

namespace threading_common {
class split {};

class auto_partitioner {};
// class static_partitioner;
// class affinity_partitioner;

//! A range over which to iterate.
template <typename Value>
class blocked_range {
 public:
  //! Type of a value
  /** Called a const_iterator for sake of algorithms that need to treat a blocked_range
      as an STL container. */
  using const_iterator = Value;

  //! Type for size of a range
  using size_type = std::size_t;

  //! Construct range over half-open interval [begin,end), with the given grainsize.
  blocked_range(Value begin_, Value end_ /*TODO , size_type grainsize_=1*/)
      : my_end(end_)
      , my_begin(begin_)  //, my_grainsize(grainsize_)
  {
    // assert( my_grainsize>0 && "grainsize must be positive" );
  }

  //! Beginning of range.
  const_iterator begin() const { return my_begin; }

  //! One past last value in range.
  const_iterator end() const { return my_end; }

  //! Size of the range
  /** Unspecified if end()<begin(). */
  size_type size() const {
    assert(!(end() < begin()) && "size() unspecified if end()<begin()");
    return size_type(my_end - my_begin);
  }

  //! The grain size for this range.
  size_type grainsize() const { return 1 /*my_grainsize*/; }

  //------------------------------------------------------------------------
  // Methods that implement Range concept
  //------------------------------------------------------------------------

  //! True if range is empty.
  bool empty() const { return !(my_begin < my_end); }

  //! True if range is divisible.
  /** Unspecified if end()<begin(). */
  bool is_divisible() const { return /*TODO my_grainsize<*/ size(); }

  //! Split range.
  /** The new Range *this has the second part, the old range r has the first part.
      Unspecified if end()<begin() or !is_divisible(). */
  blocked_range(blocked_range& r, split)
      : my_end(r.my_end)
      , my_begin(do_split(r, split()))
  // TODO , my_grainsize(r.my_grainsize)
  {
    // only comparison 'less than' is required from values of blocked_range objects
    assert(!(my_begin < r.my_end) && !(r.my_end < my_begin) &&
           "blocked_range has been split incorrectly");
  }

 private:
  /** NOTE: my_end MUST be declared before my_begin, otherwise the splitting constructor
   * will break. */
  Value my_end;
  Value my_begin;
  // TODO size_type my_grainsize;

  //! Auxiliary function used by the splitting constructor.
  static Value do_split(blocked_range& r, split) {
    assert(r.is_divisible() && "cannot split blocked_range that is not divisible");
    Value middle = r.my_begin + (r.my_end - r.my_begin) / 2u;
    r.my_end = middle;
    return middle;
  }
};
}  // namespace threading_common

namespace threading_std {

using std::future;
using namespace threading_common;
constexpr auto launch = std::launch::THREADING_STD_LAUNCH;

template <typename Fn,
          typename... Args,
          typename Result = std::result_of_t<Fn && (Args && ...)>>
future<Result> async(Fn&& fn, Args&&... args) {
  return std::async(launch, std::forward<Fn>(fn), std::forward<Args>(args)...);
}

class task_group {
  std::vector<future<void>> threads_;

 public:
  template <typename F>
  void run(F&& f) {
    threads_.emplace_back(async(std::forward<F>(f)));
  }
  void cancel() { /*not implemented*/
  }
  void wait() {  // TODO task_group_status ?
    for (auto& child : this->threads_)
      child.wait();
  }
};  // class task_group

//! Parallel iteration over range with default partitioner.
/** @ingroup algorithms **/
// template<typename Range, typename Body, typename Partitioner = auto_partitioner>
// void parallel_for( const Range& range, const Body& body, const Partitioner &p =
// Partitioner());

template <typename Int, typename Body, typename Partitioner = auto_partitioner>
void parallel_for(const blocked_range<Int>& range,
                  const Body& body,
                  const Partitioner& p = Partitioner()) {
  const Int worker_count = cpu_threads();
  std::vector<std::future<void>> worker_threads;
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
        std::async(launch, body, blocked_range<Int>(start_entry, end_entry)));
  }
  for (auto& child : worker_threads)
    child.wait();
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
        //#pragma ivdep
        //#pragma omp simd
        for (auto i = r.begin(), e = r.end(); i < e; i++)
          f(i);
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
  std::vector<std::future<Value>> worker_threads;
  worker_threads.reserve(worker_count);

  for (Int i = 0,
           start_entry = range.begin(),
           stop_entry = range.end(),
           stride = (range.size() + worker_count - 1) / worker_count;
       i < worker_count && start_entry < stop_entry;
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, stop_entry);
    // TODO grainsize?
    worker_threads.emplace_back(std::async(
        launch, real_body, blocked_range<Int>(start_entry, end_entry), Value{}));
  }
  Value v = identity;
  for (auto& child : worker_threads)
    v = reduction(v, child.get());
  return v;
}

}  // namespace threading_std
