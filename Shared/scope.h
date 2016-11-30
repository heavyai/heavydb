#ifndef SHARED_SCOPE_H
#define SHARED_SCOPE_H

#include <functional>

class ScopeGuard {
 public:
  template <class Callable>
  ScopeGuard(Callable&& at_exit) : at_exit_(std::forward<Callable>(at_exit)) {}

  // make it non-copyable
  ScopeGuard(const ScopeGuard&) = delete;
  void operator=(const ScopeGuard&) = delete;

  ScopeGuard(ScopeGuard&& other) : at_exit_(std::move(other.at_exit_)) { other.at_exit_ = nullptr; }

  ~ScopeGuard() {
    if (at_exit_) {
      // note that at_exit_ must not throw
      at_exit_();
    }
  }

 private:
  std::function<void()> at_exit_;
};

#endif  // SHARED_SCOPE_H
