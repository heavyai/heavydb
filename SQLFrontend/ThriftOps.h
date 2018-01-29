#ifndef THRIFTOPS_H
#define THRIFTOPS_H

#include "ThriftService.h"
#include "ThriftWithRetry.h"

namespace {
struct DoNothing {
  template <typename... T>
  void operator()(T&&... t) {}
};
}

template <ThriftService THRIFT_SERVICE,
          typename CONTEXT_TYPE,
          typename ON_SUCCESS_LAMBDA = DoNothing,
          typename ON_FAIL_LAMBDA = DoNothing>
bool thrift_op(CONTEXT_TYPE& context,
               ON_SUCCESS_LAMBDA success_op = ON_SUCCESS_LAMBDA(),
               ON_FAIL_LAMBDA fail_op = ON_FAIL_LAMBDA(),
               int const try_count = 1) {
  if (thrift_with_retry(THRIFT_SERVICE, context, nullptr, try_count)) {
    success_op(context);
    return true;
  }

  fail_op(context);
  return false;
}

template <ThriftService THRIFT_SERVICE,
          typename CONTEXT_TYPE,
          typename ON_SUCCESS_LAMBDA = DoNothing,
          typename ON_FAIL_LAMBDA = DoNothing>
bool thrift_op(CONTEXT_TYPE& context,
               char const* arg,
               ON_SUCCESS_LAMBDA success_op = ON_SUCCESS_LAMBDA(),
               ON_FAIL_LAMBDA fail_op = ON_FAIL_LAMBDA(),
               int const try_count = 1) {
  if (thrift_with_retry(THRIFT_SERVICE, context, arg, try_count)) {
    success_op(context);
    return true;
  }

  fail_op(context);
  return false;
}

#endif
