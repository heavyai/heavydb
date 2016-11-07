#ifndef SHARED_CUDA_GLOG_H
#define SHARED_CUDA_GLOG_H

#define CHECK(cond) \
  {                 \
    if (!cond) {    \
      abort();      \
    }               \
  }

#define CHECK_EQ(ref, val) \
  {                        \
    if (val != ref) {      \
      abort();             \
    }                      \
  }

#endif  // SHARED_CUDA_GLOG_H
