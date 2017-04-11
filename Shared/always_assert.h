#ifndef SHARED_ALWAYS_ASSERT_H
#define SHARED_ALWAYS_ASSERT_H

#define CHECK(cond) \
  {                 \
    if (!(cond)) {  \
      abort();      \
    }               \
  }

#define CHECK_EQ(ref, val) \
  {                        \
    if ((val) != (ref)) {  \
      abort();             \
    }                      \
  }

#define CHECK_GT(val, ref) \
  {                        \
    if ((val) <= (ref)) {  \
      abort();             \
    }                      \
  }

#endif  // SHARED_ALWAYS_ASSERT_H
