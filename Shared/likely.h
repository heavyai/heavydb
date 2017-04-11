#ifndef SHARED_LIKELY
#define SHARED_LIKELY
#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif  // SHARED_LIKELY
