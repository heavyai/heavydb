#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifdef __CUDACC__
#define FORCE_INLINE __forceinline__
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifdef __CUDACC__
#define ALWAYS_INLINE
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#ifdef __CUDACC__
#define NEVER_INLINE
#else
#define NEVER_INLINE __attribute__((noinline))
#endif

#ifdef __CUDACC__
#define SUFFIX(name) name##_gpu
#else
#define SUFFIX(name) name
#endif
