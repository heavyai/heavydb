#ifndef HELPERS_HPC_HELPERS_H
#define HELPERS_HPC_HELPERS_H

#include <cstdio>

// helper for gcc version check
#define GCC_VERSION (__GNUC__ * 10000                                          \
    + __GNUC_MINOR__ * 100                                                     \
    + __GNUC_PATCHLEVEL__)

// debug prinf
#ifndef NDEBUG
    #define STRINGIZE_DETAIL(x) #x
    #define STRINGIZE(x) STRINGIZE_DETAIL(x)
    #define debug_printf(fmt, ...)                                             \
        printf("[DEBUG] file " STRINGIZE(__FILE__)                             \
        ", line " STRINGIZE(__LINE__) ": " STRINGIZE(fmt) "\n",                \
        ##__VA_ARGS__);
#else
    #define debug_printf(fmt, ...)
#endif

// safe division
#ifndef SDIV
    #define SDIV(x,y)(((x)+(y)-1)/(y))
#endif

namespace helpers {

    inline
    float B2KB(std::size_t bytes) noexcept { return float(bytes)/1024.0; }

    inline
    float B2MB(std::size_t bytes) noexcept { return float(bytes)/1048576.0; }

    inline
    float B2GB(std::size_t bytes) noexcept { return float(bytes)/1073741824.0; }

    inline
    std::size_t KB2B(float kb) noexcept { return std::size_t(kb*1024); }

    inline
    std::size_t MB2B(float mb) noexcept { return std::size_t(mb*1048576); }

    inline
    std::size_t GB2B(float gb) noexcept { return std::size_t(gb*1073741824); }

} // namespace helpers

#endif /* HELPERS_HPC_HELPERS_H */
