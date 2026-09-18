#ifndef HELPERS_TYPE_HELPERS_H
#define HELPERS_TYPE_HELPERS_H

#include <cstdint>
#include <type_traits>

namespace helpers {

template<std::uint8_t Bits>
using uint_t =
    typename std::conditional<
        (Bits > 64),
        std::false_type,
        typename std::conditional<
            (Bits > 32),
            std::uint64_t,
            typename std::conditional<
                (Bits > 16),
                std::uint32_t,
                typename std::conditional<
                    (Bits > 8),
                    std::uint16_t,
                    std::uint8_t>::type>::type>::type>::type;

template<class T>
class no_init_t
{
public:
    static_assert(std::is_fundamental<T>::value &&
                  std::is_arithmetic<T>::value,
                  "wrapped type must be a fundamental, numeric type");

    //do nothing
    constexpr no_init_t() noexcept {}

    //convertible from a T
    constexpr no_init_t(T value) noexcept: v_(value) {}

    //act as a T in all conversion contexts
    constexpr operator T () const noexcept { return v_; }

private:
    T v_;
};

} // namespace helpers

#endif /* HELPERS_TYPE_HELPERS_H */
