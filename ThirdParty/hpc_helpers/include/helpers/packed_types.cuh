#ifndef HELPERS_PACKED_TYPES_CUH
#define HELPERS_PACKED_TYPES_CUH

#include <cstdint>
#include <climits>
#include <type_traits>
#include <cassert>

#include "cuda_helpers.cuh"
#include "type_helpers.h"

namespace packed_types {

using helpers::uint_t;

// INFO you can find the actual types as using statements at the end of this file

// bit-wise reinterpret one fundamental type as another fundamental type
template<class To, class From>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr To reinterpret_as(From from) noexcept
{
    static_assert(
        (std::is_fundamental<To>::value || std::is_enum<To>::value),
        "Target type must be fundamental enum.");

    static_assert(
        (std::is_fundamental<From>::value || std::is_enum<From>::value),
        "Input type must be fundamental or enum.");

    union reinterpreter_t
    {
        From from;
        To to;

        HOSTDEVICEQUALIFIER
        constexpr reinterpreter_t() noexcept : to(To()) {}
    } reinterpreter;

    // TODO add warning for narrowing conversions if desired
    reinterpreter.from = from;
    return reinterpreter.to;
}

namespace detail
{

template<
    std::uint8_t FirstBits,
    std::uint8_t SecondBits,
    std::uint8_t ThirdBits = 0,
    std::uint8_t FourthBits = 0>
class Pack
{
    // memory layout: MSB->padding|fourth|third|second|first<-LSB

public:
    using base_type = uint_t<FirstBits + SecondBits + ThirdBits + FourthBits>;

private:
    static_assert(
        FirstBits != 0 && SecondBits != 0,
        "FirstBits and SecondBits both may not be zero.");

    static_assert(
        !(ThirdBits == 0 && FourthBits != 0),
        "Third type cannot be zero-width if fourth type has non-zero width.");

    // leftover bits are padding
    static constexpr base_type PaddingBits =
        (sizeof(base_type) * CHAR_BIT) - (FirstBits + SecondBits + ThirdBits + FourthBits);

    // bit masks for each individual field
    static constexpr base_type first_mask = ((base_type{1} << FirstBits) - base_type{1});

    static constexpr base_type second_mask =
        ((base_type{1} << SecondBits) - base_type{1}) <<
            (FirstBits);

    static constexpr base_type third_mask =
        (ThirdBits == 0) ?
            base_type{0} :
            ((base_type{1} << ThirdBits) - base_type{1}) <<
                (FirstBits + SecondBits);

    static constexpr base_type fourth_mask =
        (FourthBits == 0) ?
            base_type{0} :
            ((base_type{1} << FourthBits) - base_type{1}) <<
                (FirstBits + SecondBits + ThirdBits);

    static constexpr base_type padding_mask =
        (PaddingBits == 0) ?
            base_type{0} :
            ((base_type{1} << PaddingBits) - base_type{1}) <<
                (FirstBits + SecondBits + ThirdBits + FourthBits);

public:
    // number of bits per field
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr std::uint8_t padding_bits() noexcept { return PaddingBits; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr std::uint8_t first_bits() noexcept { return FirstBits; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr std::uint8_t second_bits() noexcept { return SecondBits; }

    template<
        base_type B = ThirdBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr std::uint8_t third_bits() noexcept { return ThirdBits; }

    template<
        base_type B = FourthBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr std::uint8_t fourth_bits() noexcept { return FourthBits; }

    // CONSTRUCTORS
    HOSTDEVICEQUALIFIER
    constexpr explicit Pack() noexcept : base_{empty().base_} {}

    template<
        class FirstType,
        class SecondType,
        std::uint8_t B1 = ThirdBits,
        std::uint8_t B2 = FourthBits,
        class = std::enable_if_t<B1 == 0 && B2 == 0>>
    HOSTDEVICEQUALIFIER
    constexpr explicit Pack(
        FirstType first_,
        SecondType second_) noexcept : base_{empty().base_}
    {
        first(first_);
        second(second_);
    }

    template<
        class FirstType,
        class SecondType,
        class ThirdType,
        std::uint8_t B1 = ThirdBits,
        std::uint8_t B2 = FourthBits,
        class = std::enable_if_t<B1 != 0 && B2 == 0>>
    HOSTDEVICEQUALIFIER
    constexpr explicit Pack(
        FirstType first_,
        SecondType second_,
        ThirdType third_) noexcept : base_{empty().base_}
    {
        first(first_);
        second(second_);
        third(third_);
    }

    template<
        class FirstType,
        class SecondType,
        class ThirdType,
        class FourthType,
        std::uint8_t B1 = ThirdBits,
        std::uint8_t B2 = FourthBits,
        class = std::enable_if_t<B1 != 0 && B2 != 0>>
    HOSTDEVICEQUALIFIER
    constexpr explicit Pack(
        FirstType first_,
        SecondType second_,
        ThirdType third_,
        FourthType fourth_) noexcept : base_{empty().base_}
    {
        first(first_);
        second(second_);
        third(third_);
        fourth(fourth_);
    }

    constexpr Pack(const Pack&) noexcept = default;
    constexpr Pack(Pack&& pair) noexcept = default;

    // returns an empty pack
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Pack empty() noexcept
    {
        return Pack(base_type{0});
    }

    // SETTERS
    // by field name
    template<class First>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void first(First first_) noexcept
    {
        // TODO find a better solution to prevent truncation
        //static_assert(
        //    sizeof(First) <= sizeof(base_type),
        //    "Input type too wide. Truncation imminent.");

        first(reinterpret_as<base_type>(first_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void first(base_type first_) noexcept
    {
        assert(is_valid_first(first_));
        base_ = (base_ & ~first_mask) + (first_ & first_mask);
    }

    template<class Second>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void second(Second second_) noexcept
    {
        //static_assert(
        //    sizeof(Second) <= sizeof(base_type),
        //    "Input type too wide. Truncation imminent.");

        second(reinterpret_as<base_type>(second_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void second(base_type second_) noexcept
    {
        assert(is_valid_second(second_));
        constexpr auto shift = FirstBits;
        base_ = (base_ & ~second_mask) + ((second_ << shift) & second_mask);
    }

    template<
        class Third,
        std::uint8_t B = ThirdBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void third(Third third_) noexcept
    {
        //static_assert(
        //    sizeof(Third) <= sizeof(base_type),
        //    "Input type too wide. Truncation imminent.");

        third(reinterpret_as<base_type>(third_));
    }

    template<
        std::uint8_t B = ThirdBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void third(base_type third_) noexcept
    {
        assert(is_valid_third(third_));
        constexpr auto shift = FirstBits + SecondBits;
        base_ = (base_ & ~third_mask) + ((third_ << shift) & third_mask);
    }

    template<
        class Fourth,
        std::uint8_t B = FourthBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void fourth(Fourth fourth_) noexcept
    {
        //static_assert(
        //    sizeof(Fourth) <= sizeof(base_type),
        //    "Input type too wide. Truncation imminent.");

        fourth(reinterpret_as<base_type>(fourth_));
    }

    template<
        std::uint8_t B = FourthBits,
        class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void fourth(base_type fourth_) noexcept
    {
        assert(is_valid_fourth(fourth_));
        constexpr auto shift = FirstBits + SecondBits + ThirdBits;
        base_ = (base_ & ~fourth_mask) + ((fourth_ << shift) & fourth_mask);
    }

    // GETTERS
    // by field name
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T first_as() const noexcept
    {
        return reinterpret_as<T>(first());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr base_type first() const noexcept
    {
        return (base_ & first_mask);
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T second_as() const noexcept
    {
        return reinterpret_as<T>(second());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr base_type second() const noexcept
    {
        return ((base_ & second_mask) >> (FirstBits));
    }

    template<
        class T,
        std::uint8_t B = ThirdBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T third_as() const noexcept
    {
        return reinterpret_as<T>(third());
    }

    template<
        std::uint8_t B = ThirdBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr base_type third() const noexcept
    {
        return ((base_ & third_mask) >> (FirstBits + SecondBits));
    }

    template<
        class T,
        std::uint8_t B = FourthBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T fourth_as() const noexcept
    {
        return reinterpret_as<T>(fourth());
    }

    template<
        std::uint8_t B = FourthBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr base_type fourth() const noexcept
    {
        return ((base_ & fourth_mask) >> (FirstBits + SecondBits + ThirdBits));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr base_type base() const noexcept
    {
        return (base_ & ~padding_mask);
    }

    // SETTERS
    // set<index>(value)
    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, void>
    set(T first_) noexcept { first<T>(first_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, void>
    set(base_type first_) noexcept { first(first_); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, void>
    set(T second_) noexcept { second<T>(second_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, void>
    set(base_type second_) noexcept { second(second_); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, void>
    set(T third_) noexcept { third<T>(third_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, void>
    set(base_type third_) noexcept { third(third_); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, void>
    set(T fourth_) noexcept { fourth<T>(fourth_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, void>
    set(base_type fourth_) noexcept { fourth(fourth_); }

    // GETTERS
    // get<index>()
    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, T>
    get() const noexcept { return first_as<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, base_type>
    get() const noexcept { return first(); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, T>
    get() const noexcept { return second_as<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, base_type>
    get() const noexcept { return second(); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, T>
    get() const noexcept { return third_as<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, base_type>
    get() const noexcept { return third(); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, T>
    get() const noexcept { return fourth_as<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, base_type>
    get() const noexcept { return fourth(); }

    // INPUT VALIDATORS
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_first(T first_) noexcept
    {
        return is_valid_first(reinterpret_as<base_type>(first_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_first(base_type first_) noexcept
    {
        return !(first_ & ~((base_type{1} << FirstBits) - base_type{1}));
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_second(T second_) noexcept
    {
        return is_valid_second(reinterpret_as<base_type>(second_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_second(base_type second_) noexcept
    {
        return !(second_ & ~((base_type{1} << SecondBits) - base_type{1}));
    }

    template<
        class T,
        std::uint8_t B = ThirdBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_third(T third_) noexcept
    {
        return is_valid_third(reinterpret_as<base_type>(third_));
    }

    template<
        std::uint8_t B = ThirdBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_third(base_type third_) noexcept
    {
        return !(third_ & ~((base_type{1} << ThirdBits) - base_type{1}));
    }

    template<
        class T,
        std::uint8_t B = FourthBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_fourth(T fourth_) noexcept
    {
        return is_valid_fourth(reinterpret_as<base_type>(fourth_));
    }

    template<
        std::uint8_t B = FourthBits,
        class = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_fourth(base_type fourth_) noexcept
    {
        return !(fourth_ & ~((base_type{1} << FourthBits) - base_type{1}));
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 0, bool>
    is_valid(T first_) noexcept
    {
        return is_valid_first(first_);
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 1, bool>
    is_valid(T second_) noexcept
    {
        return is_valid_second(second_);
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 2 && ThirdBits, bool>
    is_valid(T third_) noexcept
    {
        return is_valid_third(third_);
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, bool>
    is_valid(T fourth_) noexcept
    {
        return is_valid_fourth(fourth_);
    }

    // OPERATORS
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Pack& operator=(const Pack& pack_) noexcept
    {
        base_ = (pack_.base_ & ~padding_mask);
        return *this;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator==(const Pack& pack_) const noexcept
    {
        return (base_ & ~padding_mask) == (pack_.base_ & ~padding_mask);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator!=(const Pack& pack_) const noexcept
    {
        return (base_ & ~padding_mask) != (pack_.base_ & ~padding_mask);
    }

    // CUDA ATOMICS
    DEVICEQUALIFIER INLINEQUALIFIER
    friend typename std::enable_if_t<
        (std::is_same<base_type, std::uint32_t>::value ||
         std::is_same<base_type, std::uint64_t>::value),
        Pack> atomicCAS(
        Pack * address_,
        Pack   compare_,
        Pack   val_) noexcept
    {
        return Pack(atomicCAS(&(address_->base_), compare_.base_, val_.base_));
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    friend typename std::enable_if_t<
        (std::is_same<base_type, std::uint32_t>::value ||
         std::is_same<base_type, std::uint64_t>::value),
        Pack> atomicExch(
        Pack * address_,
        Pack   val_) noexcept
    {
        return Pack(atomicExch(&(address_->base_), val_.base_));
    }

private:
    HOSTDEVICEQUALIFIER
    explicit constexpr Pack(base_type base) noexcept : base_{base} {}

    base_type base_;

}; // class Pack

} // namespace detail

// std::get support
template<
    std::size_t I,
    std::uint8_t B1,
    std::uint8_t B2,
    std::uint8_t B3,
    std::uint8_t B4>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr uint_t<B1 + B2 + B3 + B4> get(detail::Pack<B1, B2, B3, B4> pack) noexcept
{
    return pack.template get<I>();
}

// packed type aliases
template<std::uint8_t FirstBits, std::uint8_t SecondBits>
using PackedPair = detail::Pack<FirstBits, SecondBits>;

template<std::uint8_t FirstBits, std::uint8_t SecondBits, std::uint8_t ThirdBits>
using PackedTriple = detail::Pack<FirstBits, SecondBits, ThirdBits>;

template<std::uint8_t FirstBits, std::uint8_t SecondBits, std::uint8_t ThirdBits, std::uint8_t FourthBits>
using PackedQuadruple = detail::Pack<FirstBits, SecondBits, ThirdBits, FourthBits>;

} // namespace packed_types

#endif /* HELPERS_PACKED_TYPES_CUH */
