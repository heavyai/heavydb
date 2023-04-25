#ifndef WARPCORE_CHECKS_CUH
#define WARPCORE_CHECKS_CUH

namespace warpcore
{

/*! \brief assertion helpers
 */
namespace checks
{

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_valid_cg_size(index_t cgsize) noexcept
    {
        if(cgsize == 1  ||
           cgsize == 2  ||
           cgsize == 4  ||
           cgsize == 8  ||
           cgsize == 16 ||
           cgsize == 32)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_valid_key_type() noexcept
    {
        return (
            std::is_same<T, std::uint32_t>::value ||
            std::is_same<T, std::uint64_t>::value);
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_valid_slot_type() noexcept
    {
        return is_valid_key_type<T>();
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_valid_value_type() noexcept
    {
        return std::is_trivially_copyable<T>::value;
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_valid_counter_type() noexcept
    {
        return (
            std::is_same<T, int>::value ||
            std::is_same<T, std::uint32_t>::value ||
            std::is_same<T, std::uint64_t>::value);
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_hasher() noexcept
    {
        return
            std::is_same<typename T::tag, tags::hasher>::value ||
            std::is_same<typename T::tag, tags::true_permutation_hasher>::value;
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_true_permutation_hasher() noexcept
    {
        return std::is_same<typename T::tag, tags::true_permutation_hasher>::value;
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_probing_scheme() noexcept
    {
        return
            std::is_same<typename T::tag, tags::probing_scheme>::value ||
            std::is_same<typename T::tag, tags::cycle_free_probing_scheme>::value;
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_cycle_free_probing_scheme() noexcept
    {
        return std::is_same<typename T::tag, tags::cycle_free_probing_scheme>::value;
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_key_value_storage() noexcept
    {
        return std::is_same<typename T::tag, tags::key_value_storage>::value;
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_value_storage() noexcept
    {
        return
            std::is_same<typename T::tag, tags::static_value_storage>::value ||
            std::is_same<typename T::tag, tags::dynamic_value_storage>::value;
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool is_status_handler() noexcept
    {
        return std::is_same<typename T::tag, tags::status_handler>::value;
    }

} // namespace checks

} // namespace warpcore

#endif /* WARPCORE_CHECKS_CUH */