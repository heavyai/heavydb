#ifndef WARPCORE_DEFAULTS_CUH
#define WARPCORE_DEFAULTS_CUH

namespace warpcore
{

/*! \brief default types and values
 */
namespace defaults
{
    template<class Key>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Key empty_key() noexcept { return Key{0}; }

    template<class Key>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Key tombstone_key() noexcept { return ~Key{0}; }

    template<class Key>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Key seed() noexcept { return Key{0x5ad0ded}; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr index_t temp_memory_bytes() noexcept { return 2048; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr index_t probing_length() noexcept { return ~index_t(0) - 1024; }

    template<class Key>
    using hasher_t = hashers::MurmurHash<Key>;

    template<class Key, index_t CGSize>
    using probing_scheme_t =
        probing_schemes::DoubleHashing<hasher_t<Key>, hasher_t<Key>, CGSize>;

    using status_handler_t = status_handlers::ReturnNothing;

    template<class Key, class Value>
    using table_storage_t = storage::key_value::AoSStore<Key, Value>;

    template<class Value>
    using value_storage_t = storage::multi_value::BucketListStore<Value>;

} // namespace defaults

} // namespace warpcore

#endif /* WARPCORE_DEFAULTS_CUH */